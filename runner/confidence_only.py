# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import csv
import torch
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs
from protenix.data.inference.json_to_feature import SampleDictToFeatures
from protenix.model import sample_confidence
from protenix.model.protenix import update_input_feature_dict
from protenix.utils.file_io import save_json
from protenix.utils.seed import seed_everything
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.utils.torch_utils import to_device
from runner.inference import InferenceRunner, update_v100_precision_config

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}
DNA3_TO_1 = {"DA": "A", "DC": "C", "DG": "G", "DT": "T", "DI": "I"}
RNA3_TO_1 = {"A": "A", "C": "C", "G": "G", "U": "U", "I": "I"}


def load_atom_array(structure_path: str) -> AtomArray:
    suffix = Path(structure_path).suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        cif_file = CIFFile.read(structure_path)
        atom_array = get_structure(cif_file, model=1)
    elif suffix == ".pdb":
        pdb_file = PDBFile.read(structure_path)
        atom_array = pdb_file.get_structure(model=1)
    else:
        raise ValueError(f"Unsupported structure format: {suffix}")
    return atom_array


def _chain_sequence(chain_atoms: AtomArray) -> tuple[str, str]:
    sequence = []
    seq_type = None
    for res_id in sorted(set(chain_atoms.res_id.tolist())):
        residue_atoms = chain_atoms[chain_atoms.res_id == res_id]
        res_name = residue_atoms.res_name[0].upper()
        if res_name in AA3_TO_1:
            cur_type = "proteinChain"
            aa = AA3_TO_1[res_name]
        elif res_name in DNA3_TO_1:
            cur_type = "dnaSequence"
            aa = DNA3_TO_1[res_name]
        elif res_name in RNA3_TO_1:
            cur_type = "rnaSequence"
            aa = RNA3_TO_1[res_name]
        else:
            raise ValueError(f"Unsupported residue type for score-only mode: {res_name}")
        if seq_type is None:
            seq_type = cur_type
        if seq_type != cur_type:
            raise ValueError(f"Mixed polymer types in one chain are not supported: {chain_atoms.chain_id[0]}")
        sequence.append(aa)
    return seq_type, "".join(sequence)


def build_sample_dict(atom_array: AtomArray, sample_name: str) -> tuple[dict[str, Any], dict[str, str]]:
    chain_ids = []
    for chain_id in atom_array.chain_id.tolist():
        if chain_id not in chain_ids:
            chain_ids.append(chain_id)

    sequences = []
    chain_map = {}
    for i, old_chain_id in enumerate(chain_ids):
        chain_atoms = atom_array[atom_array.chain_id == old_chain_id]
        seq_type, sequence = _chain_sequence(chain_atoms)
        sequences.append({seq_type: {"sequence": sequence, "count": 1}})
        new_chain_id = chr(ord("A") + i)
        chain_map[new_chain_id] = old_chain_id

    sample_dict = {"name": sample_name, "sequences": sequences}
    return sample_dict, chain_map


def map_coordinates(
    input_atom_array: AtomArray,
    model_atom_array: AtomArray,
    chain_map: dict[str, str],
    ref_pos: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    source_coord = {}
    for i in range(input_atom_array.array_length()):
        source_coord[(
            str(input_atom_array.chain_id[i]),
            int(input_atom_array.res_id[i]),
            str(input_atom_array.atom_name[i]),
        )] = input_atom_array.coord[i]

    coordinates = torch.zeros((model_atom_array.array_length(), 3), dtype=torch.float32)
    missing = 0
    for i in range(model_atom_array.array_length()):
        key = (
            chain_map[str(model_atom_array.chain_id[i])],
            int(model_atom_array.res_id[i]),
            str(model_atom_array.atom_name[i]),
        )
        if key in source_coord:
            coordinates[i] = torch.tensor(source_coord[key], dtype=torch.float32)
        else:
            coordinates[i] = ref_pos[i].to(torch.float32)
            missing += 1
    return coordinates, missing


def run_confidence_only(
    runner: InferenceRunner,
    input_feature_dict: dict[str, torch.Tensor],
    atom_coordinate: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    eval_precision = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[runner.configs.dtype]
    enable_amp = (
        torch.autocast(device_type="cuda", dtype=eval_precision)
        if torch.cuda.is_available()
        else nullcontext()
    )

    input_feature_dict = to_device(input_feature_dict, runner.device)
    atom_coordinate = atom_coordinate.to(runner.device)

    with torch.no_grad():
        with enable_amp:
            input_feature_dict = runner.model.relative_position_encoding.generate_relp(input_feature_dict)
            input_feature_dict = update_input_feature_dict(input_feature_dict)
            s_inputs, s, z = runner.model.get_pairformer_output(
                input_feature_dict=input_feature_dict,
                N_cycle=runner.model.N_cycle,
                inplace_safe=True,
                chunk_size=runner.configs.infer_setting.chunk_size,
            )
            contact_probs = sample_confidence.compute_contact_prob(
                distogram_logits=runner.model.distogram_head(z),
                **sample_confidence.get_bin_params(runner.configs.loss.distogram),
            )
            plddt, pae, pde, _ = runner.model.run_confidence_head(
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z,
                pair_mask=None,
                x_pred_coords=atom_coordinate.unsqueeze(0),
                triangle_multiplicative=runner.configs.triangle_multiplicative,
                triangle_attention=runner.configs.triangle_attention,
                inplace_safe=True,
                chunk_size=runner.configs.infer_setting.chunk_size,
            )

    summary_confidence, full_confidence = sample_confidence.compute_full_data_and_summary(
        configs=runner.configs,
        pae_logits=pae,
        plddt_logits=plddt,
        pde_logits=pde,
        contact_probs=contact_probs,
        token_asym_id=input_feature_dict["asym_id"],
        token_has_frame=input_feature_dict["has_frame"],
        atom_coordinate=atom_coordinate.unsqueeze(0),
        atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
        atom_is_polymer=1 - input_feature_dict["is_ligand"],
        N_recycle=runner.model.N_cycle,
        return_full_data=True,
    )

    ipsae = sample_confidence.compute_ipsae_from_token_pair_pae(
        token_pair_pae=full_confidence[0]["token_pair_pae"],
        token_asym_id=input_feature_dict["asym_id"],
    )
    summary_confidence[0]["ipsae"] = ipsae

    return summary_confidence[0], full_confidence[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protenix score-only confidence inference")
    parser.add_argument("--structure", type=str, required=True, help="Input PDB/CIF file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--load_checkpoint_path", type=str, default="./release_data/checkpoints")
    parser.add_argument("--model_name", type=str, default="protenix_base")
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    configs = parse_configs(
        configs={**configs_base, **data_configs, **inference_configs},
        arg_str=[
            f"--load_checkpoint_path={args.load_checkpoint_path}",
            f"--model_name={args.model_name}",
            "--use_msa=False",
            "--use_template=False",
            "--need_atom_confidence=True",
            f"--load_checkpoint_dir={args.load_checkpoint_path}",
        ],
    )
    configs.update(model_configs[configs.model_name])
    configs = update_v100_precision_config(configs)

    input_atom_array = load_atom_array(args.structure)
    sample_name = Path(args.structure).stem
    sample_dict, chain_map = build_sample_dict(input_atom_array, sample_name=sample_name)

    sample2feat = SampleDictToFeatures(sample_dict)
    features_dict, model_atom_array, _ = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = torch.tensor(model_atom_array.distogram_rep_atom_mask).long()
    features_dict = make_dummy_feature(features_dict=features_dict, dummy_feats=["template", "msa"])
    features_dict = data_type_transform(feat_or_label_dict=features_dict)
    atom_coordinate, missing_atom_count = map_coordinates(
        input_atom_array=input_atom_array,
        model_atom_array=model_atom_array,
        chain_map=chain_map,
        ref_pos=features_dict["ref_pos"],
    )

    runner = InferenceRunner(configs)
    summary_confidence, full_confidence = run_confidence_only(
        runner=runner,
        input_feature_dict=features_dict,
        atom_coordinate=atom_coordinate,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary_confidence.json")
    full_path = os.path.join(args.output_dir, "full_confidence.json")
    pae_matrix_path = os.path.join(args.output_dir, "pae_matrix.json")

    save_json(summary_confidence, summary_path, indent=4)
    save_json(full_confidence, full_path, indent=None)

    token_pair_pae = full_confidence["token_pair_pae"]
    if isinstance(token_pair_pae, torch.Tensor):
        token_pair_pae = token_pair_pae.detach().cpu().tolist()
    save_json(token_pair_pae, pae_matrix_path, indent=None)

    def to_scalar(v):
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return float(v.detach().cpu().item())
            return json.dumps(v.detach().cpu().tolist())
        if isinstance(v, (int, float)):
            return float(v)
        return str(v)

    summary_row = {k: to_scalar(v) for k, v in summary_confidence.items()}
    summary_row["sample"] = sample_name
    summary_row["missing_atom_count"] = missing_atom_count
    csv_path = os.path.join(args.output_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    print(
        json.dumps(
            {
                "summary": summary_path,
                "full": full_path,
                "pae_matrix": pae_matrix_path,
                "csv": csv_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
