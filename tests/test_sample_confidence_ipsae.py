import torch

from protenix.model.sample_confidence import compute_ipsae_from_token_pair_pae


def test_compute_ipsae_from_token_pair_pae():
    pae = torch.tensor(
        [
            [0.0, 5.0, 10.0],
            [5.0, 0.0, 15.0],
            [10.0, 15.0, 0.0],
        ]
    )
    asym_id = torch.tensor([1, 1, 2])

    ipsae = compute_ipsae_from_token_pair_pae(pae, asym_id, pae_cutoff=10.0)

    expected = torch.tensor(
        [
            1.0 / (1.0 + 10.0 / 10.0),
            1.0 / (1.0 + 15.0 / 10.0),
            1.0 / (1.0 + 10.0 / 10.0),
            1.0 / (1.0 + 15.0 / 10.0),
        ]
    ).mean()
    assert torch.allclose(ipsae, expected)


def test_compute_ipsae_single_chain_is_zero():
    pae = torch.zeros((2, 2))
    asym_id = torch.tensor([1, 1])

    ipsae = compute_ipsae_from_token_pair_pae(pae, asym_id)

    assert ipsae.item() == 0.0
