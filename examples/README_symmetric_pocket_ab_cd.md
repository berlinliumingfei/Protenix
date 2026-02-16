# Symmetric pocket-constraint example (A/B receptor + C/D binder)

This example corresponds to four chains in one complex:
- Chain A: receptor copy 1 (`entity: 1`)
- Chain B: receptor copy 2 (`entity: 2`)
- Chain C: binder copy 1 (`entity: 3`)
- Chain D: binder copy 2 (`entity: 4`)

File: `examples/example_symmetric_pocket_ab_cd.json`

## Why there are 2 records in one JSON list

The current pocket format accepts one `binder_chain` per input item. Therefore, for symmetric constraints
(C near A203 and D near B203), this example provides **two inference items** in one JSON array:

1. `symmetric_abcd_c_to_a203`: constrain C (`entity=3`) near A203 (`entity=1, position=203`)
2. `symmetric_abcd_d_to_b203`: constrain D (`entity=4`) near B203 (`entity=2, position=203`)

## Run

```bash
protenix pred \
  -i examples/example_symmetric_pocket_ab_cd.json \
  -o ./output_symmetric_pocket \
  -n protenix_base_constraint_v0.5.0 \
  --use_default_params true
```

## Notes

- `max_distance` is set to `8.0` as a moderate soft-constraint strength.
- You can tighten to `6.0` for stronger guidance.
