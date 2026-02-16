# Pocket-constraint example (A/B receptor + C binder, D removed)

Updated based on user feedback:
- Removed chain D
- Removed the D->B203 pocket constraint
- Kept only one usable inference item: C near A203

## Entity mapping
- Chain A: `entity: 1`
- Chain B: `entity: 2`
- Chain C: `entity: 3`

## Input JSON
Use:
- `examples/example_pocket_abc_c_to_a203.json`

Constraint semantics in this file:
- `binder_chain = {"entity": 3, "copy": 1}` (chain C)
- `contact_residues = [{"entity": 1, "copy": 1, "position": 203}]` (A203)

## Run

```bash
protenix pred \
  -i examples/example_pocket_abc_c_to_a203.json \
  -o ./output_pocket_abc \
  -n protenix_base_constraint_v0.5.0 \
  --use_default_params true
```

## Note
- `max_distance: 8.0` is a moderate soft constraint.
- You can tighten to `6.0` if you want stronger guidance.
