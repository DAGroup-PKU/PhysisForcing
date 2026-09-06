# Vendored third-party model code (physics teachers)

The physics-alignment losses use three frozen teacher models. This directory
vendors the **model definition** of DepthAnythingV2 so the depth teacher can be
built without an extra pip dependency; only the architecture code lives here, the
weights are fetched separately by `tools/physics/vendor_teacher_assets.sh`.

## `depth_anything_v2/`

- Source: https://github.com/DepthAnything/Depth-Anything-V2
- Model code license: Apache-2.0.
- The pretrained *Depth-Anything-V2-Large* weights fetched by the vendor script
  are licensed **CC-BY-NC-4.0** (non-commercial).

## Teachers not vendored here

- **V-JEPA2** (relational teacher) — loaded through `transformers`
  (`VJEPA2Model`); weights: `facebook/vjepa2-vitl-fpc64-256`.
- **CoTracker3** (trajectory teacher) — loaded through `torch.hub` from
  https://github.com/facebookresearch/co-tracker (cloned by the vendor script).

See `tools/physics/vendor_teacher_assets.sh` for the exact fetch commands.
