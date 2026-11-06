# SKING_DDJ_v131

v131 adds background-invariant torso inference and conservative geometric checks for spurious outer pixels on the torso front/back. The approved body RGB and head calibration are retained. This is a parallel release: v130 remains available.

## Input and task contract

Input is a standardized **512×512 edited image**, `front_left` on the left and `back_left` on the right. Output is a classic **64×64 RGBA** skin. No source photograph, target UV, per-case annotations or provider metadata are required by UV extraction. The upstream Nano Banana request model, prompt and templates stay unchanged. Provider compatibility requires the same image size, camera layout and rendering conventions; arbitrary provider outputs are not automatically compatible.

```json
{"model_version":"SKING_DDJ_v131","dense_uv_checkpoint_file":"SKING_DDJ_v131/parser.pt","DMR_mappings_dir":"SKING_DDJ_v131/mappings"}
```

These paths resolve to `releases/SKING_DDJ_v131/`. Cross-version paths and failed asset hashes are rejected. The worker uses one resident version at a time; switching v130/v131 unloads the previous runtime. Continuous same-version requests reuse it.

## Runtime and deployment assets

The release is self-contained: encoder, body/head/refiner weights, torso adapter, face classifier, fixed geometry, background guard and runtime sources. Only inference tensors are exported. Weights and `.pt` geometry are distributed privately as hash-verified deployment assets, not stored in Git. Copy the complete release directory when migrating; startup verifies `release.json`. Set `SKING_DDJ_V131_PYTHON` to the deployment's CUDA Python executable. Use the versions listed in the manifest, with SciPy available for geometry.

`runtime/renderer.py` derives from EntropyDrop's differentiable_minecraft_renderer under AGPL-3.0 (see the repository LICENSE). The bundled DINOv2 source retains its own LICENSE and notices.

## Behavior and regression

Head outer threshold: 0.25. Torso normalization and adapter produce the candidate mask; a 0.999 whole-face gate and conservative geometry checks only remove front/back torso pixels. Retained colors and all other parts remain untouched by that final correction. The geometry gate abstains when evidence is insufficient. It is not a guarantee that every true outer pixel is preserved.

30 reviewed fixtures reproduce the accepted candidate pixel-for-pixel after independent export. These fixtures informed iteration and are not a fresh blind quality test. Worker tests cover both versions, strict paths, hash/audit failures, RGBA invariants, timeout cleanup and resident/one-shot behavior.

Use `scripts/export_semantic_release.py --help` for the inference-only export arguments. The exporter refuses to overwrite a finalized release. Production startup should keep `DENSE_UV_ALLOWED_MODEL_VERSIONS=SKING_DDJ_v130,SKING_DDJ_v131` until v130 is explicitly retired. Keep existing generation switches and v130 prices/availability unchanged.
