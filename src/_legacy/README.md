# Legacy code (do not use)

This folder contains deprecated modules kept only for reference.

Rules:
- Nothing under `src/_legacy/` should be imported by runtime code.
- `src/` is the only supported runtime package.
- If you need something from here, port it explicitly into `src/` or `tools/` and delete/retire the legacy copy.

