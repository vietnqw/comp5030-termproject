# Conventions for coding agents

- YAGNI: build only what the current experiment needs.
- DRY: one place for each piece of logic (labels, transforms, config loading).
- Keep the code simple; avoid unnecessary complexity and abstractions.
- Commit regularly with short commit messages.
- All paths come from the config or CLI flags; never hard-code cluster paths in Python.
- NIH ChestX-ray14 is the unseen test domain: never use it for training or model selection.
