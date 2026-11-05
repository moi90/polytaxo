# Design Spec: GUI Application Stub

* **Status**: Proposed
* **Component**: GUI

## 1. Context & Problem
A minimal GUI application entry point that can be launched and iterated on over time.

At this stage, the goal is not to build the full product, but to establish:
- A stable command-line interface.
- A minimal runnable NiceGUI app shell.
- Clear constraints for first usage (single dataset).

This gives us a predictable foundation for incremental feature development.

## 2. Proposed Solution
Implement a small NiceGUI-based application stub with two CLI flags:
- `--dev`: Enable NiceGUI development behavior (for example, auto-reload).
- `--dataset`: Path to a dataset folder for single-dataset annotation mode.

Behavior in this first version:
- Start the GUI application.
- Validate the dataset path when `--dataset` is provided.
- Run in single-dataset mode only.
- Keep internals intentionally simple and extensible.

## 3. Implementation Instructions

- Keep CLI parsing isolated from UI rendering.
- Avoid adding multi-dataset, multi-user, or auth concerns in this iteration.

## 4. Alternatives Considered (Optional)
- Use positional dataset argument instead of `--dataset`:
    - Rejected to keep room for future optional modes and additional flags.
- Start with multi-dataset support immediately:
    - Rejected because it adds complexity before basic app shell and CLI behavior are stable.

## 5. Outlook (Optional)
Expected incremental follow-ups:
- Initial annotation screen scaffold.