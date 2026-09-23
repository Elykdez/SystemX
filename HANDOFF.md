# SystemOne handoff

Current workspace: `E:\Workspace\Python\SystemOne`.

## Minesweeper addon (2026-09-23)

The game-specific addon is `addons/minesweeper`, launched by
`addons/minesweeper/play-minesweeper.bat`; optional dependency setup is
`addons/minesweeper/setup-minesweeper.ps1`. Both locate the root environment
relative to their own folder. Root launchers are reserved for the model server.
See `addons/minesweeper/README.md` for controls, design, and test limits. Keep
future games in their own addon packages and preserve core inference code.

The real Von server and Chrome completed 19 moves after an inspection opening
and won Beginner (18 model selections, one forced move). Loss presentation was
checked separately. The complete CPU-only suite passes 100 tests. A standalone
separate-profile gameplay test remains unperformed; the live verification used
the ChatGPT Chrome extension. Evidence is in `.cache/minesweeper/verification`.
No inference server was left running. Core server/loader/download/watchdog and
core requirements hashes match the pre-addon versions.

Extracted on 2026-09-22 as an independent local Von HTTP server. Continue development in this directory. No changes were made to the source game project during extraction.

## Current contents

- Server, model adapter, optional Windows process-owner watchdog, download worker, launch/setup scripts, and tests.
- General usage and API documentation in `README.md`, with support-routing examples.
- A full independent copy of the pinned model cache under `.cache/huggingface`. No model re-download is required on this machine.
- A fresh Python 3.12.8 virtual environment at `.venv`. Its dependency libraries were copied from the existing working installation on this same machine, after stopping a redundant PyTorch wheel download. The environment uses its own files, not links or runtime imports into another project. Run `setup.ps1` to install normally on a new machine.
- Upstream attribution and the SDK license in `THIRD_PARTY_NOTICES.md` and `licenses/`.

The console/API title is now SystemOne. The warmup and smoke-test examples use customer support questions. Model identity, revision, core inference, endpoint paths, and environment-variable names are preserved.

## Verification

- 30 tests passed in this directory without real model inference. One upstream Starlette/AnyIO deprecation warning remains.
- `pip check` reported no broken requirements.
- PyTorch 2.9.0+cu126 and Von imported successfully from this directory's own `.venv`; no source-project path was present in Python's import search path.
- The CPU-only model check reported `checking`, then `ready`.
- Copied checkpoint SHA256: `52202f176080d1efe38ca62c45264894fc7e444d332f4af6bac86f4e3146e0f2`.
- Python source compilation, server CLI help, and PowerShell script parsing passed.
- Original server source-file hashes remained unchanged after extraction.

No real-model benchmark was run for this extraction. The performance figures in the README are previous measurements. No inference server is left running.

## Continue here

Double-click `start.bat` to launch in a visible server console, or run `./start.ps1` from this directory. Windows server launches show a console by default; background integrations can set `VON_SHOW_CONSOLE=0`. Poll `http://127.0.0.1:8000/health` and send requests after it reports ready. Stop with Ctrl+C. Use a different port if another local service already occupies 8000. Read `README.md` for CPU mode, offline cache use, network authentication, and ownership integration.

After adding the batch launcher and default visible console, 31 tests passed without model inference, including the default Windows console check.

The copied virtual environment and model cache are intentionally ignored by Git. The directory has not been initialized as a Git repository or published remotely.
