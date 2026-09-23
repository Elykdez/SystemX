# Desktop Minesweeper player

This optional player reads screen pixels and sends ordinary Windows mouse input.
It attaches to a visible game window, including a browser showing Minesweeper.
It uses the existing Minesweeper planner and optional Von model service.
The model server does not capture the screen or control the mouse.

The first implementation is a calibrated, profile-based player. It is not yet
an automatically universal recognizer. See the [test report](../../docs/desktop-visual-player-validation.md)
for what was actually tested and the remaining validation gaps.

## First setup

1. On a fresh checkout, run the root `setup.bat` for server dependencies and the model.
2. Run `addons/minesweeper/setup-desktop.bat` for the optional CPU vision packages.
3. Start the model with the root `start.bat` if you want Von decisions.
4. Open the game yourself, then run `addons/minesweeper/play-desktop.bat`.

Windows 10/11 and Python 3.12 are required. Desktop mode needs no Playwright,
browser extension, URL, or Codex plugin. The separate `play-minesweeper.bat`
continues to launch the existing DOM/Playwright player.

For a remote model server, pass `--server http://your-server:8000` to
`play-desktop.bat` and use the server's documented authentication/network setup.
Capture and mouse control always run on the machine displaying the game.
`--policy logic` explicitly runs without a model for comparison.

## Calibrate and play

1. Choose the game window and click **Capture**. This briefly activates the selected
   window to capture it, then brings back the controller preview. Keep the complete
   game window on screen and unobscured while it is being captured.
2. Enter rows, columns, and optionally the total mine count. The total is not the
   changing "mines remaining" counter. Leave it blank if unknown; automatic win
   detection then requires further support and may finish as stalled.
3. On a covered board, try **Suggest grid**. Otherwise drag from the board's top-left
   outer cell edge to its bottom-right outer cell edge in the preview.
4. Click **Check preview** and inspect the recognized cells. `.` means covered,
   `F` flagged, `*` mine, digits are revealed clues, and `?` means unknown.
5. Focus the game, release the mouse, and press **F9** for one move.
   If the result is correct, **F8** runs continuously. **F10** pauses.

The default profile is for the local hosted game's classic Segoe UI appearance.
It currently contains covered, empty, flag, mine, and digits 1-4. A new appearance
or a digit 5-8 can require teaching a sample before play continues. The player
pauses when a cell is unknown; it does not fill in missing labels with guesses.

To teach an appearance, capture the current board while paused, mark its grid,
click a sample cell in the preview, select its actual visible label, and click
**Teach cell**. Save the profile to preserve it. Use **New profile** for a different
theme, rather than mixing unrelated themes into one profile. Similarity scores
are recognition measurements, not calibrated probabilities.

Existing flags are treated as correct claims of mines by the planner. A new
attachment containing flags requires **Trust existing flags**. Prefer starting
with an unflagged board. The controller enables that option after its own
successful actions so stepping can continue through flags it placed.

Keep at least 24 pixels above or left of the board inside the game window. At
startup and after each click, the player parks the pointer there so hover/cursor highlights do not
cover clues. Do not operate the mouse at the same time as the player.

## Pausing and recovery

The controller pauses on focus loss, a covered cell, unexpected pointer movement,
changed geometry, uncertain recognition, contradictory clues, model errors, or
an unconfirmed click. It never repeats an unconfirmed click automatically.

If you move/resize the window, change browser zoom, switch difficulty, or change
theme, capture and calibrate again. F8 always requests a fresh decision; a result
from before a pause cannot authorize a new click. If inference is still finishing,
wait for the controller's result message before resuming.

F8/F9/F10 are global hotkeys; another program reserving them prevents this player
from starting. Close the controller to stop it. Your game stays open. The model
server stays running independently and retains VRAM until you stop it.

Only normal foreground mouse input is supported. Elevated windows, exclusive
fullscreen, unusual boards/rules, minimized play, and unattended rematches have
not been validated. Use a normal visible window with standard eight-neighbor rules.

## Saved calibration and measurements

**Save calibration** writes the current window/process identity, geometry, grid,
and a neighboring profile JSON file. It is for the same live window; reopening or
moving that window requires recalibration. From the project root:

```powershell
& ./addons/minesweeper/play-desktop.bat --config "C:/path/to/calibration.json" --record-vram
```

This starts a run after a three-second focus delay. F10 pauses; rerun the command
to start a new session. `--max-steps 1` limits it to a single verified move.
`--trust-flags` explicitly permits attaching to a flagged board.

Logs go under `.cache/minesweeper/desktop/<timestamp>/`: `actions.jsonl`,
`result.json`, and a final cropped board image. A pause report includes the last
attempted action when available. The GUI preview is a captured snapshot updated
during play, not a continuous paused video stream.

Calibration captures the complete window. During play, capture and preview focus
on the board at its original pixel resolution. The preview updates when board
pixels change. Teaching, mouse coordinates, and saved calibration retain their
original positions inside the game window.

Each action includes `timing_ms`: observation, decision (including HTTP),
revalidation, input (including the driver's final observation), verification,
and total. Use these timings to find the limiting stage on your machine.

The player checks matching frames at least 20 ms apart and proceeds immediately
after a verified move. Each check takes a fresh capture; a matching observation
from the preceding 100 ms can supply the earlier frame, reducing redundant captures.
Changed frames require a new matching pair. Clicks and observation errors discard
that stability history. Identical normalized cell images reuse cached classifications;
changed pixels are classified again, and teaching invalidates the cache. Fresh
capture, target checks, and click verification still run for every action.
An exactly unchanged board also reuses its recognition result, avoiding per-cell
processing and preview redraws. This comparison uses all board pixels; it does not
skip detecting changes or lower image quality.
Three to four clicks per second is the performance target, not a guaranteed rate:
rendering, hardware, animations, and recognition pauses affect actual throughput.
The recorded user run averaged about 3.1 moves/sec including startup before the
latest optimization. That optimization has replay measurements and regression
coverage; its live throughput still needs measuring. See the test report.

`--record-vram` adds `vram.jsonl` to saved-calibration runs. It samples total GPU
memory once a second through `nvidia-smi`, including other applications. It does
not import Torch or load a model. Missing NVIDIA tooling is recorded as an error
in that file rather than stopping play. For baseline/idle measurements:

```powershell
& ./.venv/Scripts/python.exe -m desktop.gpu_metrics --output .cache/vram.jsonl --phase baseline --seconds 10
```

The GUI itself does not start a GPU sampler. The opt-in test server
`python -m addons.minesweeper.tests.measure_server --output .cache/cuda.jsonl`
records exact Torch allocator statistics around real inference on port 8002;
it loads the real model, runs until stopped, and is intended only for measurement.

## Verification

After installing desktop dependencies, run `./.venv/Scripts/python.exe -m pytest -q`
from the root. Desktop tests skip if those optional packages are absent or the
platform is not Windows. Normal unit tests use screenshots and fake inputs, not
the model or actual mouse clicks. The measurement server is never started by pytest.
