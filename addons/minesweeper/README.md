# Minesweeper player

Terminology: [System 1 (System One)](../../docs/system-one-model.md) is the
decision-model role, currently StartLux through the HTTP API. The browser adapter
reads visible DOM labels; the [screen-pixel adapter](DESKTOP.md) uses OpenCV as System 0.
Game logic constructs candidates and the controller executes and verifies choices.

`play-minesweeper.bat` opens one inspector with two adapters. Both use the same
validated board representation, classic Minesweeper planner, StartLux decision
client, and observe/decide/revalidate/act/verify loop.

| Adapter | Reads | Controls | Setup |
| --- | --- | --- | --- |
| **Screen pixels** (default) | Captured cells using a taught OpenCV appearance profile | A selected visible Windows window, including a browser | Select window, calibrate grid, check recognition |
| **Browser labels** | Rendered accessibility labels and cell bounding boxes on a supported webpage | A separate Chrome instance through Playwright | Enter URL and difficulty, then Connect |

The dark inspector shows an annotated preview, status banner, and scrollable
board and decision panel. Wide windows put preview and state side by side;
narrow windows stack them. Verified moves report their decision source,
candidate count, reason, uncertainty, and timing.

Support covers classic rectangular boards with eight neighboring cells. Different
webpages may need another DOM observer; use screen pixels where visible cells
can be calibrated and taught. New themes or scales can require new samples.
Hexagonal boards and other rule variants need additional solver support.

This is a Minesweeper-specific addon. It does not alter the core server, model
loader, or HTTP API. Future games should have separate sibling addons with
their own logic, calling the same general inference service. This folder contains
the player, not the game's source or the model server.

## Separate components

| Component | Start | Stop |
| --- | --- | --- |
| Model server | `start-server.bat` in this folder (one int8 worker) | Ctrl+C in its console; this releases model VRAM. |
| Minesweeper inspector | `play-minesweeper.bat` in this folder | Pause/F10 stops input; close the inspector to exit. |
| Game | Open it yourself for screen pixels; Connect opens Chrome for browser labels. | Closing the inspector closes its browser-label session; an externally opened game stays open. |

The standalone player needs neither Codex nor the ChatGPT extension/plugin.
It shares the project's `.venv` for dependencies but talks to the model only
through HTTP. See the [server README](../../README.md) for server installation
and API details.

## Start

1. On a fresh installation, double-click the root `setup.bat`, then double-click
   `addons/minesweeper/setup-minesweeper.bat`. This installs both adapters' optional
   dependencies. Browser-label mode also requires installed Google Chrome.
2. Double-click `addons/minesweeper/start-server.bat` and wait for the SystemX server  to report ready.
3. Open `addons/minesweeper` and double-click `play-minesweeper.bat`.
4. Choose an adapter. For **Screen pixels**, follow the [calibration guide](DESKTOP.md).
   For **Browser labels**, enter the complete `http://` or `https://` URL, select
   difficulty, and click **Connect / Capture**. Connection selects the game's
   difficulty; it does not authorize playing moves.
5. Inspect the preview. Browser-label mode has **Run** and **Step** buttons.
   For screen pixels, focus the game and use **F8** to run or **F9** for one move.

The setup and player batch launchers locate the root environment themselves.
They work when double-clicked or launched from another working directory.
Setup keeps its console open so you can read the result. Run it once per
installation; it installs the addon dependencies without loading the model.

The inspector reports model or connection errors in its status banner. Wait until
the separate model server is ready before running with the `systemone` policy.
The explicit `logic` policy needs no model server.

The player opens installed Google Chrome with a separate profile under
`.cache/minesweeper/chrome-profile`, keeping everyday tabs separate. Only one
player may use this profile at a time. The standalone launcher does not need the
ChatGPT Chrome extension; that extension was used for development verification.

Certificate validation is not disabled. Browser connection errors and unknown
page structures stop the adapter with a message. Use a trusted game URL or a
calibrated screen-pixel attachment to a game you have opened yourself.

## Controls

| Control | Action |
| --- | --- |
| F8 | Run or resume; focus the game first in screen-pixel mode. |
| F9 | Perform one verified move, then pause. |
| F10 / **Pause** | Cancel pending input and pause the agent; the game itself continues. |
| **Run** / **Step** | Browser-label mode's alternatives to F8/F9. |
| **Trust existing flags** | Permit attaching to a flagged board; the solver assumes those flags are correct. |
| Close inspector | Stop control and close the browser-label session. |

Changing adapters cancels pending input and clears calibration and trusted flags.
If a worker is still finishing a decision, the inspector keeps the current
adapter and asks you to wait before switching. Recalibrate screen pixels after
moving or resizing its game window, changing zoom, or changing theme.

Input stops on a win, loss, step limit, model error, contradictory observation,
obscured target, or missing click result. There is no automatic rematch. After
normal completion the inspector stays open for review. Closing the player browser
also ends control. The separate SystemX server
keeps running until its console is stopped, retaining its model VRAM while idle.

```powershell
# From your SystemX checkout root:
Set-Location ./addons/minesweeper
.\play-minesweeper.bat --interface visual
.\play-minesweeper.bat --interface browser --difficulty intermediate --delay 0.5 --max-steps 500
.\play-minesweeper.bat --server http://127.0.0.1:8001
.\play-minesweeper.bat --policy logic
.\play-minesweeper.bat --url "https://your-host/apps/minesweeper/"
```

`systemone` is the default policy and currently uses StartLux. `logic` is an
explicit CPU-only comparison mode
requiring no model server. Difficulty options are `beginner`, `intermediate`,
and `expert`; only Beginner gameplay has been live-verified. There is no default
game URL. `--url` pre-fills the URL and selects the browser-label inspector;
click Connect before playing. `--difficulty` and `--delay` apply to browser play;
the default browser delay is 0.35 seconds. `play-desktop.bat` remains a shortcut
to the same inspector with screen pixels selected.

## Console compatibility

Pass `--console` to retain the original browser console controls: `r` runs,
`p` pauses, `s` steps, and `q` closes the player browser. The console prompts for
a URL if omitted. Its model-readiness wait reports progress and times out after
five minutes. `--autoplay` also selects this console path, starts immediately
after browser setup, and closes the browser when finished.

```powershell
.\play-minesweeper.bat --console --url "https://your-host/apps/minesweeper/"
.\play-minesweeper.bat --url "https://your-host/apps/minesweeper/" --difficulty beginner --autoplay
```

For console-only installations, the Playwright packages in `requirements.txt`
are sufficient in addition to root dependencies. For screen pixels only,
`setup-desktop.bat` installs the visual dependencies without Playwright.

## How it works

1. In browser-label mode, `observe.js` reads rendered accessibility labels,
   viewport-space AABBs, difficulty, and status indicators. `observation.py`
   validates the complete grid. In screen-pixel mode, `visual/recognition.py`
   classifies calibrated cells using taught appearance samples. Neither adapter
   reads concealed mines or internal game state.
2. Both adapters produce the same `board.py` representation of covered,
   flagged, numbered, empty, winning, and losing presentations. Unknown visual
   cells block play rather than becoming guessed observations.
3. `planner.py` uses adjacent-number and subset deductions, including the
   displayed total mine count. Existing flags are assumed correct. When no
   move is proved, candidates are explicitly uncertain guesses.
4. `decision.py` sends visible state and up to 12 candidates to `/v1/systemone`.
   System One chooses between multiple candidates. A single forced move skips the
   model call. Model failures never silently switch to a heuristic choice.
5. `session.py` re-observes before acting; changed state or geometry cancels the
   move. `browser_driver.py` hit-tests the cell center, checks cancellation
   immediately before input, and waits for the selected
   cell's expected result or a loss. Geometry movement alone is not success.
   `desktop_driver.py` provides the corresponding screen-pixel input checks.

System One ranks the candidate actions; exact Minesweeper deductions belong to this
addon. Guesses can lose. This does not calculate exact probabilities for every
ambiguous board. There is no continuous inference or extra vision model.

## Logs and verification

Inspector browser runs write `actions.jsonl` and `result.json` under
`.cache/minesweeper/browser/<timestamp>`. Screen-pixel runs use
`.cache/minesweeper/desktop/<timestamp>` and also save a final board crop.
Console runs retain `.cache/minesweeper/<timestamp>`, including `initial.png`
and `final.png` on normal completion. Action logs include bounding boxes, visible
before/after states, decisions, and their model/logic source.

```powershell
# Run from your SystemX checkout root.
& ./.venv/Scripts/python.exe -m pytest addons/minesweeper/tests -q
& ./.venv/Scripts/python.exe -m addons.minesweeper --help
& ./.venv/Scripts/python.exe -m addons.minesweeper --console --help
& ./.venv/Scripts/python.exe -m addons.minesweeper.recommend observation.json
```

Measured Minesweeper timings are recorded in the
[performance report](../../docs/model-performance.md#live-minesweeper-follow-up). Model-specific
measurements apply only to the configuration recorded in that report.
The unified inspector has automated controller and adapter checks; the redesigned
UI has not yet completed a fresh live visual verification.

## Other games

Use `addons/<game_name>` for future games and keep core inference unchanged.
Reuse bounding-box and observe/decide/validate/act concepts when useful; there is
no generic game framework to conform to.

An AABB identifies location, not object identity. Here accessible labels supply
identity. Warcraft would need different perception, tracked state over time,
action generation, and evaluation. StartLux currently accepts text/structured state,
not screenshots directly.
