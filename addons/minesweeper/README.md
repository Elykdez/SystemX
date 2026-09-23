# Minesweeper player

Terminology: [System 1 (System One)](../../docs/system-one-model.md) is the
decision-model role, currently StartLux through the HTTP API. This browser mode
reads visible DOM labels; the [desktop mode](DESKTOP.md) uses OpenCV as System 0.
Game logic constructs candidates and the controller executes and verifies choices.

There are two entry points: this page describes the existing web/Playwright
player; the [desktop player](DESKTOP.md) reads pixels and controls a selected
Windows game window. Each has its own optional setup batch file.

This addon controls a separately hosted Minesweeper game at a URL you enter.
It reads visible cell labels and
bounding boxes, asks the System One model for decisions through the current SystemX server, and sends normal
left/right mouse clicks through Python Playwright and installed Google Chrome.

This is a Minesweeper-specific addon. It does not alter the core server, model
loader, or HTTP API. Future games should have separate sibling addons with
their own logic, calling the same general inference service. This folder contains
the player, not the game's source or the model server.

## Separate components

| Component | Start | Stop |
| --- | --- | --- |
| Model server | `start-server.bat` in this folder (one int8 worker) | Ctrl+C in its console; this releases model VRAM. |
| Minesweeper player | `play-minesweeper.bat` in this folder | `q` + Enter or Ctrl+C in the player console. |
| Game webpage | Opened by the player in Chrome | Closing the player closes its browser; the remote game host is unaffected. |

The standalone player needs neither Codex nor the ChatGPT extension/plugin.
It shares the project's `.venv` for dependencies but talks to the model only
through HTTP. See the [server README](../../README.md) for server installation
and API details.

## Start

1. On a fresh installation, double-click the root `setup.bat`, then double-click
   `addons/minesweeper/setup-minesweeper.bat`. Google Chrome must already be
   installed. This working copy already has the dependencies.
2. Double-click `addons/minesweeper/start-server.bat` and wait for the SystemX server  to report ready.
3. Open `addons/minesweeper` and double-click `play-minesweeper.bat`.
4. Enter the full game URL, including `http://` or `https://`, when prompted.
5. Once the board opens, type `r` and press Enter to start.

The setup and player batch launchers locate the root environment themselves.
They work when double-clicked or launched from another working directory.
Setup keeps its console open so you can read the result. Run it once per
installation; it installs the addon dependencies without loading the model.

While waiting, the console reports whether the server is unreachable or the model
is loading, with elapsed seconds and Ctrl+C cancellation. It polls `/health`
every two seconds, reports progress about every six seconds, and times out after
five minutes. A loading model is expected; it no longer floods the server console
with `/ready` 503 responses. Initialization failures stop immediately.

The player opens installed Google Chrome with a separate profile under
`.cache/minesweeper/chrome-profile`, keeping everyday tabs separate. Only one
player may use this profile at a time. The standalone launcher does not need the
ChatGPT Chrome extension; that extension was used for development verification.

If the local HTTPS certificate is rejected, handle the warning yourself in that
Chrome window and press Enter in the player console. Certificate validation is
not disabled. Unknown page structure stops the player instead of guessing.

## Controls

Type a command in the player console followed by Enter:

| Command | Action |
| --- | --- |
| `r` | Run or resume. |
| `p` | Pause; discard a pending decision before clicking. |
| `s` | Perform one move, then pause. |
| `q` | Stop and close the player browser. |
| Ctrl+C | Interrupt and exit. |

Input stops on a win, loss, step limit, model error, contradictory observation,
obscured target, or missing click result. There is no automatic rematch. After
normal completion the browser stays open until `q`; `--autoplay` closes it when
finished. Closing the player browser also ends control. The separate SystemX server
keeps running until its console is stopped, retaining its model VRAM while idle.

```powershell
# From your SystemX checkout root:
Set-Location ./addons/minesweeper
.\play-minesweeper.bat --difficulty beginner --autoplay
.\play-minesweeper.bat --difficulty intermediate --delay 0.5 --max-steps 500
.\play-minesweeper.bat --server http://127.0.0.1:8001
.\play-minesweeper.bat --policy logic
.\play-minesweeper.bat --url "https://your-host/apps/minesweeper/"
```

`startlux` is the default policy. `logic` is an explicit CPU-only comparison mode
requiring no model server. Difficulty options are `beginner`, `intermediate`,
and `expert`; only Beginner gameplay has been live-verified. There is no default
game URL: the launcher prompts before connecting to the model server or opening
Chrome. Pass `--url` to skip the prompt, including for unattended launches.
The page must use the supported Minesweeper structure; arbitrary Minesweeper
websites may need a different observer. Default delay: 0.35 seconds.

## How it works

1. `observe.js` reads rendered accessibility labels, viewport-space AABBs,
   selected difficulty, and the face/status indicator. It does not inspect
   concealed mines, framework state, application scripts, or browser storage.
2. `observation.py` validates the complete grid and recognizes the site's
   covered, flagged, numbered, empty, winning, and losing presentations.
3. `planner.py` uses adjacent-number and subset deductions, including the
   displayed total mine count. Existing flags are assumed correct. When no
   move is proved, candidates are explicitly uncertain guesses.
4. `decision.py` sends visible state and up to 12 candidates to `/v1/systemone`.
   System One chooses between multiple candidates. A single forced move skips the
   model call. Model failures never silently switch to a heuristic choice.
5. `session.py` re-observes before acting; changed state or geometry cancels the
   move. `browser_driver.py` hit-tests the cell center before mouse input and
   waits for an observable result. Geometry movement alone is not success.

System One ranks the candidate actions; exact Minesweeper deductions belong to this
addon. Guesses can lose. This does not calculate exact probabilities for every
ambiguous board. There is no continuous inference or extra vision model.

## Logs and verification

Runs write `initial.png`, `final.png` on normal completion, and `actions.jsonl`
under the root `.cache/minesweeper/<timestamp>`. Logs include bounding boxes, visible
before/after states, decisions, and their model/logic source.

```powershell
# Run from your SystemX checkout root.
& ./.venv/Scripts/python.exe -m pytest addons/minesweeper/tests -q
& ./.venv/Scripts/python.exe -m addons.minesweeper --help
& ./.venv/Scripts/python.exe -m addons.minesweeper.recommend observation.json
```

Current desktop integration results are recorded in the
[validation report](../../docs/minesweeper-validation.md). Model-specific
measurements apply only to the configuration recorded in that report.

## Other games

Use `addons/<game_name>` for future games and keep core inference unchanged.
Reuse bounding-box and observe/decide/validate/act concepts when useful; there is
no generic game framework to conform to.

An AABB identifies location, not object identity. Here accessible labels supply
identity. Warcraft would need different perception, tracked state over time,
action generation, and evaluation. StartLux currently accepts text/structured state,
not screenshots directly.
