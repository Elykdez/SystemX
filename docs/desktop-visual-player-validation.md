# Desktop visual player validation

Date: 2026-09-23. Status: implemented, with partial live validation.
The supervised pixel/Von test won in 21 moves; a later user-operated desktop Von
run recorded a win in 40 moves. 146 automated tests passed. The latest observation
optimization needs a live retry, and the ten-game campaign remains outstanding.

## Implemented

- Shared Windows target identity, physical-pixel capture, coordinate conversion,
  foreground/occlusion checks, cancellable input, pointer parking, and global hotkeys.
- Minesweeper calibration UI, grid suggestion, explicit appearance teaching and
  profile files, unknown-cell rejection, stable observations, and click verification.
- Existing Minesweeper planner and Von HTTP client reused; DOM player retained.
- Desktop setup/play batch files, session logs, whole-GPU sampler, and an opt-in
  real-model test server that records CUDA allocator measurements.

Core `server.py`, `backend.py`, `model_download.py`, and `owner_watchdog.py` have
the same hashes as at the start of implementation. No WebSocket endpoint was added:
transport benchmarking remains deferred as specified by the design.

## Automated checks

The final suite passed 146 tests in 3.72 seconds, with one existing Starlette/AnyIO
deprecation warning. The desktop setup batch file completed successfully in the
existing environment, and the play batch file's `--help` entry point passed.

Tests cover stale model responses after pause/resume, geometry invalidation,
unknown/ambiguous visual input, profile validation, clipping, blocked input,
partial SendInput cleanup, selected-cell postconditions, and screenshot regressions.
Additional tests cover initial pointer parking, bounded recognition caching,
cache invalidation after teaching or threshold changes, two-frame stability,
and excluding tool overlays from the window picker. Four controller tests exercise
capture, drag calibration, recognition preview, teaching, profile/calibration
persistence, capture failure, and pause-generation invalidation with fake desktop
inputs and a withdrawn real Tk root.
The existing server, download, owner lifecycle, and DOM player tests also passed.

Coordinate calculations were tested with negative monitor origins and scale
factors 1, 1.25, 1.5, and 2. Actual input was exercised at 100% DPI on the current
monitor. Physical tests at other DPIs/multiple monitors remain outstanding.

## Live desktop results

Target: the user's locally hosted Minesweeper page, in installed Chrome, Beginner
9 x 9 with 10 mines. The player read desktop pixels only and used SendInput.
Browser tools were used separately to start test boards, collect labeled appearance
examples, and verify visible outcomes. They did not supply state to the running player.

| Run | Result | Interpretation |
| --- | --- | --- |
| logic-01 | Paused before input | Foreground guard rejected focus loss. |
| logic-02 | One click, then paused | Cursor highlight obscured the selected cell; no repeat was sent. Pointer parking was added. |
| logic-03 | Won, 19 verified actions, 33.094 seconds | Pixel-only gameplay succeeded using the explicit logic policy, resuming the previous board. The webpage independently displayed a win. This was not a complete Von-driven game. |
| von-01 | Seven verified actions, then paused; webpage showed loss | Real Von choices and mouse input worked. An additional exploded-mine appearance was unknown; it was added to the profile. Loss-screen handling was replay-tested afterward. |
| von-02 | Paused before input | Chrome showed another tab. All 81 expected cells were rejected as unknown. |
| von-03 | Paused before input | Foreground guard rejected focus loss. |

The tool used to coordinate the desktop test then stopped because it could not
determine the active browser URL confidently enough to enforce its policy.
No further desktop inputs were issued through that tool. Test processes were
stopped and the measurement server was shut down. The user selected the existing
web game for live testing, so no separate native Minesweeper was tested.

The recorded fixtures are small board crops. Some samples were used to construct
the profile; they are regression checks, not independent accuracy measurements.
The later winning board was not used to train the profile, but its detailed labels
came from the player log and its terminal result was checked against the webpage.
A larger independently labeled held-out corpus and unknown-theme rejection-rate
measurement remain necessary before claiming broad recognition reliability.

The bundled profile supports the tested appearance and observed digits 1-4.
Digits 5-8 and other themes need explicit sample teaching. This is a calibrated
desktop player, not a proven recognizer for every Minesweeper implementation.

## Supervised visual/Von verification

After the user authorized a retry, the page was verified in the user's Chrome.
Automatic approval review still rejected the autonomous model-driven desktop
launcher with only `blocked by policy`. That launch was not retried through an
alternate mechanism. The permitted alternative below tests separate components;
it is not a complete autonomous Windows-input test.

For each turn, a saved browser screenshot was read by the addon recognizer,
and the actual server chose from the legal candidate list. An approved browser
control executed that individual move. Visible DOM labels were collected
separately as an independent oracle and were not provided to the model.

- Won from a fresh Beginner board after 21 moves and 21 actual Von calls.
  The webpage independently displayed the win.
- Paused at move 15 on a previously unseen digit 4. After inspection and explicit
  teaching, the same board resumed and won. This is not an uninterrupted frozen-profile run.
- All 20 independently labeled saved boards matched: 1620 of 1620 cell observations.
  These are correlated frames from one calibrated theme, not a general accuracy estimate.
- The final board is committed as `supervised-win.png/json` with independently
  collected labels. It was not used for template teaching. Earlier fixtures retain
  the provenance limitations described above.
- Model-reported latency during this supervised game: 33.22-182.57 ms, median
  99.19 ms. Browser tool/human delays make the game clock unsuitable as a speed benchmark.
- The real controller window was launched and visually checked. Its callback
  workflows were tested automatically; a complete physical-mouse UI walkthrough
  remains outstanding. Tool overlay windows are now excluded from its picker.

The addon now parks the pointer before its first observation as well as after
each click, fixing the initial cursor-highlight obstruction. The agent-operated
autonomous desktop launch was blocked at this stage; a later user-operated run
completed as recorded below. The owned test server and controller were shut down.

## Speed optimization

The original loop spent about 101 ms recognizing each Beginner board, repeated
eight times per move, plus 550 ms in fixed waits. The addon now caches up to 512
classifications keyed by exact normalized pixels and thresholds. Teaching clears
the cache. No board state, chosen action, or input permission is cached.

The stable-frame gap is now 20 ms instead of 100 ms, and the extra 150 ms delay
after a verified move is removed. Every action retains all four observations,
foreground/occlusion checks, generation cancellation, and selected-cell verification.
Each action log now includes per-stage timings.

A separate benchmark replayed the 20 saved screenshots with the final profile
and sent 19 real requests through one persistent HTTP client. It did not send input.
The benchmark ran separately from pytest to reduce CPU contention:

| Stage | Median | Maximum |
| --- | --- | --- |
| Recognition with classification reuse disabled | 100.867 ms | 104.537 ms |
| Recognition with caching, sequential saved frames | 8.738 ms | 58.033 ms |
| Decision including planner and HTTP | 45.056 ms | 209.800 ms |
| Server-reported inference | 33.850 ms | 201.320 ms |
| Four stable observations plus decision, simulated capture | 146.706 ms | 309.823 ms |

Recognition improved by about 11.5x in this replay. Cold/new appearances still
need classification. Continuous requests and supervised requests have different
latency distributions; the measurements do not establish the cause of that difference.

The replay cycle excludes Windows screen capture, real hit tests/input, UI redraw,
and game rendering. It supports targeting 3-4 clicks per second, but does not
prove that live rate. The slower measured request alone took about 200 ms.
No WebSocket endpoint or model/backend modification was necessary for these gains.
Von's [upstream benchmark](https://github.com/wfzyx/von#empirical-benchmark)
reports approximately 18 ms GPU latency in its own workload; that is not a
guarantee for this hardware, prompt size, server path, or complete visual loop.

Reproduce with the local recorded screenshots and an already running test server:

```powershell
./.venv/Scripts/python.exe -m addons.minesweeper.tests.benchmark_visual --folder .cache/minesweeper/desktop-recheck/supervised --grid 156 162 324 324 --server http://127.0.0.1:8002 --output .cache/minesweeper/desktop-recheck/speed.json
```

The command only reads saved images and requests decisions. It never controls the mouse.
The raw screenshot campaign is an ignored local artifact, not part of a fresh checkout.

### Flag redraw regression

The user's subsequent desktop run (`20260923-163932-359961`) verified four moves,
then stopped while placing a flag. Its captured frame showed the selected cell's
empty background before the flag glyph appeared. The planner raised a constraint
error from that intermediate frame, bypassing post-click retry handling.

Constraint validation now happens after two matching frames and reports an
observation error that post-click verification can retry within its existing
three-second deadline. Pre-click contradictions still prevent input. No click
is repeated. Three regression tests use the saved frame to cover successful
redraw recovery, persistent contradiction timeout, and pre-click rejection.
The fixture's corrected flag state is an expected transition, not an additional
independently captured final frame. The subsequent user run below exercised the fix.

### User run and further observation optimization

The user's run `20260923-164643-683873` recorded `won` after 40 verified moves in
13.062 seconds including startup, about 3.06 moves/sec. Median action time was
316.17 ms; median decision time was 72.35 ms. The other stages accounted for about
241 ms per move on average. This is a desktop pixel/Windows-input run with Von,
with no agent browser tools issuing its actions. Its outcome comes from the
player's log and the user's report that it worked, rather than an independent
DOM oracle for this game.

The next optimization carries a recent matching observation into the next
stability check. Every check still captures and validates the window anew.
Matching frames must be at least 20 ms apart, and evidence older than 100 ms
cannot be reused. Changes require another matching frame; errors and clicks clear
the history. Tests cover recency, expiry, changed frames, errors, and post-click
invalidation, as well as the existing redraw and stale-action regressions.

A before/after replay of 19 playing screenshots used the user's already-running
GPU server on port 8000. No mouse input was sent, and that server was left running:

| Median measurement | Before | After |
| --- | --- | --- |
| Captures per replay cycle | 8 | 6 |
| Observation time, excluding decision | 100.45 ms | 76.18 ms |
| Decision including HTTP | 67.51 ms | 33.39 ms |
| Complete replay cycle | 168.20 ms | 109.67 ms |

The observation reduction was about 24%. Model timing also varied between runs,
so the entire cycle reduction cannot be attributed to this change. These replays
exclude real desktop capture, input, UI redraw, and game rendering. Post-click
stability history was explicitly cleared in the replay to match the real driver.
The user's next controller launch is needed to measure the actual speed increase.
Core model/server files and request payloads were not changed, and this optimization
does not allocate additional GPU memory.

Local benchmark records: `.cache/minesweeper/desktop-recheck/`
`speed-before-rolling-von.json` and `speed-after-rolling-von.json`.

### Region capture and change-driven processing

The shared capture utility now accepts a client-coordinate region and retains
the original window geometry plus the pixel origin. Calibration still captures
the complete window. Gameplay grabs only the board at full resolution, and the
recognizer, controller, teaching tools, final images, and input coordinates account
for that origin. For the earlier 635 x 921 client window, a 324 x 324 board has
82.1% fewer captured pixels. The benchmark screenshot below was 635 x 744, with
77.8% fewer pixels after cropping. Neither ratio is a measured speedup.

Each fresh capture still verifies geometry and target ownership at all cell
centers. The driver compares every board pixel against a retained copy; only
changed pixels or recognition thresholds trigger recognition and a preview update.
Profile revision changes still require recalibration. Constraint validation,
stable-frame checks, cancellation, and click postconditions remain active.

Measured on saved pixels and a real withdrawn Tk canvas:

| Component | Before | After | Samples |
| --- | --- | --- | --- |
| Unchanged-board CPU read path, median | 1.717 ms | 0.075 ms | 100 paired |
| Preview drawing, median | 11.152 ms | 5.032 ms | 20 paired |
| Preview notifications for 103 identical reads | 103 | 1 | 103 reads |

These compare CPU image copies/recognition and Tk drawing, excluding the Windows
capture API, OS hit tests, input, and game rendering. They are not live click-rate
measurements. The 20 independently labeled saved boards still matched all 1620
cell observations after running through region capture offsets and the new cache.
Additional tests cover negative desktop coordinates, capture bounds, preserved
cell/input coordinates, cache invalidation, continued occlusion checks, and
calibration/teaching on a cropped preview. Raw component timings are in
`.cache/minesweeper/desktop-recheck/engineering.json`.

The real-model replay attempt found the user's server stopped (connection refused).
No replacement model process was started for this CPU-only optimization. No new
GPU allocation or live-game performance measurement is claimed.

### HTTP versus WebSocket

The existing HTTPX client already reuses its TCP connection; see the
[HTTPX client documentation](https://www.python-httpx.org/advanced/clients/).
In the user's 39 model-backed moves, the median difference between the complete
decision stage and server-reported computation was 2.409 ms (maximum 3.744 ms).
That includes planning, serialization, HTTP, and scheduling, not just transport.
This leaves little measured headroom for a protocol change in the current local
setup. No WebSocket implementation or direct protocol comparison was performed.
Evidence: `.cache/minesweeper/desktop-recheck/transport-analysis.json`.

Further downsampling was considered but not enabled. The current CPU matcher
already normalizes individual cells to 32 x 32, and tiny clue glyphs can become
ambiguous at lower resolution. Region selection removes irrelevant pixels without
reducing clue quality. A future RTS detector could separately evaluate a coarse
overview and high-resolution object regions against its own recognition data.

## VRAM measurements

Hardware: NVIDIA GeForce RTX 3070, 8192 MiB. Other desktop applications, including
Unity, were running. MiB below uses 1024 x 1024 bytes.

| Measurement | Value |
| --- | --- |
| Model idle, Torch allocated | 763.1 MiB |
| Model idle, Torch reserved | 852.0 MiB |
| Peak Torch allocation through warmup and seven gameplay inferences | 783.9 MiB |
| Peak Torch reservation | 878.0 MiB |
| Whole GPU before model load, 3 samples | 3152-3163 MiB; median 3162 MiB |
| Whole GPU with model idle, 3 samples | 4245-4261 MiB; median 4260 MiB |
| Whole GPU during Von gameplay, 36 samples | 4186-4252 MiB |
| Whole GPU after measurement server shutdown, 3 samples | 3134-3136 MiB |

The median before/idle difference was about **1098 MiB (1.07 GiB)**. This is an
approximate whole-device increase, not a precise process attribution: unrelated
applications changed their allocations. Allocator measurements are scoped to the
test model process; whole-device numbers include CUDA context overhead and all apps.

Whole-GPU sampling was once per second, so it can miss short peaks. The allocator
peak includes transient Torch tensor allocations but does not include every driver
allocation. These are short Beginner-game requests, not worst-case model inputs.

Vision runs with CPU OpenCV/NumPy and does not load another GPU model. The earlier
logic-only run recorded 3187-3204 MiB whole-device usage while other apps were also
running; this is not evidence of a specific GPU allocation by vision.

Cached model load/warmup took 14.57 seconds in this run. The loading sampler ended
after ten seconds, before CUDA loading completed; the idle samples and allocator
records provide the model-resident measurements. Do not read the early loading
samples as its full startup peak.

Raw local evidence is under `.cache/minesweeper/desktop-validation/`:
`cuda-allocator.jsonl`, `vram-phases.jsonl`, per-run `vram.jsonl`, action logs,
cropped final boards, and server logs. These are ignored local artifacts.

The authorized retry and two speed replays used a separate instrumented server.
Across warmup and 59 inference records, its Torch peak was **783.4 MiB allocated,
878 MiB reserved**. Whole-GPU samples at the end were 3108-3110 MiB before shutdown
and 2029-2034 MiB afterward, an approximate 1076 MiB median decrease. Both the
measurement server and test controller were closed; unrelated app use can also
affect that difference. No whole-GPU sampler ran throughout the supervised game,
so these end samples are not a new gameplay peak measurement.

Retry evidence is under `.cache/minesweeper/desktop-recheck/`:
`cuda-authorized.jsonl`, `vram-authorized.jsonl`, `speed.json`, `controller.png`,
and `supervised/` screenshots, decisions, visible-label oracles, and `oracle-checks.json`.

## Remaining validation

- Repeat the successful autonomous Windows-input Von run with the latest
  observation optimization, then the ten-attempt campaign on the frozen final
  profile. Measure actual clicks per second with the updated loop.
- Walk through all controller interactions using the actual physical mouse and
  hotkeys; the current coverage is visual inspection plus automated callbacks.
- Test real monitor/DPI changes, overlays, hotkey conflicts, and stop during a
  pending real-model request. Current coverage includes simulated versions.
- Test fresh-environment installation and a broader independently labeled visual
  corpus, including unknown themes. Existing-environment dependency installation
  and launcher checks are not a clean install.
