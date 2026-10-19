# Chrono Divide desktop player

Terminology: **System 0** is visual perception, implemented mainly with OpenCV
and NumPy. **System 1 (System One)** is bounded decision-making, currently supplied
by StartLux through the HTTP API. Model identity and revision live in
`model_config.py`; a new provider requires qualification. See the
[shared role definitions](../../docs/system-one-model.md#system-0-and-system-1-in-the-current-player).

A local agent for Red Alert 2, played in the Chrono Divide browser client.

Allied Prism formations stage outside the enemy base instead of ordering the entire mixed
army onto a building. Before the first wave leaves home, native hover names verify separate
visible Prism, Mirage, IFV and Grizzly groups. Every member is rechecked before saving its
role, so a click on a moving vehicle's former position cannot silently save another type.
System One chooses hold, bombard, reposition or a short advance from range, health,
nearby threats and escort evidence. Only Prisms receive direct building attacks; escorts
screen them and attack-move only toward close threats. An advance requires repeated clear
views. Missing type identity or escort cover prevents a siege advance. These pixel distances
are conservative estimates at the calibrated canvas size, not measurements of weapon range.
Native unit-type groups and live Soviet and Allied wins against Easy Enemy
were verified during development, before the inspector redesign.

If cash and production stall for 90 seconds with a fresh large army at home, System One can
consider that force instead of waiting indefinitely for a still larger first wave.
This does not relax the minimum after failed attacks or when the census is stale.

It reads only the visible game window, keeps a compact description of the match, asks the System One
model for bounded decisions through the local SystemX server, and acts with mouse and keyboard input.
Like the Minesweeper desktop player, it finds the game through a chosen window and taught
recognition profiles rather than one client's internals (see "Other RA2 clients").
Design: [`docs/chronodivide-agent-design.md`](../../docs/chronodivide-agent-design.md).

## Run it

1. Start the model server: double-click `start.bat` in the SystemX folder (the
   launcher below also starts it if nothing answers on port 8000). The server now
   uses one worker on 8 GB GPUs and up to three on 24 GB GPUs.
   Override the hardware default with `-Workers N`.
2. Open Chrono Divide in the browser at 100% zoom, with a window larger than the
   1024x768 game canvas. Use Demo Mode with Iraq as the Soviet default versus an
   Allied opponent on Isle of War. For Allied play, use the Allied setup below.
3. Double-click `addons/chronodivide/play-chronodivide.bat`. The controller window
   opens at the right edge of the screen, beside the game.
4. Choose **Execute** and **System One** (or **Rules only**), press **Start**. The first time, or for
   another client or window, press **Setup...** first (below).
5. Click into the game so it captures the pointer, then press **F8**. Keep your hands
   off the mouse and keyboard while it plays.

- **F10** pauses only the agent; the match continues. Open the game's menu with Esc to
  stop play where the client supports it. Any physical mouse or keyboard use also pauses
  the agent at once; F8 resumes it.
- The agent acts only while the match is actually being played. The game menu (Esc),
  a released pointer, another window in front, loading and score screens all make
  it wait without sending anything.
- **Watch** and **Recommend** modes never send input: Watch shows perception only,
  Recommend also shows what the agent would do and what System One chose.

Hands-free: double-click `addons/chronodivide/play-unattended.bat` instead. It starts in
Execute + System One, arms itself for the match, brings the game window to the front, and
resumes by itself a few seconds after you stop touching the mouse or keyboard. You
only start the skirmish in the browser.
The inspector process keeps playing independently of the chat. With `--unattended --front`,
it also restores a minimized game or brings it back after the quiet period. F10 / Pause agent
holds it stopped until F8 / Act. Capture failures invalidate the previous playing state;
input resumes only after a fresh valid capture.

Headless read-only checks: `python -m addons.chronodivide watch --seconds 30`.

### Live inspector

The controller shares its dark layout with the Minesweeper inspector: an annotated
game preview, a prominent status banner, and a scrollable state panel. Resize it
for a side-by-side preview and inspector; narrower windows stack the panels.
The preview fits the available space without changing game calibration or input
coordinates.

The state panel reports observed resources, production lanes, tracked objects,
army activity, persistent intent, System One information, and recent events.
Capture failures or unrecognized HUD anchors appear as explicit warnings. The
game window does not need to be maximized: the complete calibrated game view
must be visible and unobscured. Keep the inspector beside the game while playing.

**Start**, **Act**, **Pause agent**, the Watch/Recommend/Execute modes, and F8/F10
retain their existing behavior. Pausing the agent does not pause the match.

### Setup: window, view, and what is recognized

**Setup...** opens the calibration window (the Minesweeper desktop player's technique). It sends
no input to the game; it only captures its window.

1. Choose the client profile and the game's window (windows whose title names the game come
   first; File Explorer folders never count). Start a match, then **Capture**.
2. The game view is found from the sidebar's HUD anchors (**Find game**); if they are not found,
   drag around the game view in the preview.
3. The preview shows what is read: cameos (red when unknown), credits, power, play or menu, and
   the scene's units and buildings.
4. Teach what is wrong: click a cameo and choose its item (in its normal, available look), enter
   the credits shown, **Teach HUD look**, or pick our and the enemy's house colors by clicking a
   unit of that side. **Save profile** writes a new profile (bundled ones are never changed).
5. **Use this setup**: the next Start plays that window with that profile (saved in
   `.cache/chronodivide/setup.json`).

## What it does now

- Calibrates the game's pointer-locked cursor from its own sprite (two move sizes,
  delivered in short steps so browser edge clipping cannot look like acceleration;
  stops with advice if in-game mouse acceleration makes motion non-linear).
- Deploys the MCV at once (keys N then D only while no construction yard exists, since
  "MCV Repacks" would pack a yard up again; clicks it if the keys fail).
- Follows the profile's Soviet build order (reactor, refinery, barracks, war factory,
  two Rhinos before the second refinery and four before radar), builds a reactor first when
  power is low, and places buildings where the footprint preview is all green.
  A visible but unaffordable building keeps its place in the order; only unavailable
  items use the configured wait timeout.
- Produces sensible units only (the profile's numbers): Rhino Tanks first, War Miners up to
  two per refinery, the first Flak Track eligible after one Rhino before expansion, then roughly one
  per four tanks, Apocalypse Tanks and available Desolators only with a funded, observed advantage, two early
  Conscripts, and one basic Sentry Gun while reserving mobile-army funds.
- Reads the radar once a Radar Tower has power: map outline, camera box, our base, enemy
  dots, shroud.
- Reconciles lost last production buildings from repeated matches to recorded disabled
  sidebar buttons, clearing stale lanes and their reservations. Unknown button appearances
  cannot prove a loss; partial losses of multiple refineries or reactors remain untracked.
  Allied aircraft keep the vehicle button enabled after a factory loss: repeated complete
  aircraft-only sidebars clear the ground factory count and its queued orders so it can be
  rebuilt. The working aircraft lane is preserved. Allied radar accepts the recorded white
  and native pale-blue outlines for offscreen warnings and camera checks.
- Army: guards the base; defense sessions use P twice to recall owned military across
  the map, recheck the threatened view, and attack-move toward freshly observed raiders;
  attacks in waves (camera to the target by a radar click, the army selected, Ctrl+Shift+click
  attack-moves it, H returns the camera), with reinforcement waves and battle sessions (focus
  fire, sieges) until the enemy base falls. Targets: seen enemies, then the likely enemy start
  (our base mirrored through the map center), then shrouded areas.
- Scouting: two Attack Dogs come first and are saved as team 2 (Ctrl+2, while other infantry
  waits a moment so P selects the dogs alone); System One's "explore" walks them around the base to
  lift the shroud, and "scout" sends them to the likely enemy start once the radar is up.
  Camera looks at the enemy base (how strong it is) and at our fight.
- The inspector's scene (perception/scene.py), on its own thread, a few scans a second
  (30-50 ms each; the 20 Hz loop only hands frames over): every selected unit by its health bar
  (the bar's white frame gives its size class whatever the damage: 18 px infantry, 36 px
  vehicles; the pips its health; a cargo strip under the sprite marks a transport), named from
  the types we built ("?" when several fit); Soviet construction yards and Tesla reactors
  by recorded sprite appearance (`perception/buildings.py`), with house color used only
  to identify ownership; and, once trained, the unit detector's finds. Other color patches
  remain unconfirmed and never count as units or buildings. The bundled building templates
  cover native-scale, intact appearances; other buildings, damage, occlusion, and scaled
  clients can be missed. Objects are tracked: when the camera pans, the view's shift is measured (phase
  correlation) and every box moves with it, a jump drops them; a unit named while selected keeps
  its name after the selection ends while its color is still seen there; boxes not seen lately
  fade, then go. The inspector uses single-pixel outlines and compact labels on dark backgrounds.
  Unconfirmed color patches get muted outlines without labels; the panel lists their counts
  separately. A question mark marks a tentative name, a percentage is health, and seconds
  show how long an object has gone unseen. The census counts the army at home by health
  bars, and battle sessions aim from where our units stand.
- Unit detector (perception/detector.py): a small network (about 0.5 M parameters) that finds
  infantry and vehicles by shape, selected or not; the house color inside a box says whose unit
  it is. It is kept in the SystemX server's memory when the server runs (the player copies it
  there once; 4.7 ms per view on an RTX 3070 with about 40 MB of GPU memory at its peak), else it
  runs in the player on the CPU (about 70 ms). Train it with `tools/detector.py` from recorded
  games: frames are used as the game drew them; unselected units are labeled from census pairs
  (the view just before P selects the army at home, labeled by the health bars just after), which
  every census in execute mode records under `.cache/chronodivide/detector/census/`. No model is
  shipped yet: it needs those pairs from live games.
- Asks System One every 15 s, as separate small requests: what to spend on, what the army does,
  how many tanks it needs at home before it attacks, and what to find out next; plus
  production picks when several fit. Rule defaults apply until an answer arrives or when
  it is late, so the player never waits for System One.

## Perception, decisions and execution

- **System 0** reads visible HUD, sidebar, radar, units, buildings and health.
  The readers in `perception/` preserve unknowns; `world.py` and `intel.py`
  retain observations with their age and distinguish memories from fresh sightings.
- **System 1** receives structured facts and feasible choices from the controller.
  StartLux chooses workflow focus, spending, force size, reconnaissance, production
  and bounded tactical actions using the faction profile's criteria.
- **The controller** supplies candidates, deadlines and rule defaults, keeps the
  chosen attack/defend/build focus between incidents, and schedules skills.
  Skills recheck visible targets, send guarded input and verify later frames.
  Routine or forced actions can run without another model request.

For example, OpenCV detects a visible threat and selected unit health; it does not
decide a Prism attack. StartLux can choose bombardment from the offered options,
and the skill then rechecks the target and controls only the verified Prism group.
The next visual observation establishes whether that action had the expected effect.
System 2 is not required for this loop and remains a proposal.

## System One decision profile (current StartLux backend)

Faction strategy settings and question criteria live in
`profiles/startlux-ra2-soviet.json` and `profiles/startlux-ra2-allied.json`, read by
`playbook.py`. Python measures the game, constructs feasible choices and executes
skills; StartLux selects among the choices. The profiles supply:

- `questions`: every question put to System One, with its instructions, its options in words,
  the facts its state holds, and `rules`, its default answer (the first rule whose option
  is offered and whose fact classes hold). Attack size options carry their number of tanks.
- `facts`: the words of every fact System One reads, by class (for example `money` is
  `pile_up` from 5000 credits: "credits pile up unused"). Option words must use the same
  words as the facts. Changed wording needs new model and gameplay checks.
- `playbook`: the build order and its gates, unit counts and ratios, army strength,
  attack waves, battle sessions, defense and scouting numbers, each with the live-game
  evidence behind it.
- `startlux`: how often strategy is asked, deadlines, and the answer cache.

Edit the JSON and restart the controller. After changing any wording, check StartLux's answers
on set situations (needs the server): `python -m addons.chronodivide.tools.decisions`
(`--show` prints states and answers, `--profile` checks another file).
The Soviet profile's `decision_checks` entry points to `decision-checks-soviet.json`,
resolved relative to the profile file. That JSON contains the scenario defaults, visual
fixtures, cases and expected answers. Per-case `structures` and `units` override individual
default counts; `null` removes an entry. These are evaluation fixtures; live policy remains
in the current System One profile's `playbook`, `questions` and `facts` sections.

Gameplay tuning belongs in that JSON, not Python constants. Current policy entries include:

| Policy | JSON location under `playbook` | Shipped Soviet behavior |
| --- | --- | --- |
| Harvesters and expansion | `units.miners`, `units.second_refinery`, `buildings.order`, `buildings.tech_reserve` | Two miners per refinery, capped at six; no extra expansion miner. A second miner may follow two Rhinos and first anti-air support during a quiet interval. Two tanks precede refinery expansion. Building funding reserves one tank's cost while the refinery or radar is pending, including during raids. |
| Infantry, armor and support | `units.scouts`, `units.conscripts`, `units.troopers`, `units.anti_air` | Two dogs, two early Conscripts, first Flak Track eligible after one Rhino and one Flak Track per four tanks thereafter; after a failed attack, fund four available V3 launchers, then grow the cumulative quota by one per four Rhinos to permit replacements. Gates and cash requirements live beside the counts. |
| Basic anti-air infantry | `units.anti_air_infantry` | After the dog quota and scout-grouping hold, offer available Flak Troopers before filling Conscripts once two tanks are built. Target one per tank, capped at twelve trained troopers, with at least 600 spendable credits before buying each 300-credit trooper. Sidebar availability supplies the radar prerequisite; optional high-tech readiness is not required. Counts are cumulative production, not surviving infantry. |
| Static defense | `units.defenses` | One proactive Sentry Gun after two tanks, at most one gun. Placement direction before radar is configured by `front_without_radar`. |
| Production batching | `units.queue_two` | `count` is two for configured lanes with surplus credits; `default_count` is one. The section name is retained for compatibility. |
| Optional high tech | `high_tech`, `buildings.order[].high_tech`, `buildings.tech_up` | Battle Lab expansion and advanced units need at least 5000 spendable credits, two refineries, four estimated miners, 60 quiet seconds, a census of at least 12 combat units including eight healthy vehicles, and a visible enemy base with weak or medium defenses. Both observations must be within 45 seconds and enemy-view shroud at most 40%. Missing or stale evidence keeps production on the basic force. Radar remains basic reconnaissance. |
| Army census | `army.census` | Count every 20 seconds, accept counts for 45 seconds, require a count within 10 seconds before starting an attack. |
| Attack decisions | `attack`, plus `questions.attack_size` outside `playbook` | Requirements are 4 / 5 / 6 / 8 / 10 tank equivalents; the initial unknown-defense choice requires six. `early_min_above_small` allows an existing early-game answer only above the small-force size by the configured margin (one tank); late game, failed attacks and strong defenses still use the feasible rule minimum. The six-vehicle clean trial dispatched a wave, reached the enemy base and lost it; this is not a proven winning composition. |
| Defense observation | `defense.watch_every`, `defense.watch_rounds` | Inspect an approach every eight seconds; issue up to two defense orders on confirmed contact. Before radar, inspect a fresh attack or building-loss incident with Space (recent EVA events). Fresh pixels determine whether to fight; the shortcut itself does not prove an enemy is there. |
| Emergency War Miner | `defense.emergency_miners` | In the zero-starting-unit opening, group the first refinery's sole vehicle as team 4 before the factory exists and verify reselection. Against recognized ground infantry, use it only with at most two combat vehicles at 50% health or better and the miner at least 60% healthy. Support lasts 12 seconds, checks every three seconds, retreats below 45%, and has a 25-second cooldown. Require visible gold ore before committing and re-detect it for the return order. Missing ore stops the miner and preserves a return retry. Gems and unknown vehicle types are not identified by this reader; logs distinguish a harvest order from observed harvesting. |
| Resource survey and relocation | `resources`, plus `questions.resources` and `facts.resources` | Count visible home ore every five seconds across a 900-pixel reach. Fewer than twelve chips for 25 seconds establishes scarcity. With cash below 2500, a saved miner and 25 quiet seconds, System One can survey configured nearby radar offsets that are at least 60% explored and clear of observed enemies, or choose a fresh gem field. The movement/harvest task reselects one healthy saved vehicle and confirms fresh crystals before clicking. Field observations expire after 90 seconds; input and a later cash rise remain separate from verified delivery. Autonomous live income recovery is still unverified. |
| Existing-force patrol | `patrol` | Three vehicles and one infantry, four vehicles reserved at home, 2000 credits, 30 quiet seconds, at least 75% health. Prefer a transport if one is visually indicated. Check every eight seconds and refresh after 40; retreat below 50% health. |
| Patrol scouting | `patrol.scout`, `patrol.route` | Require 4000 credits, two refineries, 60 quiet seconds, 85% health and a fresh squad observation. Route offsets, allowed distances and enemy-base avoidance are configurable. |
| Dog exploration | `intel.explore` | Configured route and return offset; check every four seconds. Deadline is route-node count times `leg_seconds` times `deadline.leg_multiplier`, bounded by `min_seconds` and `max_seconds` (60 to 180 seconds). Return needs fresh visual evidence and `return_min_health`. |

Times are real seconds unless a key explicitly says milliseconds; health is a fraction from
zero to one. Patrol route offsets, `front_reach` and `retreat_reach` are tactical pixels;
patrol scout offsets/distances and `threat_radius_extra` are radar pixels. `requires` lists
the buildings that must still exist before forming a patrol. Formation and replacement
never purchase units. UI settling waits, frame-evidence thresholds, coordinate clamps and
selection verification remain implementation details in Python.

### Allied Faction

Use `--strategy addons/chronodivide/profiles/startlux-ra2-allied.json` when launching the
controller, and choose the `chronodivide-allied` client profile in Setup. Its recorded
assets target a 1024x768 canvas with France blue and the opponent red. Select **Demo
Mode** and set Iraq as the opponent for the France trial.
Use the classic left mouse attack/move setting and turn Target Lines off; red order
lines can resemble enemy house color in the visual threat reader.

The Allied opening builds four Grizzlies and IFV support, then powers Airforce Command
before the second refinery. Expansion waits for six Grizzlies. Early radar enables
approach monitoring and a forward gathering point. Battle Lab follows the economy checks.
The first mixed force builds two Mirages to screen four Prism tanks, then completes
six Mirages. Optional Spy Satellite funding waits until the first two Mirages
have been queued so reconnaissance cannot block that initial screen; empty IFVs provide
anti-air. Two dogs scout and two additional dogs guard the base in both faction profiles.
The Allied first siege normally gathers twelve combat vehicles even against a dense
static base; its Prism fire works at range. A failed attack raises the threshold to
twenty, and repeated failures to twenty-four. Weak observed defenses can still allow
a smaller force. Soviet attack-size rules retain their own thresholds.
Home infantry is deselected before an assault team is saved. A funded Grand Cannon
can cover the base after two Mirages have been produced, when its icon is available.
Both StartLux profiles describe their
faction's shared strategy rather than requiring a country. Desolators and Grand
Cannons are optional country units: absent icons are skipped without blocking the
shared production plan. Chrono Miners are
unarmed and are excluded from emergency combat support. The Soviet profile retains
its dog scouts and armed War Miner support. Prism production continues at one per
two Mirages after the initial quota, so cumulative production does not stop all
siege replacements. Chrono Miners can use visible ore when no safe gems are found.

Allied Battle Lab placement prefers the rear of the base when radar establishes
the approach direction. Blocked or shrouded sites still require another legal
footprint. Detours around blocked footprints retain that rear priority rather
than stepping toward the front ahead of untried rear sites. Without radar, the
existing candidate order remains in use.

Allied perception includes blue credits and queue text, the white radar camera frame,
disabled production tabs, thirty cameos, and construction-yard/power-plant templates.
Strategy selection also switches unit costs and prerequisites; it is separate from
client layout and color calibration in `profiles/clients/*.json`. Production counts
remain cumulative, so attacks require a fresh visible army census. Repeated loss
of radar and its exclusive sidebar options clears a destroyed Airforce Command
from the ledger and reserves funds to rebuild it.

The shared intent stores StartLux's workflow focus (`attack`, `defend`, or `build`).
Every four seconds, visual battlefield facts are checked for a changed incident:
force readiness, assault start/end, a serious base raid, an expired defense option,
or a lost assault. Own red placement previews do not count as tactical enemies;
radar contacts still trigger defense while a building is being placed.
Only a changed incident asks StartLux for another focus decision when multiple actions
are feasible; a single feasible action uses its safety rule. The current focus
persists between checks, and late answers from older incidents are ignored.
Attack focus gives combat priority over optional scouting and resource surveys
and keeps the camera with the field army between battle sessions.
When field dots are missing, inspect the attack target before falling back to
the team's center. Loss verification inspects the battle location or known
enemy base: a team split between home and the front can center on empty ground.

Live StartLux wins for both factions predate the inspector redesign; they do not
constitute a fresh live verification of the redesigned UI.

### Multiple Opponents And Allies

In Setup, teach every hostile house color with **Add enemy color** and every
teammate color with **Add ally color**. These lists persist in the client profile;
adding one opponent does not replace the others. Friendly contacts are excluded
from threat detection, assault targets and own army strength. Multiple opponents
share the hostile target pool; an assault finishes its current visible enemy base
before choosing the next one. Scouting avoids known teammate bases.

Enable **Build off Ally** in the actual match and check **Build off Ally enabled
in match** in Setup to allow teammate-side construction. Once our economy and
base cover are established, System One can consider a finished utility or defense
building near a visible, quiet ally base while preserving two tanks' funds.
Attack focus, a home raid, low power or a nearby hostile contact keep placement
at home. Remote placement still requires a visible green footprint, tries at most
eight locations and falls back to home if no legal site is found. It does not
queue extra structures solely to support an ally or command friendly units.

Regression checks cover one player against three hostile bases, separate hostile
and friendly colors, stale placement decisions and verified remote placement.
The live team smoke test uses Britain blue and Germany green against Iraq red
and Russia purple in Demo Mode on Country Swing (2-4). Britain built shared
Allied infrastructure, Grizzlies and dogs, but lost before radar was established.
The trial exposed a too-bright lobby-swatch calibration for purple uniforms;
teach colors from actual battlefield units or buildings. Live teammate-side
construction and a complete team match remain unverified.

## Other RA2 clients

Every RA2 client draws the same game art (Chrono Divide uses the original game's files), so what
differs between clients is kept in a client profile (`profiles/clients/*.json`, `client.py`): the
window's title and process, whether the game is drawn in a page or fills its window, the
sidebar's geometry, the taught templates (cameos, credits digits, HUD anchors, cursor), the house
colors, and the input: `locked` (browser pointer lock: relative motion, as in Chrono Divide) or
`absolute` (the ordinary cursor, as in the original game in a window). The layout of any view size
follows from the sidebar's geometry (`Layout.derive`; at 1024x768 it equals the measured layout).

- `chronodivide`: verified in live games on a 1024x768 view.
- `ra2-original`: a starting point for Westwood's Red Alert 2 or Yuri's Revenge in a window (for
  example through CnCNet). Its sidebar geometry and templates are copied from Chrono Divide and
  nothing in it was checked on the original game: capture a match in Setup, press Find game,
  teach what shows as unknown, and save it as your own profile.

## How it is built

| Module | Role |
| --- | --- |
| `client.py`, `calibration.py`, `setup_ui.py` | Client profiles; ranking windows and finding the game view by the HUD anchors; the setup window (capture, find or mark, check, teach). |
| `window.py`, `recorder.py` | Find the game's window and, in a browser, the canvas; read-only recording with pointer-lock evidence. |
| `perception/` | Sidebar cameos (item + available/disabled/building %/ready), credits digits, power bar, play/menu mode, game cursor sprite, house colors (`colors.py`), unit boxes (`unitboxes.py`), the inspector's tracked scene (`scene.py`), the unit detector (`detector.py`, network in `detector_net.py`). |
| `world.py` | Facts with ages, production lanes, ledger with credit reservations, events. |
| `broker.py` | Multi-route System One client: priorities, replaceable per-key slots, deadlines, defaults, memoization, provenance. |
| `playbook.py` | Loads and checks the current System One profile: questions, fact words, rule defaults, playbook numbers. |
| `intent.py`, `reflex.py` | Standing intent written by System One; the fast rule policy that words the facts, asks System One, and executes the answers. |
| `army.py`, `intel.py` | Army postures, attack waves, scouting and camera looks; the battlefield picture. |
| `skills.py`, `tasks.py` | Skills as coroutines (act, wait, look, expect); one task per lane; earned trust. |
| `executor.py` | Guarded, paced input: safe zones, blocked keys (browser shortcuts, Esc), IME check, playing gate, pointer-lock motion. |
| `player.py`, `controller.py` | The 20 Hz loop and the debug window (the inspector: the annotated view with unit boxes and names, and the state panels). |
| `frame_player.py` | Screenshot-only player for external input transports; accepts the existing System One broker or uses rule defaults. Emits one canvas-coordinate transaction and waits for its acknowledgement plus a fresh frame. Does not capture or control the desktop. |
| `profiles/` | The current System One decision profile; calibrated layout, cameo/digit/HUD templates, cursor sprite, RA2 Soviet knowledge, key map. |

Calibration tools: `tools/sheet.py` (contact sheets), `tools/cameos.py` (cluster cameo
appearances for labeling), `tools/replay.py` (run perception and policy over a
recording), `tools/systemone_capacity.py` (StartLux latency and worker scaling), `tools/decisions.py`
(StartLux's answers and the profile's rule defaults on set situations), `tools/units.py` (unit
boxes and names drawn on a run's recorded frames), `tools/detector.py` (the unit detector's
dataset from recorded games and census pairs, training, held-out evaluation, export).

Runs are logged under `.cache/chronodivide/runs/<time>/`: `events.jsonl`,
`decisions.jsonl`, `actions.jsonl`, `summary.json`, frames at 1 fps, failure frames, and
`units/` (JPEG views with the recognized units drawn, at most every 5 s while units are
selected).

Historical debug recordings may retain only their first, last and referenced frames
after cleanup. Named evidence, event logs, calibration and tracked fixtures are kept;
a complete frame-by-frame replay requires an unpruned recording.

## Limits

Defense selects owned military across the map with P twice, even when no defenders
are visible in the threat view. Camera sweeps and EVA incident checks can form a
remote defense selection after removing visible home infantry; they no longer
require a vehicle to be visible at home. Attack formation still requires visible
verification. Defense logs distinguish map selection from the local visible count,
which cannot establish whether distant units survived.
Each defense order reselects the map army (or the prepared field team) in the same
input transaction as its attack-move. Sidebar work between observations cannot
clear the selection and silently discard a remote recall.
When a defense that issued orders ends and the AI returns to gathering, it sends
one map-wide regroup order toward home. A fresh threat or active assault prevents
that regroup. The next home census verifies who returned; sending the order does
not establish arrival or survival. Outside an assault, that census briefly centers
on the saved regrouped army so formations near the screen edge are visible, then
restores the home camera. During an assault it still counts only reinforcements
at home.
After selection, defense rechecks enemies around the threatened camera view, not
around the selected army. Otherwise distant defenders could put a visible raid
outside the target search radius and cancel the recall. In Allied native trial 16,
the army returned to the base and cleared an infantry wave after this correction.
Fresh visible home raids preempt attack preparation at an input-batch boundary.
This also covers lengthy native unit-role verification before the first Allied wave.
Bounded role probes continue beyond the previous batch instead of repeatedly missing
new tanks behind the same first sixteen vehicles. They first center on the saved
formation so escorts outside the home viewport can be inspected. Verified groups survive retries.
Saving a role rebuilds its verified local selection and assigns the control group
in one transaction. The saved group is then reselected and its count and native
type labels are checked before the role is accepted.
Allied siege sessions center on the verified Prism control group during travel.
An empty Prism group triggers a fresh whole-army view and role recheck before
counting a missing force; losing the siege subgroup alone must not stall survivors.
The saved Prism selection is rechecked first so an interrupted selection does not
discard a valid role. New sidebar observations and queues wait while an army skill
inspects its selection; existing production continues in the game. Army work waits
for an existing queue, repair or placement operation to finish.
If radar is offline and no new destination can be chosen, an active campaign
still inspects its saved field army; waiting reinforcements must not idle it.
If no building target is visible but enemy structure trim is clipped at a view
edge, a short camera-only arrow-key pan can reveal it. The visible formation must
have room to stay in view, and fresh frames restart the clear-ground check before
any movement or firing decision.
Army loss verification checks the battle view, then centers on the saved army if
that view is empty. An empty destination alone cannot establish that the traveling
army died.
Looking only at an empty enemy destination can miss fights along the route when
small Allied minimap dots do not establish a front.
After repeated clear views with nearby escorts, Prisms can advance in short steps
toward a still-confirmed enemy radar objective even before a building enters view.
That route supplies movement direction only; firing still requires a visible building.
After Allied radar and power recover, repeated complete infantry and vehicle lists
can confirm Battle Lab loss when all of its exclusive options remain absent.
This clears its stale placed count so the normal build plan can replace it. Power
outages, partial lists and a single missing lane cannot establish that loss.
Allied repair entry needs 300 credits; the old 1,200-credit headroom delayed repairs
while the native run was spending continuously on units and technology.
Radar recovery pauses siege spending, but a fresh raid can still fund an available
basic tank; rebuilding radar must not leave the base without mobile defenders.
Small-raider confirmation keeps the first nearby sighting's time. Captures arriving
faster than half a second no longer reset the timer forever; the native lone-infantry
fixture reproduces the missed attacker without lowering color thresholds.

When the enemy base remains unknown after Battle Lab, the Allied profile can fund
an available native Spy Satellite, retaining one tank's cost and requiring normal
power. This reveals shroud through the game's building; camera-only sweeps cannot.
The native 2026-10-14 HUD calibration also covers radar overlays at the unchanged
70% recognition threshold.

Missing minimap camera geometry now stops staging instead of raising an exception.
The native Allied radar activation flash is rejected before recording a home
anchor. Its emblem produced a false 132-by-30 frame one capture before the real
105-by-109 minimap, putting home far from the actual base and misclassifying nearby
raiders as a distant enemy base.

K repair is restricted to freshly recognized owned buildings in the side's repair list:
construction yards, power buildings and both sides' War Factories. Healthy and burning
factory appearances and selected diagonal bars were recorded in the sandbox. Allied
repair scans use the compressed-frame house-color reader for JPEG browser screenshots.
The Allied blue inactive and cyan active repair icons are calibrated separately from
the Soviet red inactive and yellow active indicators.
A live Iraq-versus-France run issued `repair-requested` for a yellow-health factory,
exited repair mode and later showed a full green bar while vehicle production continued.
A newly selected diagonal health bar must show yellow or red before the repair click;
right-click exits the mode and both mode icons must verify inactive before other actions.
Burning Allied power plants have a separate recorded appearance. Their calibrated
health line can use a ten-pixel selected pip segment; unknown bars still require
twenty pixels, and the other calibrated building bars retain their existing minimum.
Each side's `playbook.repair` sets starting cash, health bands and scan timing.
Only one repair attempt per building is allowed per run, since repeat clicks can cancel
an ongoing repair. Scans can interrupt a sidebar observation, but yield to production,
combat and ready building placement. The cash
threshold controls entry, not continuing repair expenditure under fire. Refinery repair
was tested manually; autonomous refinery recognition still needs live validation.

A finished ground unit can show native `Ready` when vehicles obstruct the factory exit,
including a Prism tank without a queue numeral. Ready text is sufficient evidence;
requiring the numeral hid that blocked queue and allowed more orders behind it.
Two fresh readings at least three seconds apart trigger a vehicle gathering order outside
the nearby building sprites, with at most one retry every 20 seconds. Selection must
verify combat vehicles and remove visible home infantry first; miners and aircraft do
not trigger this response. Active assaults and recent threats take precedence.
The live Allied trial cleared a waiting Grizzly and resumed production while its
dogs and GIs remained at home. A ready unit does not block building-repair inspection.
After a confirmed factory loss, a funded and available replacement factory takes
precedence over routine low-power recovery. The Allied trial otherwise repeatedly
replaced reactors under a surviving tank's fire without restoring combat production.

Waypoint routes use the verified Z planning control. Each route is finite: Chrono Divide
v0.84 rejects duplicate nodes, so a closed patrol loop from the original RA2 manual is not
available. The player stops the selected team, queues the route, and releases planning mode.
Scouting no longer occupies the army task lane while units travel; arrival needs fresh visual
evidence. A small patrol can reuse three healthy vehicles and one infantry while leaving four
vehicles at home. It spends nothing on formation or replacement. Exact Rhino/Flak/dog identities
are not verified by health bars, so this squad is labeled by unit category. Wider scouting needs
surplus credits and a home reserve; fresh contact or damage gives defense priority.
With radar online, a camera sweep checks one revealed approach or large offscreen own
contact every 8 seconds and stores the fresh survey with its observation time. Radar
contacts can suggest buildings; their size never changes confirmed building counts. The sweep
clears selection before clicking the radar, and returns home. A visible contact gets a fresh
defense session before the camera leaves it. If radar reports a contact outside
the inspected approach, the sweep centers on that contact before checking fresh
pixels for defense orders. During defense, an offscreen radar
contact also moves the camera to the attacker for a full defense session before routine
scouting or census work, even if no defenders remain. Orders
require fresh visible enemies; home infantry stays on guard. Construction and combat take
priority. Camera movement does not uncover shroud; that still needs surviving scouts.

Bindings and tactics were checked against the [RA2 manual](https://static.wikia.nocookie.net/cnc_gamepedia_en/images/9/94/Command_%26_Conquer_-_Red_Alert_2.pdf/revision/latest)
and the [user's hotkey guide](https://steamcommunity.com/sharedfiles/filedetails/?id=3175448213),
then tested in the isolated Chrono Divide browser. Ctrl+Shift is attack-move; Ctrl alone is force
fire and is not used for uncertain targets. Wins require observed natural match completion
and captured result evidence; aborted or interrupted trials do not count.

The desktop controller still needs the foreground game window. `FramePlayer` separates
perception and decisions from input for isolated browser testing; it is an API, not a
background mode for the desktop controller. Its caller must execute each transaction once,
acknowledge it on the next screenshot, and stop on transport errors. Live browser input is
still being validated. Browser adapters must pace hover and modifier clicks through a game
frame, as the native executor does; instant Shift-clicks can become move orders instead of
deselection. Assault formation re-reads guards after each click and requires selected vehicles.
Pointer parking must actually move the browser pointer: a zero-distance scroll is a no-op
and leaves tooltips covering the next health-bar observation. The sandbox adapter verifies
that parking on unused HUD space preserves the selection.
Guard clicks use the ground tile 25 pixels below the health bar; a sprite-center click can
miss the unit and move the whole army. A remembered battle view expires from its own last
selected-unit observation, independently of newer radar contacts elsewhere.
Once the army has reached the target, repeated siege orders cannot extend the missing-army
grace period. After an earlier failed attack, a freshly verified team no larger than the
reinforcement batch (four vehicles in these profiles) regroups before another assault.
Wave counts and guess deadlines start after the attack order is acknowledged, not at task start.
For JPEG screenshots, pass `compressed=True` to enable the tested
credits, queue counts, selected-unit census, enemy-color, and radar readers for compressed pixels.
HUD anchors still need to match the browser capture; native desktop defaults stay unchanged.
Combat uses attack-move for units and Ctrl attack for building-sized enemy-color targets.
The crowded Soviet field force ignored contextual clicks until a Ctrl attack on the
visible construction-yard footprint destroyed it. Building click targets still need live
verification for other clients and appearances. On the verified Allied run, clicks at
the lower colored edge of enemy buildings acquired their ground footprint and destroyed
reactors, production buildings and defenses.

Pointer: Chrono Divide profiles use relative input under pointer lock (`relative_motion: true`).
An injected absolute OS cursor can remain pinned at Chrome's window edge, blocking calibration
and every building click. Relative probes do not depend on OS cursor recentering. Motion still
goes in steps of at most 120 px; calibration checks gain and linearity before construction.
Clients that recenter absolute input can retain the older path with `relative_motion: false`.
The calibration regression tests cover both paths and a cursor pinned at the window edge.

Chrome can be windowed if the complete canvas fits. Capture failures and unrecognized HUDs
show a **Cannot proceed** inspector hint and block game input. HUD matching requires 70%;
the hint reports measured scores and the threshold. Allied HUD samples include the native
desktop view before the MCV deploys as well as the existing browser samples.

Verified on Chrono Divide at 100% zoom and a 1024x768 view: the Soviet sidebar with
red us / blue opponent, and the Allied sidebar with blue us / red opponent.
Other view sizes use the derived layout, checked
on synthetic frames only; another client, UI scale, or stretched window needs
a taught profile (Setup), and the original game has not been tried. The menu screens (autoplay,
score screen) are known at 1024x768 only. Placement reads the footprint preview around the base after
pressing H. Recognition rejects anything it does not know instead of guessing; unknown
cameos (for example vehicles not yet seen) are learned from recordings with
`tools/cameos.py`. The home army is counted by selected-unit health bars before an attack.
Missing radar dots during an attack trigger a fresh team view; two empty observations are
required before declaring the army lost. Surviving teams retain a camera focus even when
their tiny radar dots disappear. The last refinery's loss can also be inferred when its previously
observed build options disappear across repeated complete sidebar readings. Unknown icons
prevent that inference, and rebuilding still requires funds.
