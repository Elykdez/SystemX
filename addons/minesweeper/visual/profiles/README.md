# Appearance profiles

`boyfriends-arcade.json` contains visible tile samples from Boyfriend's Arcade
Collection on Windows at 96 DPI, with a 447 x 447 pixel beginner board. Load it
with the desktop controller's Load profile button, then calibrate the current
window and grid. It covers hidden cells, empty cells, flags, and digits 1-4.
Digits 5-8 and mine tiles remain unsupported and safely pause for teaching.
The profile includes pixel-alignment variants observed during live play;
another scale or theme may require new samples.

`classic-web.json` contains normalized 32 x 32 BGR cell samples captured from
the user's locally hosted Minesweeper page. It uses visible rendered pixels,
not hidden mines or application state. Samples cover the classic Segoe UI theme:
covered, empty, flags, normal/exploded mines, and digits 1-4.

Digit 4 was explicitly taught after it paused the supervised validation game.
Digits 5-8 and different themes require teaching samples in the desktop controller.
Each label supports up to 32 distinct samples. Duplicate samples are ignored.
The profile format stores similarity and separation thresholds, not statistical
confidence probabilities. Unsupported or ambiguous cells must remain unknown.

Do not claim broad accuracy from the regression fixtures alone. Some of those
fixtures contributed samples to this profile. See the repository's desktop
validation report for the exact test provenance and outstanding checks.
