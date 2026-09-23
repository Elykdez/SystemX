# Appearance profiles

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
