"""Conservative visible-board deductions and explicitly uncertain guesses.

Flags are treated as claims of mines. Guarantees hold if the observed numbers and
existing flags are correct. The planner has no access to concealed board state.
"""

from .board import Board, Candidate


def _neighbors(row, col, rows, cols):
    return {
        (r, c)
        for r in range(max(0, row - 1), min(rows, row + 2))
        for c in range(max(0, col - 1), min(cols, col + 2))
        if (r, c) != (row, col)
    }


def _deductions(constraints):
    safe, mines = set(), set()
    while True:
        reduced = {}
        for positions, count in constraints:
            count -= len(positions & mines)
            positions = frozenset(positions - safe - mines)
            if not 0 <= count <= len(positions):
                raise ValueError("Visible mine constraints contradict one another")
            if positions in reduced and reduced[positions] != count:
                raise ValueError("Visible numbers disagree on the same cells")
            if positions:
                reduced[positions] = count
        new_safe, new_mines = set(), set()
        items = list(reduced.items())
        for positions, count in items:
            if count == 0:
                new_safe.update(positions)
            elif count == len(positions):
                new_mines.update(positions)
            for other, other_count in items:
                if positions < other:
                    difference = other - positions
                    remaining = other_count - count
                    if not 0 <= remaining <= len(difference):
                        raise ValueError("Visible subset constraints contradict one another")
                    if remaining == 0:
                        new_safe.update(difference)
                    elif remaining == len(difference):
                        new_mines.update(difference)
        if new_safe & new_mines:
            raise ValueError("A cell is constrained to be both safe and a mine")
        if not new_safe and not new_mines:
            return safe, mines
        safe.update(new_safe)
        mines.update(new_mines)


def candidates(board: Board, max_candidates: int = 12) -> list[Candidate]:
    """Return certain actions first; otherwise a spatially varied set of guesses.

    Probabilities are only supplied for proven safe cells (0) and mines (1).
    Guaranteed-safe flags describe the safety of the action, not of the cell.
    """
    if type(max_candidates) is not int or max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer")
    if board.status != "playing":
        return []
    visible = {(cell.row, cell.col): cell.state for cell in board.cells}
    hidden = {position for position, state in visible.items() if state == "hidden"}
    flags = {position for position, state in visible.items() if state == "flag"}
    constraints = []
    frontier = set()
    for (row, col), state in visible.items():
        if state.isdigit():
            neighbors = _neighbors(row, col, board.rows, board.cols)
            unknown = frozenset(neighbors & hidden)
            constraints.append((unknown, int(state) - len(neighbors & flags)))
            frontier.update(unknown)
    if board.mines_total is not None:
        constraints.append((frozenset(hidden), board.mines_total - len(flags)))
    safe, mines = _deductions(constraints)

    def action(position, button, reason, probability):
        row, col = position
        return Candidate(f"{button}_{row}_{col}", row, col, button, reason,
                         probability is not None, probability)

    certain = [action(p, "left", "Visible number constraints prove this cell safe.", 0.0)
               for p in sorted(safe)]
    certain += [action(p, "right", "Visible number constraints prove this cell is a mine.", 1.0)
                for p in sorted(mines)]
    if certain:
        return certain[:max_candidates]
    if not hidden:
        return []

    center = ((board.rows - 1) / 2, (board.cols - 1) / 2)

    def distance(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    # Include a central option and then spread choices across visible frontiers.
    pool = set(hidden)
    chosen = [min(pool, key=lambda p: (distance(p, center), p))]
    pool.remove(chosen[0])
    while pool and len(chosen) < max_candidates:
        position = min(pool, key=lambda p: (
            p not in frontier, -min(distance(p, other) for other in chosen), p
        ))
        chosen.append(position)
        pool.remove(position)
    return [action(p, "left", "No certain move was deduced; mine risk is unknown.", None)
            for p in chosen]
