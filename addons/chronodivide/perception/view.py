"""Where the tactical view lies on the canvas, for code that is not handed the layout.

The player sets it once from its layout when it attaches to a game (`use`), so every reader
of the battlefield follows the game's view size. Tests and tools keep the default: the
tactical view of the 1024x768 layout.
"""

TACTICAL = (0, 0, 850, 736)


def use(value):
    """Take the tactical view from a layout (or an (x, y, w, h) tuple); returns the previous one."""
    global TACTICAL
    previous = TACTICAL
    box = getattr(value, 'tactical', value)
    TACTICAL = (box.x, box.y, box.w, box.h) if hasattr(box, 'w') else tuple(box)
    return previous


def tactical(region=None):
    """The region to read: the one given, else the attached game's tactical view."""
    return TACTICAL if region is None else region
