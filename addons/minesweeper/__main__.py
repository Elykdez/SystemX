"""Unified Minesweeper inspector: screen pixels or supported browser labels."""

import sys

from .browser_player import Controls  # Keep the existing console controls import available.


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Existing unattended browser commands keep their console lifecycle.
    if '--console' in argv or '--autoplay' in argv:
        from .browser_player import main as browser_main
        return browser_main([arg for arg in argv if arg != '--console'])
    from .desktop_player import main as inspector_main
    if any(arg == '--url' or arg.startswith('--url=') for arg in argv) and not any(
            arg == '--interface' or arg.startswith('--interface=') for arg in argv):
        argv = ['--interface', 'browser', *argv]
    return inspector_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
