"""Chrono Divide desktop player.

    python -m addons.chronodivide                   controller window (watch / recommend / execute)
    python -m addons.chronodivide watch --seconds 30 headless, read-only status lines
    python -m addons.chronodivide recommend          headless: also asks System One, never sends input
    python -m addons.chronodivide record             read-only frame recorder

Unattended runs (controller window, no clicks needed; the agent arms itself, resumes
after physical input stops, and lets the game capture the pointer again):
    python -m addons.chronodivide ui --autostart --unattended --front --at 1930,10
"""

import argparse
import json
import threading
import time


def headless(mode, seconds, server, policy, record_fps, strategy=None):
    from .player import Player
    player = Player(mode=mode, server=server, policy=policy, record_fps=record_fps, strategy=strategy)
    thread = threading.Thread(target=player.run, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        while time.monotonic() - started < seconds and thread.is_alive():
            time.sleep(1.0)
            state = player.view_state()
            world = state['world']
            lanes = {k: f"{v['status']}{'/' + v['item'] if v['item'] else ''}"
                     f"{' ' + format(v['progress'], '.0%') if v['progress'] is not None else ''}"
                     for k, v in world['lanes'].items() if v['age'] is not None}
            print(json.dumps(dict(t=round(time.monotonic() - started), state=state['status']['state'],
                                  reason=state['status']['reason'][:70], mode=world['mode'],
                                  credits=world['credits'], power=(world['power'] or {}).get('level'),
                                  tab=world['tab'], lanes=lanes, plan=state['plan'],
                                  objective=state['intent']['objective'], ms=state['timings_ms'])), flush=True)
    finally:
        player.stop()
        thread.join(5)
        print('run folder:', player.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', default='ui', choices=['ui', 'watch', 'recommend', 'record'])
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--server', default='http://127.0.0.1:8000')
    parser.add_argument('--policy', choices=['systemone', 'rules'], default='systemone')
    parser.add_argument('--strategy', help='Faction strategy JSON (default: Soviet).')
    parser.add_argument('--record-fps', type=float, default=0.0)
    parser.add_argument('--mode', choices=['watch', 'recommend', 'execute'], default='execute',
                        help='Controller mode selected at start.')
    parser.add_argument('--autostart', action='store_true', help='Controller: press Start automatically.')
    parser.add_argument('--unattended', action='store_true',
                        help='Controller in execute mode: arm automatically and resume after physical input stops.')
    parser.add_argument('--resume-after', type=float, default=8.0, help='Quiet seconds before resuming.')
    parser.add_argument('--front', action='store_true', help='Bring the game window to the front after starting.')
    parser.add_argument('--autoplay', action='store_true',
                        help='Unattended, and also start matches from the setup and score screens; exit when a '
                             'match ends (play-autoplay.bat starts the next one).')
    parser.add_argument('--at', type=lambda s: tuple(int(v) for v in s.split(',')), default=None,
                        help='Controller window position X,Y (desktop pixels).')
    args = parser.parse_args()
    if args.autoplay:
        args.autostart = args.unattended = args.front = True
    if args.command == 'ui':
        from .controller import main as controller
        controller(args)
    elif args.command == 'record':
        from .recorder import main as recorder
        recorder()
    else:
        headless(args.command, args.seconds, args.server, args.policy, args.record_fps, args.strategy)


if __name__ == '__main__':
    main()
