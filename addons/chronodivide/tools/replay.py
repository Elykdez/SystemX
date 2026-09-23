"""Replay a recording through perception, world model, and policy (recommend only, no input).

Prints what the agent observed and what it would do at each recorded moment, so the
full decision path can be tested offline, optionally with the real System One server.
"""

import argparse
import json
import time

from .. import playbook
from ..army import Army
from ..broker import Broker, server_routes
from ..intel import Intel
from ..intent import StandingIntent
from ..perception.reader import Perception
from ..reflex import Reflex
from ..skills import Context
from ..world import World
from .sheet import canvas_frames


def replay(folder, server=None, every=1, limit=None, quiet=False):
    clock = [0.0]
    world = World(clock=lambda: clock[0])
    perception = Perception()
    intent = StandingIntent()
    book = playbook.load()
    broker = Broker(server_routes(server), memo_seconds=book.systemone['memo_seconds']) if server else None
    ctx = Context(perception.layout, world)
    army = Army(ctx, None, intent, clock=lambda: clock[0], intel=Intel(lambda: clock[0], book=book), book=book)
    reflex = Reflex(ctx, broker=broker, intent=intent, clock=lambda: clock[0], army=army, book=book)
    timeline = []
    seen = 0
    timings = []
    for number, t, canvas in canvas_frames(folder):
        if canvas.shape[:2] != (768, 1024) or number % every:
            continue
        clock[0] = float(t or 0)
        started = time.perf_counter()
        mode, readings, lane = perception.read(canvas, world, clock[0])
        ctx.last_canvas = canvas
        radar = world.radar.value
        if radar is not None and radar.online and ctx.radar_home is None and radar.own:
            ctx.radar_home = radar.own[0][1]
        snapshot = world.snapshot()
        plan = reflex.step(snapshot, now=clock[0], act=False)
        order = army.step(snapshot, now=clock[0], act=False)
        if broker:
            time.sleep(0.05)   # let a pending System One answer land before the next frame
        timings.append((time.perf_counter() - started) * 1000)
        row = dict(frame=number, t=t, mode=mode['mode'], credits=snapshot['credits'], tab=snapshot['tab'],
                   lanes={k: f"{v['status']}{'/' + v['item'] if v['item'] else ''}"
                          f"{' ' + format(v['progress'], '.0%') if v['progress'] is not None else ''}"
                          for k, v in snapshot['lanes'].items() if v['age'] is not None},
                   plan=plan[0] if plan else None, objective=intent.objective, posture=intent.posture,
                   intel=intent.intel, enemy_base=army.intel.enemy_base,
                   army=order, radar=(radar.online and dict(enemy=len(radar.enemy), explored=round(radar.explored_share, 2)))
                   if radar is not None else None)
        timeline.append(row)
        if not quiet:
            print(json.dumps(row), flush=True)
        seen += 1
        if limit and seen >= limit:
            break
    summary = dict(frames=seen, perceive_policy_ms_median=sorted(timings)[len(timings) // 2] if timings else None,
                   systemone=broker.summary() if broker else None, intent=intent.view())
    if broker:
        broker.close()
    return timeline, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    parser.add_argument('--server', help='System One server URL for live decisions (omit for rules only).')
    parser.add_argument('--every', type=int, default=1)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    _, summary = replay(args.folder, args.server, args.every, args.limit)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == '__main__':
    main()
