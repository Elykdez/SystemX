"""Stage-0 System One capacity benchmark with synthetic RTS-sized packets. Needs no game.

Measures server-reported and client-observed latency per question type and packet
size, so the broker's question budget comes from measurements, not assumptions.
"""

from model_config import MODEL_NAME

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import time

import httpx

ROOT = Path(__file__).resolve().parents[3]

OBJECTIVES = {
    'expand_economy': 'Build another ore refinery and miners to raise income.',
    'tech_up': 'Build radar and battle lab to unlock stronger units.',
    'build_army': 'Keep factories producing combat units at the assembly point.',
    'defend_base': 'Hold the army near the base entrance until the threat passes.',
    'attack_enemy_base': 'Send the main army to attack the enemy construction yard.',
    'harass_miners': 'Send a fast group to destroy enemy ore miners.',
    'scout': 'Send one cheap unit to reveal the enemy base.',
    'protect_miners': 'Move the army to cover our ore field.',
}
PRODUCTION = {
    'rhino_tank': 'Rhino Tank, 900 credits: strong main battle tank against vehicles.',
    'flak_track': 'Flak Track, 500 credits: anti-air and anti-infantry transport.',
    'war_miner': 'War Miner, 1400 credits: armed ore collector that raises income.',
    'v3_launcher': 'V3 Launcher, 800 credits: long-range artillery against buildings.',
    'conscript': 'Conscript, 100 credits: cheap infantry.',
    'tesla_trooper': 'Tesla Trooper, 500 credits: anti-vehicle infantry.',
}
ORDERS = {
    'attack_move_enemy_base': 'Attack-move group 1 to the enemy base in the north-east.',
    'hold_choke': 'Hold group 1 at the bridge choke west of our base.',
    'retreat_home': 'Retreat group 1 to our base for repair.',
    'intercept_raid': 'Attack-move group 1 to the enemy tanks seen near our ore field.',
    'keep_orders': 'Keep the current order and observe.',
}


def state(padding_lines):
    memory = [f'Sector {chr(65 + i % 8)}{i % 6}: enemy {("tanks", "infantry", "miner", "defense")[i % 4]} '
              f'seen {15 + i * 7} s ago, count about {1 + i % 5}.' for i in range(padding_lines)]
    return {
        'game': 'Red Alert 2 skirmish (Chrono Divide), Soviet vs Allied easy AI',
        'time': '06:40', 'credits': 3450, 'power': 'normal (+60)',
        'base': 'Construction Yard, 2 Tesla Reactors, Barracks, Ore Refinery, War Factory; radar not built.',
        'economy': '2 War Miners harvesting the home ore field; income about 900 per minute.',
        'army': 'Group 1: 6 Rhino Tanks healthy at assembly point; Group 2: 4 Conscripts at base.',
        'threats': 'None visible now. Last enemy attack 90 s ago, repelled.',
        'objective': 'build_army (set 40 s ago)',
        'memory': memory,
    }


QUESTIONS = {
    'strategy': lambda: {'type': 'choice', 'instructions': 'Choose the next strategic objective for our side.',
                         'criteria': OBJECTIVES},
    'production': lambda: {'type': 'choice', 'instructions': 'Choose the next unit for the war factory queue.',
                           'criteria': PRODUCTION},
    'group_order': lambda: {'type': 'choice', 'instructions': 'Choose the order for group 1.', 'criteria': ORDERS},
    'aggression': lambda: {'type': 'score', 'instructions': 'How aggressively should the army act now?',
                           'criteria': ['Turtle at base.', 'Defend nearby.', 'Contest map.', 'Pressure enemy.', 'All-in attack.']},
    'threat': lambda: {'type': 'noul', 'instructions': 'Is our base under a threat that needs the army now?',
                       'criteria': {'true': 'Yes, enemies are attacking the base.', 'false': 'No active threat.'}},
}


def percentile(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def summarize(samples):
    server = [s['server_ms'] for s in samples]
    client = [s['client_ms'] for s in samples]
    return dict(n=len(samples), tokens=round(statistics.median(s['tokens'] for s in samples)),
                server_p50=round(statistics.median(server), 2), server_p95=round(percentile(server, 95), 2),
                server_max=round(max(server), 2), client_p50=round(statistics.median(client), 2),
                client_p95=round(percentile(client, 95), 2))


def calibrate_padding(client, target_tokens):
    """Find memory padding giving about target_tokens for the strategy question."""
    low, high = 0, 120
    best = 0
    while low <= high:
        mid = (low + high) // 2
        body = dict(model=MODEL_NAME, state=state(mid), questions={'q': QUESTIONS['strategy']()})
        response = client.post('/v1/systemone', json=body)
        if response.status_code == 422:
            high = mid - 1
            continue
        response.raise_for_status()
        tokens = response.json()['usage']['input_tokens']
        if tokens <= target_tokens:
            best, low = mid, mid + 1
        else:
            high = mid - 1
    return best


def run(server, repeats, sizes):
    client = httpx.Client(base_url=server, timeout=30)
    report = dict(time=datetime.now().isoformat(timespec='seconds'), server=server,
                  health=client.get('/health').json(), results=[])
    for target in sizes:
        padding = calibrate_padding(client, target)
        for name, make in QUESTIONS.items():
            samples = []
            for _ in range(repeats):
                body = dict(model=MODEL_NAME, state=state(padding), questions={name: make()})
                started = time.perf_counter()
                response = client.post('/v1/systemone', json=body)
                elapsed = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                result = response.json()
                samples.append(dict(server_ms=result['latency_ms'], client_ms=elapsed,
                                    tokens=result['usage']['input_tokens']))
            row = dict(target_tokens=target, padding=padding, question=name, **summarize(samples))
            report['results'].append(row)
            print(json.dumps(row), flush=True)
        # One request carrying strategy + aggression + production, the planned strategic packet.
        samples = []
        for _ in range(max(5, repeats // 2)):
            body = dict(model=MODEL_NAME, state=state(padding),
                        questions={k: QUESTIONS[k]() for k in ('strategy', 'aggression', 'production')})
            started = time.perf_counter()
            response = client.post('/v1/systemone', json=body)
            elapsed = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            result = response.json()
            samples.append(dict(server_ms=result['latency_ms'], client_ms=elapsed, tokens=result['usage']['input_tokens']))
        row = dict(target_tokens=target, padding=padding, question='strategic_packet_3q', **summarize(samples))
        report['results'].append(row)
        print(json.dumps(row), flush=True)
    client.close()
    return report


def worker_scaling(server, seconds, target_tokens=800):
    """Aggregate throughput and packet latency using the server's parallel workers.

    1..N client routes each keep one request in flight on the same endpoint, then a
    3-question strategic packet is timed with and without per-question splitting.
    """
    import threading
    probe = httpx.Client(base_url=server, timeout=30)
    workers = probe.get('/health').json().get('workers', {}).get('ready', 1)
    padding = calibrate_padding(probe, target_tokens)
    body = dict(model=MODEL_NAME, state=state(padding), questions={'q': QUESTIONS['production']()})
    report = dict(time=datetime.now().isoformat(timespec='seconds'), server=server, workers=workers, results=[])
    for count in range(1, workers + 2):
        samples, busy, lock = [], [0], threading.Lock()
        stop = time.perf_counter() + seconds

        def route():
            with httpx.Client(base_url=server, timeout=30) as client:
                while time.perf_counter() < stop:
                    started = time.perf_counter()
                    response = client.post('/v1/systemone', json=body)
                    elapsed = (time.perf_counter() - started) * 1000
                    if response.status_code == 429:
                        with lock:
                            busy[0] += 1
                        time.sleep(0.002)
                        continue
                    response.raise_for_status()
                    result = response.json()
                    with lock:
                        samples.append(dict(server_ms=result['latency_ms'], client_ms=elapsed,
                                            tokens=result['usage']['input_tokens']))

        threads = [threading.Thread(target=route) for _ in range(count)]
        began = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = time.perf_counter() - began
        row = dict(routes=count, questions_per_second=round(len(samples) / wall, 1), busy_429=busy[0],
                   **summarize(samples))
        report['results'].append(row)
        print(json.dumps(row), flush=True)
    packet = {k: QUESTIONS[k]() for k in ('strategy', 'aggression', 'production')}
    for parallel in (False, True):
        samples, answers = [], None
        for _ in range(20):
            started = time.perf_counter()
            response = probe.post('/v1/systemone', json=dict(state=state(padding), questions=packet, parallel=parallel))
            elapsed = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            result = response.json()
            answers = {k: v.get('choice', v.get('score')) for k, v in result['answers'].items()}
            samples.append(dict(server_ms=result['latency_ms'], client_ms=elapsed, tokens=result['usage']['input_tokens']))
        row = dict(packet='strategic_3q', parallel=parallel, answers=answers, **summarize(samples))
        report['results'].append(row)
        print(json.dumps(row), flush=True)
    probe.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='http://127.0.0.1:8000')
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--sizes', type=int, nargs='+', default=[300, 800, 1600])
    parser.add_argument('--workers', action='store_true', help='Measure the server\'s parallel workers instead.')
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = worker_scaling(args.server, args.seconds) if args.workers else run(args.server, args.repeats, args.sizes)
    output = args.output or ROOT / '.cache' / 'chronodivide' / 'benchmarks' / f'systemone-capacity-{datetime.now():%Y%m%d-%H%M%S}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('saved', output)


if __name__ == '__main__':
    main()
