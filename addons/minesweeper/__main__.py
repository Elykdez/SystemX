"""Visible Chrome Minesweeper player; run with python -m addons.minesweeper."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
from urllib.parse import urlsplit

from .browser_driver import MinesweeperBrowser
from .decision import VonDecision
from .session import MinesweeperSession, StaleObservation

ROOT = Path(__file__).resolve().parents[2]


class Controls:
    def __init__(self, running=False):
        self.commands = queue.Queue()
        self.running = running
        self.single = False
        self.stopped = False

    def listen(self):
        def reader():
            while True:
                try:
                    self.commands.put(input().strip().lower())
                except EOFError:
                    return
        threading.Thread(target=reader, daemon=True).start()

    def poll(self):
        while not self.commands.empty():
            command = self.commands.get_nowait()
            if command in ("q", "quit", "stop"):
                self.stopped = True
            elif command in ("p", "pause"):
                self.running = self.single = False
                print("Paused.", flush=True)
            elif command in ("r", "run"):
                self.running = True
            elif command in ("s", "step"):
                self.running = False
                self.single = True
        return not self.stopped and (self.running or self.single)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Minesweeper game URL; prompts when omitted")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--policy", choices=("von", "logic"), default="von")
    parser.add_argument("--difficulty", choices=("beginner", "intermediate", "expert"), default="beginner")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--delay", type=float, default=.35)
    parser.add_argument("--autoplay", action="store_true", help="Start immediately after setup instead of waiting for r")
    args = parser.parse_args(argv)
    if args.max_steps < 1 or args.delay < .1:
        parser.error("Use at least 1 step and a delay of at least 0.1 seconds.")
    if args.url is None:
        try:
            args.url = input("Minesweeper game URL (http:// or https://): ")
        except EOFError:
            parser.error("No URL entered. Use --url when running without interactive input.")
        except KeyboardInterrupt:
            print("\nStopped by user.")
            return 0
    args.url = args.url.strip()
    try:
        target = urlsplit(args.url)
        valid_url = target.scheme in ("http", "https") and bool(target.hostname)
    except ValueError:
        valid_url = False
    if not valid_url:
        parser.error("Enter a complete http:// or https:// game URL.")
    from playwright.sync_api import sync_playwright, Error

    run_dir = ROOT / ".cache" / "minesweeper" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True)
    brain = VonDecision(args.server) if args.policy == "von" else None
    try:
        if brain:
            brain.wait_ready(progress=lambda message: print(message, flush=True))
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                str(ROOT / ".cache" / "minesweeper" / "chrome-profile"),
                channel="chrome", headless=False, viewport={"width": 1280, "height": 900},
            )
            try:
                page = context.pages[0] if context.pages else context.new_page()
                try:
                    page.goto(args.url, wait_until="domcontentloaded")
                except Error as error:
                    if "ERR_CERT" not in str(error):
                        raise
                    input("Chrome rejected the certificate. Handle the warning yourself in that window, then press Enter here: ")
                page.get_by_role("table", name="Minesweeper board").wait_for(state="visible", timeout=30000)
                page.get_by_label("Difficulty", exact=True).select_option(args.difficulty)
                driver = MinesweeperBrowser(page)
                initial = driver.observe()
                print(f"Detected {initial.rows} x {initial.cols} cells. Policy: {args.policy}. Logs: {run_dir}", flush=True)
                page.screenshot(path=str(run_dir / "initial.png"))
                controls = Controls(args.autoplay)
                print("Type r + Enter to run, p to pause, s for one step, q to stop. Ctrl+C exits.", flush=True)
                controls.listen()
                with (run_dir / "actions.jsonl").open("w", encoding="utf-8") as log:
                    session = MinesweeperSession(driver, brain, log, controls.poll)
                    while session.steps < args.max_steps and not controls.stopped:
                        if not controls.poll():
                            page.wait_for_timeout(100)
                            continue
                        try:
                            event = session.step()
                        except StaleObservation as error:
                            controls.running = controls.single = False
                            print(str(error) + " Paused; type r to resume.", flush=True)
                            continue
                        controls.single = False
                        print(json.dumps({key: value for key, value in event.items()
                                          if key not in ("visible_before", "visible_after")}), flush=True)
                        if event["status"] in ("won", "lost", "stalled"):
                            break
                        page.wait_for_timeout(args.delay * 1000)
                page.screenshot(path=str(run_dir / "final.png"))
                print("Player stopped. No more input will be sent.", flush=True)
                if not args.autoplay:
                    print("Type q to close the player browser.", flush=True)
                    while not controls.stopped:
                        controls.poll()
                        page.wait_for_timeout(100)
            finally:
                context.close()
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as error:
        print(f"Player stopped: {error}", flush=True)
        return 1
    finally:
        if brain:
            brain.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
