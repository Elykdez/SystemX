"""A thread-owned browser adapter for the shared Minesweeper inspector."""

from datetime import datetime
import json
from pathlib import Path
import queue
import threading

from .browser_driver import MinesweeperBrowser
from .decision import SystemOneDecision
from .session import MinesweeperSession


ROOT = Path(__file__).resolve().parents[2]


class BrowserWorker:
    def __init__(self, url, difficulty, server, gate, notify):
        self.url, self.difficulty, self.server = url, difficulty, server
        self.gate, self.notify = gate, notify
        self.commands = queue.Queue()
        self.closed = threading.Event()
        self.busy = True
        self.connected = False
        self.thread = threading.Thread(target=self.work, name='minesweeper-browser', daemon=True)

    def start(self):
        self.thread.start()

    def play(self, generation, single, policy, max_steps, delay):
        self.busy = True
        self.commands.put(('play', (generation, single, policy, max_steps, delay)))

    def capture(self):
        self.busy = True
        self.commands.put(('capture', None))

    def close(self):
        self.closed.set()
        self.gate.pause()
        self.commands.put(('close', None))

    def preview(self, page, driver):
        board = driver.observe()
        self.notify('preview', (board, page.screenshot()))
        return board

    def play_session(self, page, driver, output, generation, single, policy, max_steps, delay):
        brain = None
        session = None
        try:
            driver = MinesweeperBrowser(page, can_act=lambda: not self.closed.is_set() and self.gate.allowed(generation))
            if policy == 'systemone':
                brain = SystemOneDecision(self.server)
                brain.wait_ready(timeout=3)
            with (output / 'actions.jsonl').open('a', encoding='utf-8') as log:
                session = MinesweeperSession(driver, brain, log,
                    lambda: not self.closed.is_set() and self.gate.allowed(generation))
                result = dict(status='paused', steps=0, reason='Paused before input.')
                while not self.closed.is_set() and self.gate.allowed(generation) and session.steps < max_steps:
                    result = session.step()
                    self.notify('step', result)
                    self.preview(page, driver)
                    if result['status'] in ('won', 'lost', 'stalled', 'paused'):
                        break
                    if single:
                        result = dict(status='paused', steps=session.steps, reason='Single step complete.')
                        break
                    page.wait_for_timeout(delay * 1000)
                else:
                    result = dict(status='paused', steps=session.steps, reason='Step limit or pause requested.')
            return result
        except Exception as error:
            return dict(status='paused', steps=session.steps if session else 0, reason=str(error))
        finally:
            self.gate.pause()
            if brain:
                brain.close()

    def work(self):
        try:
            from playwright.sync_api import sync_playwright
            output = ROOT / '.cache/minesweeper/browser' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            output.mkdir(parents=True)
            with sync_playwright() as playwright:
                context = playwright.chromium.launch_persistent_context(
                    str(ROOT / '.cache/minesweeper/chrome-profile'), channel='chrome',
                    headless=False, viewport={'width': 1280, 'height': 900})
                try:
                    page = context.pages[0] if context.pages else context.new_page()
                    page.goto(self.url, wait_until='domcontentloaded', timeout=30000)
                    page.get_by_role('table', name='Minesweeper board').wait_for(state='visible', timeout=30000)
                    page.get_by_label('Difficulty', exact=True).select_option(self.difficulty)
                    driver = MinesweeperBrowser(page)
                    self.preview(page, driver)
                    self.connected, self.busy = True, False
                    self.notify('ready', str(output))
                    while not self.closed.is_set():
                        try:
                            command, values = self.commands.get(timeout=.1)
                        except queue.Empty:
                            page.wait_for_timeout(50)
                            continue
                        if command == 'close':
                            break
                        try:
                            if command == 'capture':
                                self.preview(page, driver)
                            else:
                                result = self.play_session(page, driver, output, *values)
                                (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                                self.notify('result', result)
                        except Exception as error:
                            self.notify('result', dict(status='paused', reason=str(error)))
                        finally:
                            self.busy = False
                finally:
                    context.close()
        except Exception as error:
            self.notify('error', str(error))
        finally:
            self.connected = self.busy = False
            self.notify('closed', None)
