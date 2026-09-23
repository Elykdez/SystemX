"""Window-based Minesweeper player. No DOM or browser automation dependency."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time

from desktop.target import Target, DesktopPause, enable_dpi, windows
from desktop.capture import Capture
from desktop.input import Gate, Hotkeys, Mouse
from desktop.inspector import Inspector, STATUS
from .desktop_driver import DesktopDriver
from .visual.recognition import Grid, Profile, LABELS, recognize, suggest_grid
from .decision import SystemOneDecision
from .session import MinesweeperSession
from .board import Box

ROOT = Path(__file__).resolve().parents[2]
PROFILES = Path(__file__).with_name('visual') / 'profiles'


def run_session(target, grid, profile, geometry, gate, generation, server, policy, output,
                notify=None, max_steps=500, trust_flags=False, on_step=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    brain = None
    capture = Capture(target)
    driver = DesktopDriver(capture, Mouse(target, gate), grid, profile, gate, generation, geometry, notify)
    result = {'status': 'paused', 'steps': 0}
    started = time.monotonic()
    try:
        driver.prepare()
        initial = driver.observe()
        if not trust_flags and any(c.state == 'flag' for c in initial.cells):
            raise DesktopPause('Existing flags require confirmation. Enable "Trust existing flags" to attach mid-game.')
        if policy == 'systemone':
            brain = SystemOneDecision(server)
            # The worker may finish this request after pause; the generation gate prevents input.
            health = brain.client.get('/health', timeout=3)
            health.raise_for_status()
            if not health.json().get('ready'):
                raise DesktopPause('Model is not ready. Start the server, wait for readiness, then resume.')
        with (output / 'actions.jsonl').open('a', encoding='utf-8') as log:
            session = MinesweeperSession(driver, brain, log, lambda: gate.allowed(generation))
            while session.steps < max_steps and gate.allowed(generation):
                result = session.step()
                if on_step:
                    on_step(result)
                result['steps'] = session.steps
                if result['status'] in ('won', 'lost', 'stalled', 'paused'):
                    break
                if gate.single:
                    result['status'] = 'paused'
                    result['reason'] = 'Single step complete.'
                    break
            else:
                result = dict(status='paused', steps=session.steps, reason='Step limit or pause requested.')
    except Exception as error:
        result = dict(status='paused', steps=locals().get('session').steps if 'session' in locals() else 0,
                      reason=str(error), error_type=type(error).__name__)
        if 'session' in locals() and hasattr(session, 'last_action'):
            result['last_attempt'] = session.last_action
    finally:
        gate.pause()
        if driver.last_frame is not None:
            from PIL import Image
            g = grid
            frame = driver.last_frame
            pixels = g.crop(frame.pixels, Box(g.x, g.y, g.width, g.height), frame.origin)[:, :, ::-1]
            Image.fromarray(pixels).save(output / 'last-board.png')
        if brain:
            brain.close()
        capture.close()
        result = {k: v for k, v in result.items() if k not in ('visible_before', 'visible_after')}
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
        result['profile'] = profile.name
        result['profile_revision'] = profile.revision
        (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


class Controller:
    def __init__(self, root, args):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root, self.args = tk, ttk, root, args
        self.gate, self.events = Gate(), queue.Queue()
        self.frame = self.grid = self.target = self.selected = None
        self.worker = None
        self.browser = None
        self.browser_image = self.board = self.last_step = None
        self.profile = Profile()
        self.details = []
        self.scale = 1.0
        root.title('SystemX Minesweeper / AI Inspector')
        root.geometry('1120x880')
        root.minsize(700, 680)
        self.inspector = Inspector(root, 'Classic Minesweeper / one solver, browser and visual adapters', 'MINESWEEPER')
        self.status = self.inspector.status
        self.status.set('Select a game window. Capture, mark its board, then check the preview.')
        body = self.inspector.controls
        adapter_row = self.adapter_row = ttk.Frame(body)
        adapter_row.pack(fill='x', pady=(0, 6))
        self.interface = tk.StringVar(value=getattr(args, 'interface', 'visual'))
        ttk.Label(adapter_row, text='Read and control via:').pack(side='left')
        for value, label in [('visual', 'Screen pixels (desktop or browser)'), ('browser', 'Browser labels')]:
            ttk.Radiobutton(adapter_row, text=label, variable=self.interface, value=value,
                            command=lambda: self.safe(self.change_interface)).pack(side='left', padx=8)
        self.browser_settings = ttk.Frame(body)
        self.url = tk.StringVar(value=getattr(args, 'url', None) or '')
        ttk.Label(self.browser_settings, text='Game URL').pack(side='left')
        ttk.Entry(self.browser_settings, textvariable=self.url, width=42).pack(side='left', fill='x', expand=True, padx=5)
        self.difficulty = tk.StringVar(value=getattr(args, 'difficulty', 'beginner'))
        ttk.Combobox(self.browser_settings, textvariable=self.difficulty, values=('beginner', 'intermediate', 'expert'),
                     state='readonly', width=13).pack(side='left')
        ttk.Button(self.browser_settings, text='Connect / Capture', command=lambda: self.safe(self.capture_browser)).pack(side='left', padx=5)
        self.visual_settings = ttk.Frame(body)
        self.target_text = tk.StringVar()
        header = ttk.Frame(self.visual_settings)
        header.pack(fill='x')
        self.targets = ttk.Combobox(header, textvariable=self.target_text, width=72, state='readonly')
        self.targets.pack(side='left', fill='x', expand=True)
        ttk.Button(header, text='Refresh windows', command=self.refresh).pack(side='left')
        ttk.Button(header, text='Capture', command=lambda: self.safe(self.capture)).pack(side='left')
        settings = ttk.Frame(self.visual_settings, padding=(0, 6, 0, 0))
        settings.pack(fill='x')
        self.rows, self.cols, self.mines = [tk.StringVar(value=v) for v in ('9', '9', '10')]
        for label, variable in [('Rows', self.rows), ('Columns', self.cols), ('Total mines (optional)', self.mines)]:
            ttk.Label(settings, text=label).pack(side='left', padx=3)
            ttk.Entry(settings, textvariable=variable, width=5).pack(side='left')
        ttk.Button(settings, text='Suggest grid', command=lambda: self.safe(self.suggest)).pack(side='left', padx=5)
        ttk.Button(settings, text='Check preview', command=lambda: self.safe(self.preview)).pack(side='left')
        self.canvas = self.inspector.canvas
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<ButtonRelease-1>', lambda event: self.safe(lambda: self.release(event)))
        self.canvas.bind('<Configure>', lambda event: self.draw() if self.frame is not None or self.browser_image is not None else None)
        teach = ttk.Frame(self.visual_settings, padding=(0, 6, 0, 0))
        teach.pack(fill='x')
        self.label = tk.StringVar(value='hidden')
        ttk.Label(teach, text='Click a cell to label it:').pack(side='left')
        ttk.Combobox(teach, textvariable=self.label, values=LABELS, width=9, state='readonly').pack(side='left')
        ttk.Button(teach, text='Teach cell', command=lambda: self.safe(self.teach)).pack(side='left', padx=3)
        ttk.Button(teach, text='Load profile', command=lambda: self.safe(self.load_profile)).pack(side='left')
        ttk.Button(teach, text='Save profile', command=lambda: self.safe(self.save_profile)).pack(side='left')
        ttk.Button(teach, text='New profile', command=self.new_profile).pack(side='left')
        ttk.Button(teach, text='Save calibration', command=lambda: self.safe(self.save_config)).pack(side='left')
        controls = ttk.Frame(body, padding=(0, 6, 0, 0))
        controls.pack(fill='x')
        self.trust = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text='Trust existing flags', variable=self.trust).pack(side='left')
        self.policy = tk.StringVar(value=args.policy)
        ttk.Combobox(controls, textvariable=self.policy, values=('systemone', 'logic'), width=8, state='readonly').pack(side='left')
        self.browser_actions = ttk.Frame(controls)
        ttk.Button(self.browser_actions, text='Run', command=lambda: self.safe(lambda: self.start(False))).pack(side='left', padx=4)
        ttk.Button(self.browser_actions, text='Step', command=lambda: self.safe(lambda: self.start(True))).pack(side='left')
        ttk.Label(controls, text='F8 Run / F9 Step / F10 Pause').pack(side='left', padx=12)
        ttk.Button(controls, text='Pause', command=lambda: self.hotkey('pause')).pack(side='right')
        ttk.Label(body, text='Visual: focus the game before F8/F9. Unknown cells block play. Browser labels require a supported page.',
                  style='Muted.TLabel', wraplength=1000).pack(fill='x', pady=(6, 0))
        self.hotkeys = Hotkeys(self.hotkey)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.refresh()
        default = PROFILES / 'classic-web.json'
        if default.exists():
            self.profile = Profile.load(default)
        self.change_interface()
        self.inspector.describe('PERCEPTION\nChoose screen pixels for desktop games or any visible browser board. '
                                'Use browser labels for a supported accessible page.\n\n'
                                'DECISION\nBoth adapters use the same classic Minesweeper solver and optional StartLux decisions.\n\n'
                                'CONTROL\nEvery move is revalidated and its result checked before the next move.')
        root.after(80, self.poll)

    def safe(self, function):
        try:
            function()
        except Exception as error:
            self.status.set(str(error))
            self.inspector.banner.configure(bg=STATUS['paused'])

    def ensure_idle(self):
        self.gate.pause()
        if self.worker and self.worker.is_alive():
            raise DesktopPause('Stopping the pending operation. Try again when it finishes.')
        if self.browser and self.browser.busy:
            raise DesktopPause('Stopping the browser operation. Wait for it to finish before changing setup.')

    def change_interface(self):
        try:
            self.ensure_idle()
        except DesktopPause:
            self.interface.set(self.active_interface)
            raise
        if self.browser:
            self.browser.close()
            self.browser = None
        self.frame = self.grid = self.selected = self.board = self.browser_image = self.last_step = None
        self.pending_preview = None
        self.trust.set(False)
        self.details = []
        self.canvas.delete('all')
        self.visual_settings.pack_forget()
        self.browser_settings.pack_forget()
        self.browser_actions.pack_forget()
        browser = self.interface.get() == 'browser'
        (self.browser_settings if browser else self.visual_settings).pack(fill='x', after=self.adapter_row)
        if browser:
            self.browser_actions.pack(side='left')
        self.inspector.badge.set('MINESWEEPER / ' + ('BROWSER LABELS' if browser else 'SCREEN PIXELS'))
        self.active_interface = self.interface.get()
        self.inspector.banner.configure(bg=STATUS['idle'])
        self.status.set('Enter a supported game URL and Connect.' if browser else
                        'Select a visible game window. Capture, mark its board, then check the preview.')

    def capture_browser(self):
        from urllib.parse import urlsplit
        from .browser_worker import BrowserWorker
        self.ensure_idle()
        if self.browser and self.browser.connected:
            self.browser.capture()
            return
        url = self.url.get().strip()
        target = urlsplit(url)
        if target.scheme not in ('http', 'https') or not target.hostname:
            raise ValueError('Enter a complete http:// or https:// game URL.')
        browser = BrowserWorker(url, self.difficulty.get(), self.args.server, self.gate,
                                lambda kind, value: self.events.put(('browser', (browser, kind, value))))
        self.browser = browser
        browser.start()
        self.status.set('Connecting to the browser. No game input until Run or Step.')

    def describe_board(self):
        board = self.board
        if board is None:
            self.inspector.describe('PERCEPTION\nThe board is uncertain. Teach unknown cells before play.\n\n'
                                    'CONTROL\nNo move is authorized without a complete recognized board.')
            return
        states = {(c.row, c.col): c.state for c in board.cells}
        symbols = {'hidden': '.', 'flag': 'F', 'mine': '*'}
        lines = ['PERCEPTION', f'Board {board.rows} x {board.cols} / {board.status}',
                 f'Total mines: {board.mines_total if board.mines_total is not None else "unknown"}',
                 f'Covered: {sum(c.state == "hidden" for c in board.cells)} / Flags: {sum(c.state == "flag" for c in board.cells)}', '',
                 *[' '.join(symbols.get(states[r, c], states[r, c]) for c in range(board.cols)) for r in range(board.rows)],
                 '', 'DECISION']
        if self.last_step and self.last_step.get('action'):
            action = self.last_step['action']
            decision = self.last_step['decision']
            lines += [f'Source: {decision["source"]} / candidates: {decision["candidates"]}',
                      f'{action["button"]} click: row {action["row"] + 1}, column {action["col"] + 1}',
                      action['reason'], 'Proved safe' if action['guaranteed_safe'] else 'Uncertain guess',
                      '', 'VERIFICATION', f'Observed result: {self.last_step["status"]}',
                      f'Move time: {self.last_step["timing_ms"]["total"]:.0f} ms']
        else:
            lines += ['Waiting for a verified move.']
        self.inspector.describe('\n'.join(lines))

    def refresh(self):
        self.window_list = windows()
        self.targets['values'] = [f'{t.title} [{t.hwnd}]' for t in self.window_list]
        if self.window_list:
            self.targets.current(0)

    def capture(self):
        self.ensure_idle()
        index = self.targets.current()
        if index < 0:
            raise ValueError('Select a game window first.')
        self.frame = self.grid = self.selected = None
        self.details = []
        self.canvas.delete('all')
        self.target = self.window_list[index]
        from desktop.target import user32
        user32.SetForegroundWindow(self.target.hwnd)
        time.sleep(.2)
        capture = Capture(self.target)
        try:
            self.frame = capture.read()
        finally:
            capture.close()
        self.grid = self.selected = None
        self.details = []
        self.draw()
        self.root.lift()
        self.status.set('Captured. Drag around the full grid, or use Suggest grid on a covered board.')

    def dimensions(self):
        return int(self.rows.get()), int(self.cols.get()), int(self.mines.get()) if self.mines.get().strip() else None

    def suggest(self):
        self.ensure_idle()
        if self.frame is None:
            raise ValueError('Capture a game window first.')
        grid = suggest_grid(self.frame.pixels, *self.dimensions())
        self.grid = Grid(grid.x + self.frame.origin[0], grid.y + self.frame.origin[1],
                         grid.width, grid.height, grid.rows, grid.cols, grid.mines)
        self.preview()

    def press(self, event):
        self.drag_start = event.x, event.y

    def release(self, event):
        if self.interface.get() == 'browser':
            return
        self.ensure_idle()
        if self.frame is None:
            return
        x0, y0 = self.drag_start
        if abs(event.x - x0) > 8 and abs(event.y - y0) > 8:
            x, y = round(min(x0, event.x) / self.scale), round(min(y0, event.y) / self.scale)
            w, h = round(abs(event.x - x0) / self.scale), round(abs(event.y - y0) / self.scale)
            self.grid = Grid(x + self.frame.origin[0], y + self.frame.origin[1], w, h, *self.dimensions())
            self.preview()
        elif self.grid:
            x, y = event.x / self.scale + self.frame.origin[0], event.y / self.scale + self.frame.origin[1]
            for row, col, box in self.grid.boxes():
                if box.x <= x < box.x + box.width and box.y <= y < box.y + box.height:
                    self.selected = row, col, box
                    self.status.set(f'Selected row {row + 1}, column {col + 1}. Choose its visible label and Teach cell.')
                    self.draw()
                    break

    def preview(self):
        self.ensure_idle()
        if self.frame is None or self.grid is None:
            raise ValueError('Capture and mark a grid first.')
        g = self.grid
        self.grid = Grid(g.x, g.y, g.width, g.height, *self.dimensions())
        self.board, self.details = recognize(self.frame.pixels, self.grid, self.profile, self.frame.origin)
        self.draw()
        self.describe_board()
        unknown = sum(d['label'] is None for d in self.details)
        self.status.set(f'Profile: {self.profile.name}. {unknown} unknown cells. '
                        + ('Inspect labels, focus game, then F8 to run or F9 for one step.' if self.board else 'Teach missing/uncertain labels before play.'))
        self.inspector.banner.configure(bg=STATUS['ready' if self.board else 'paused'])

    def draw(self):
        from PIL import Image, ImageTk
        self.canvas.delete('all')
        if self.interface.get() == 'browser':
            if self.browser_image is None:
                return
            image = self.browser_image
        else:
            image = Image.fromarray(self.frame.pixels[:, :, ::-1])
        self.scale = self.inspector.fit(image.width, image.height)
        image = image.resize((round(image.width * self.scale), round(image.height * self.scale)))
        self.photo = ImageTk.PhotoImage(image, master=self.canvas)
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        labels = {(d['row'], d['col']): d['label'] for d in self.details}
        if self.interface.get() == 'browser' and self.board:
            for cell in self.board.cells:
                box = cell.box
                self.canvas.create_rectangle(*[v * self.scale for v in (box.x, box.y, box.x + box.width, box.y + box.height)],
                                             outline='#31d589')
        elif self.grid:
            for row, col, box in self.grid.boxes():
                label = labels.get((row, col))
                color = '#31d589' if label is not None else '#ff5b62'
                if self.selected and (row, col) == self.selected[:2]:
                    color = '#f6df45'
                x, y = box.x - self.frame.origin[0], box.y - self.frame.origin[1]
                coords = [x, y, x + box.width, y + box.height]
                self.canvas.create_rectangle(*[v * self.scale for v in coords], outline=color)
                text = {'hidden': '.', 'flag': 'F', 'mine': '*'}.get(label, label or '?')
                self.canvas.create_text((x + box.width / 2) * self.scale, (y + box.height / 2) * self.scale, text=text, fill=color)

    def teach(self):
        self.ensure_idle()
        if not self.selected:
            raise ValueError('Click a cell in the preview first.')
        self.profile.add(self.label.get(), self.grid.crop(self.frame.pixels, self.selected[2], self.frame.origin))
        self.preview()

    def load_profile(self):
        from tkinter import filedialog
        self.ensure_idle()
        path = filedialog.askopenfilename(initialdir=str(PROFILES), filetypes=[('Appearance profile', '*.json')])
        if path:
            self.profile = Profile.load(path)
            if self.grid:
                self.preview()

    def save_profile(self):
        from tkinter import filedialog
        self.ensure_idle()
        path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('Appearance profile', '*.json')])
        if path:
            self.profile.save(path)
            self.status.set(f'Profile saved: {path}')

    def new_profile(self):
        self.safe(self.ensure_idle)
        if not self.worker or not self.worker.is_alive():
            self.profile = Profile()
            if self.grid:
                self.preview()

    def save_config(self):
        from tkinter import filedialog
        self.ensure_idle()
        if not self.grid or not self.target:
            raise ValueError('Calibrate first.')
        path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('Calibration', '*.json')])
        if path:
            profile_path = Path(path).with_suffix('.profile.json')
            self.profile.save(profile_path)
            Path(path).write_text(json.dumps(dict(target=asdict(self.target), grid=asdict(self.grid),
                geometry=asdict(self.frame.geometry), profile=str(profile_path.resolve())), indent=2), encoding='utf-8')
            self.status.set('Calibration saved. A changed target or window size requires recalibration.')

    def hotkey(self, action):
        if action == 'pause':
            self.gate.pause()
        self.events.put(('command', action))

    def start(self, single):
        if self.interface.get() == 'browser':
            if not self.browser or not self.browser.connected:
                raise ValueError('Connect to a supported browser game first.')
            if self.browser.busy:
                self.gate.pause()
                raise ValueError('Pending browser operation cancelled. Wait before resuming.')
            if self.board and not self.trust.get() and any(c.state == 'flag' for c in self.board.cells):
                raise ValueError('Existing flags require Trust existing flags before attaching mid-game.')
            self.gate.start(single)
            self.browser.play(self.gate.generation, single, self.policy.get(),
                              getattr(self.args, 'max_steps', 500), getattr(self.args, 'delay', .35))
            self.status.set('Running browser adapter. F10 or Pause stops input.')
            self.inspector.banner.configure(bg=STATUS['running'])
            return
        if self.worker and self.worker.is_alive():
            self.gate.pause()
            self.status.set('Pending operation cancelled. Wait for it to finish before resuming.')
            return
        if not self.target or not self.grid:
            raise ValueError('Capture and calibrate the board first.')
        self.gate.start(single)
        generation = self.gate.generation
        output = ROOT / '.cache/minesweeper/desktop' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        policy, trust = self.policy.get(), self.trust.get()
        geometry = self.frame.geometry
        def work():
            try:
                result = run_session(self.target, self.grid, self.profile, geometry, self.gate, generation,
                    self.args.server, policy, output,
                    notify=lambda frame, board, details: self.latest_preview(frame, details, board), trust_flags=trust,
                    max_steps=getattr(self.args, 'max_steps', 500),
                    on_step=lambda event: self.events.put(('step', event)))
            except Exception as error:
                self.gate.pause()
                result = dict(status='paused', reason=str(error))
            self.events.put(('result', result))
        self.status.set(f'Running with {policy}. F10 pauses immediately. Logs: {output}')
        self.inspector.banner.configure(bg=STATUS['running'])
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def latest_preview(self, frame, details, board=None):
        # One replaceable snapshot; inference never accumulates UI frames.
        self.pending_preview = frame, details, board

    def show_result(self, result):
        self.status.set(f"{result['status']}: {result.get('reason', 'Session finished.')} Steps: {result.get('steps', 0)}")
        self.inspector.banner.configure(bg=STATUS['ready' if result['status'] == 'won' else 'paused'])
        if result.get('steps', 0):
            self.trust.set(True)

    def poll(self):
        while not self.events.empty():
            kind, value = self.events.get_nowait()
            if kind == 'command':
                if value == 'pause':
                    self.status.set('Paused. Browser: Run or F8 to resume. Visual: focus the game, then F8.')
                    self.inspector.banner.configure(bg=STATUS['paused'])
                else:
                    self.safe(lambda: self.start(value == 'step'))
            elif kind == 'browser':
                browser, event, data = value
                if browser is not self.browser:
                    continue
                if event == 'preview':
                    from io import BytesIO
                    from PIL import Image
                    self.board, pixels = data
                    self.browser_image = Image.open(BytesIO(pixels)).convert('RGB')
                    self.draw()
                    self.describe_board()
                elif event == 'step':
                    self.last_step = data
                    if data.get('steps', 0):
                        self.trust.set(True)
                    self.describe_board()
                elif event == 'ready':
                    self.status.set(f'Browser connected. Run or Step to play. Logs: {data}')
                    self.inspector.banner.configure(bg=STATUS['ready'])
                elif event == 'result':
                    self.show_result(data)
                elif event == 'error':
                    self.status.set(f'Browser stopped: {data}. Screen pixels can read other websites after calibration.')
                    self.inspector.banner.configure(bg=STATUS['paused'])
            elif kind == 'step':
                self.last_step = value
                self.describe_board()
            else:
                self.show_result(value)
        preview = getattr(self, 'pending_preview', None)
        if preview:
            self.pending_preview = None
            self.frame, self.details, self.board = preview
            self.draw()
            self.describe_board()
        if not self.gate.closed:
            self.root.after(80, self.poll)

    def close(self):
        if self.browser:
            self.browser.close()
        self.gate.stop()
        self.hotkeys.close()
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='http://127.0.0.1:8000')
    parser.add_argument('--policy', choices=('systemone', 'logic'), default='systemone')
    parser.add_argument('--interface', choices=('visual', 'browser'), default='visual')
    parser.add_argument('--url', help='Supported browser game URL; Connect before playing')
    parser.add_argument('--difficulty', choices=('beginner', 'intermediate', 'expert'), default='beginner')
    parser.add_argument('--delay', type=float, default=.35)
    parser.add_argument('--config', type=Path, help='Run a saved calibration without the controller UI')
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--trust-flags', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--record-vram', action='store_true',
                        help='Diagnostic whole-GPU memory polling; can substantially slow live inference on Windows')
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error('--max-steps must be positive')
    if args.delay < .1:
        parser.error('--delay must be at least 0.1 seconds')
    if args.config and args.interface != 'visual':
        parser.error('--config is a screen-pixel calibration; use --interface visual')
    enable_dpi()
    if args.config:
        from desktop.target import Geometry
        config = json.loads(args.config.read_text(encoding='utf-8'))
        target, grid = Target(**config['target']), Grid(**config['grid'])
        geometry = Geometry(**config['geometry'])
        profile = Profile.load(config['profile'])
        gate = Gate()
        hotkeys = Hotkeys(lambda command: gate.pause() if command == 'pause' else None)
        output = args.output or ROOT / '.cache/minesweeper/desktop' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        print('Focus the calibrated game. Starting in 3 seconds; F10 pauses.', flush=True)
        gate.start()
        generation = gate.generation
        try:
            time.sleep(3)
            from contextlib import nullcontext
            from desktop.gpu_metrics import GpuRecorder
            recorder = GpuRecorder(output / 'vram.jsonl', 'play') if args.record_vram else nullcontext()
            with recorder:
                result = run_session(target, grid, profile, geometry, gate, generation, args.server, args.policy,
                                     output, max_steps=args.max_steps, trust_flags=args.trust_flags)
            print(json.dumps(result), flush=True)
            return 0 if result['status'] in ('won', 'lost') else 2
        finally:
            gate.stop()
            hotkeys.close()
    import tkinter as tk
    root = tk.Tk()
    Controller(root, args)
    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
