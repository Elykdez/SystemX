"""Controller and debug window: live annotated view, world/intent/System One/task panels, hotkeys.

The window sits beside the game (never over it) and runs the player in a worker
thread. "Act in 5 s" (or F8) lets the agent act in execute mode; "Pause" (or F10)
stops it; physical mouse or keyboard use also pauses. The agent only acts while the
match is being played. Only one controller may run at a time (global hotkeys).
Setup picks the game's window and client profile and finds its view (setup_ui.py); the
view shows the scene's tracked units and buildings (perception/scene.py), fading when unseen.
"""

import ctypes
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from desktop.input import Hotkeys
from desktop.inspector import Inspector, STATUS
from .perception.scene import draw_scene

TITLE = 'SystemX RA2 Agent'      # must not contain "Chrono Divide": the window finder uses that
ROOT = Path(__file__).resolve().parents[2]
ARM_DELAY = 5
MUTEX = 'Local\\SystemOneRA2AgentController'


def single_instance():
    """Named mutex held for the process lifetime; None if another controller is running."""
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, False, MUTEX)
    if not handle or ctypes.get_last_error() == 183:          # ERROR_ALREADY_EXISTS
        return None
    return handle


def ensure_server(url):
    """If no System One server answers, open start.bat in its own visible console window."""
    try:
        from systemone_client import SystemOneClient
        with SystemOneClient(url, timeout=2) as client:
            client.health()
        return 'running'
    except Exception:
        pass
    if not url.startswith(('http://127.0.0.1:8000', 'http://localhost:8000')):
        return 'unreachable'
    subprocess.Popen(['cmd.exe', '/c', str(ROOT / 'start.bat')], cwd=ROOT,
                     creationflags=subprocess.CREATE_NEW_CONSOLE)
    return 'starting'


STATE_COLORS = {'acting': STATUS['running'], 'ready': STATUS['ready'], 'waiting': STATUS['idle'],
                'paused': STATUS['paused'], 'watch': STATUS['running'], 'recommend': '#4c3e70',
                'starting': STATUS['idle'], 'stopped': STATUS['idle']}


class Controller:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.player = self.thread = self.supervisor = None
        self.photo = None
        self.profile = self.setup = self.target = None      # from the setup window; else the player's defaults
        root.title(TITLE)
        width = min(1100, max(700, root.winfo_screenwidth() - 1100))
        height = min(940, root.winfo_screenheight() - 80)
        at = getattr(args, 'at', None)       # e.g. on a second monitor, clear of the game window
        x, y = at if at else (root.winfo_screenwidth() - width - 12, 20)
        root.geometry(f'{width}x{height}+{x}+{y}')
        root.minsize(640, 640)
        self.inspector = Inspector(root, 'OpenCV perception, persistent intent and verified game actions', 'RA2 / LIVE')
        top = ttk.Frame(self.inspector.controls)
        top.pack(fill='x')
        self.mode = tk.StringVar(value='execute')
        self.policy = tk.StringVar(value=args.policy)
        for value in ('watch', 'recommend', 'execute'):
            ttk.Radiobutton(top, text=value.capitalize(), value=value, variable=self.mode).pack(side='left')
        ttk.Separator(top, orient='vertical').pack(side='left', fill='y', padx=6)
        for value in ('systemone', 'rules'):
            ttk.Radiobutton(top, text='System One' if value == 'systemone' else 'Rules only', value=value,
                            variable=self.policy).pack(side='left')
        self.start_button = ttk.Button(top, text='Start', command=self.start)
        self.start_button.pack(side='left', padx=(12, 2))
        ttk.Button(top, text='Stop', command=self.stop).pack(side='left')
        ttk.Button(top, text='Setup...', command=self.open_setup).pack(side='left', padx=(12, 0))
        self.game = tk.StringVar(value=self._game_label())
        ttk.Label(self.inspector.controls, textvariable=self.game, style='Muted.TLabel', wraplength=width - 40).pack(fill='x', pady=4)
        controls = ttk.Frame(self.inspector.controls, padding=(0, 0, 0, 4))
        controls.pack(fill='x')
        self.act_button = ttk.Button(controls, text='Act', command=self.act)
        self.act_button.pack(side='left')
        ttk.Button(controls, text='Pause agent', command=lambda: self.hotkey('pause')).pack(side='left', padx=4)
        self.hotkey_status = tk.StringVar(value='')
        ttk.Label(controls, textvariable=self.hotkey_status, style='Muted.TLabel').pack(side='left', padx=8)
        self.countdown = None
        self.state, self.banner = self.inspector.status, self.inspector.banner
        self.state.set('Choose a mode and press Start.')
        self.window_hint = tk.StringVar(value='')
        self.window_warning = tk.Label(root, textvariable=self.window_hint, anchor='w', justify='left',
                                       fg='#ffe0a6', bg=STATUS['warning'], font=('Segoe UI', 10), padx=8, pady=6,
                                       wraplength=width - 20)
        ttk.Label(self.inspector.controls, style='Muted.TLabel', font=('Segoe UI', 9),
                 text='Act in the menu = act the moment the match starts; Act in a match = act after 5 s. '
                      'F10 stops only the agent; the match continues. Esc in game opens its menu.',
                 wraplength=width - 20, justify='left').pack(fill='x')
        self.canvas, self.text = self.inspector.canvas, self.inspector.text
        self.last_view = None
        self.canvas.bind('<Configure>', lambda e: self.show(*self.last_view) if self.last_view else None)
        self.inspector.describe('PERCEPTION\nConnect to a game to see tracked units, buildings and resources.\n\n'
                                'DECISION\nSystem One selects from the available choices.\n\n'
                                'CONTROL\nAct arms the agent. F10 pauses input; the match continues.')
        try:
            self.hotkeys = Hotkeys(self.hotkey)
            self.hotkey_status.set('Hotkeys: F8 act, F10 pause agent')
        except RuntimeError as error:
            self.hotkeys = None
            self.hotkey_status.set(f'Hotkeys unavailable ({error}); use the buttons.')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(200, self.refresh)
        self.exit_code = 0
        self.launched = self.last_autostart = time.monotonic()
        self.ended_seen = None
        if getattr(args, 'autostart', False):
            self.mode.set(getattr(args, 'mode', None) or 'execute')
            root.after(300, self.autostart)
        if getattr(args, 'autoplay', False):
            root.after(1000, self.autoplay_watch)

    def _game_label(self):
        from .calibration import load_setup
        setup = self.setup or load_setup()
        name = self.profile.name if self.profile is not None else (setup.profile if setup else 'chronodivide')
        window = (setup.window or {}).get('title', '') if setup else ''
        size = 'x'.join(map(str, setup.size)) if setup and setup.size else 'found at start'
        return f'Game: {name}   window: {window or "found by title at start"}   view: {size}   (Setup... to change)'

    def open_setup(self):
        if self.player is not None:
            self.state.set('Stop the player before changing the setup.')
            return
        from .setup_ui import SetupWindow

        def done(profile, setup, target):
            self.profile, self.setup, self.target = profile, setup, target
            self.game.set(self._game_label())
        SetupWindow(self.root, self.profile, self.setup, on_done=done)

    # -- actions -------------------------------------------------------------
    def autostart(self):
        """Start without clicks; unattended runs also arm themselves and bring the game forward.
        Autoplay also restores a minimized game window, or opens the game in Chrome."""
        if getattr(self.args, 'autoplay', False):
            from .window import ensure_game
            try:
                ensure_game()
            except Exception as error:                     # retried by autoplay_watch
                self.state.set(f'Waiting for the game: {error}')
                return
        self.start()
        if self.player is None or self.player.mode != 'execute' or not getattr(self.args, 'unattended', False):
            return
        from .unattended import Supervisor
        self.supervisor = Supervisor(self.player, resume_after=getattr(self.args, 'resume_after', 8.0),
                                     log=self.player.events,
                                     front=self.front if getattr(self.args, 'front', False) else None)
        self.supervisor.release()
        if getattr(self.args, 'front', False):
            self.root.after(700, self.front)

    def autoplay_watch(self):
        """Autoplay: exit a few seconds after the match ends (the launcher starts the next one
        with fresh code); give up if no match starts within 8 minutes."""
        player, now = self.player, time.monotonic()
        if player is None:
            # The game window was missing (browser closed): open it again and retry.
            if now - self.launched > 480:
                self.exit_code = 3
                self.close()
                return
            if now - self.last_autostart >= 15.0:
                self.last_autostart = now
                self.autostart()
        if player is not None:
            if player.match.get('ended') is not None:
                self.ended_seen = self.ended_seen or now
                if now - self.ended_seen >= 3.0:
                    self.exit_code = 0
                    self.close()
                    return
            elif player.match.get('started') is None and now - self.launched > 480:
                self.exit_code = 3
                self.close()
                return
        self.root.after(1000, self.autoplay_watch)

    def front(self):
        from .window import bring_to_front
        restored = self.player is not None and bring_to_front(self.player.target)
        if not restored:
            self.hotkey_status.set('Could not bring the game window to the front; click its title bar.')
        return restored

    def act(self):
        """In a running match: arm after a countdown. Otherwise: arm when the match starts."""
        if self.player is None:
            self.state.set('Press Start first.')
            return
        if self.supervisor is not None:
            self.supervisor.release()
        elif self.player.last_playing[0]:
            self.arm_later()
        else:
            self.player.arm_on_start()

    def hotkey(self, command):
        player = self.player
        if player is None:
            return
        if command == 'run':
            if self.supervisor is not None:
                self.supervisor.release()
            else:
                player.arm_on_start()
        elif command == 'pause':
            self.cancel_countdown()
            if self.supervisor is not None:
                self.supervisor.hold('paused by the user')
            else:
                player.pause('paused by the user')

    def arm_later(self, remaining=ARM_DELAY):
        """Countdown so the user can click into the game before the agent starts acting."""
        if self.player is None:
            self.state.set('Press Start first.')
            return
        if remaining <= 0:
            self.countdown = None
            self.act_button.configure(text='Act')
            self.player.arm()
            return
        self.act_button.configure(text=f'Acting in {remaining} s: click into the game now')
        self.countdown = self.root.after(1000, lambda: self.arm_later(remaining - 1))

    def cancel_countdown(self):
        if self.countdown is not None:
            self.root.after_cancel(self.countdown)
            self.countdown = None
            self.act_button.configure(text='Act')

    def start(self):
        if self.player is not None:
            return
        if self.policy.get() == 'systemone' and self.mode.get() != 'watch':
            if ensure_server(self.args.server) == 'starting':
                self.hotkey_status.set('System One server is starting in its own window; '
                                       'the agent uses its rules until then.')
        try:
            from .player import Player
            self.player = Player(mode=self.mode.get(), server=self.args.server, policy=self.policy.get(),
                                 record_fps=self.args.record_fps or (1.0 if self.mode.get() == 'execute' else 0.0),
                                 autoplay=getattr(self.args, 'autoplay', False), profile=self.profile,
                                 setup=self.setup, target=self.target, strategy=getattr(self.args, 'strategy', None))
        except Exception as error:
            self.player = None
            self.state.set(f'Cannot start: {error}')
            self.banner.configure(bg=STATE_COLORS['paused'])
            return
        self.thread = threading.Thread(target=self.player.run, name='player', daemon=True)
        self.thread.start()
        self.start_button.state(['disabled'])

    def stop(self):
        player, self.player = self.player, None
        self.supervisor = None
        if player is not None:
            player.stop()
            if self.thread:
                self.thread.join(5)
            self.state.set(f'Stopped. Run folder: {player.output}')
            self.banner.configure(bg=STATE_COLORS['stopped'])
        self.start_button.state(['!disabled'])

    def close(self):
        self.stop()
        if self.hotkeys:
            self.hotkeys.close()
        self.root.destroy()

    # -- view ----------------------------------------------------------------
    def refresh(self):
        try:
            player = self.player
            if player is not None:
                self.supervise()
                state = player.view_state()
                status = state['status']
                label = f"{status['state'].upper()}  {status['reason']}".strip()
                if state.get('plan'):
                    label += f"   next: {state['plan']}"
                self.state.set(label)
                self.banner.configure(bg=STATE_COLORS.get(status['state'], '#6b7280'))
                self.update_window_hint(state.get('hud'))
                if player.preview is not None:
                    self.show(player.preview, player.units, player.items)
                description = self.describe(state)
                if self.supervisor is not None:
                    counts = self.supervisor.counts
                    description = (f"UNATTENDED  held {self.supervisor.held}  resumes {counts['resumes']}"
                                   f"  pointer recaptures {counts['recaptures']}"
                                   f"  menu resumes {counts['menu_resumes']}\n" + description)
                self.inspector.describe(description)
                self.inspector.badge.set(f"RA2 / {state['mode'].upper()}")
            else:
                self.window_warning.pack_forget()
                self.window_hint.set('')
        finally:
            self.root.after(200, self.refresh)

    def update_window_hint(self, hud=None):
        error = getattr(getattr(self.player, 'view', None), 'error', '')
        hint = f'Cannot proceed: {error}' if error else ''
        if not hint and hud and hud['mode'] == 'other' and hud.get('scores'):
            scores = ', '.join(f'{name} {score:.0%}' for name, score in hud['scores'].items())
            hint = (f"Cannot proceed: match HUD not recognized ({scores}; required {hud['required']:.0%}). "
                    'Start or resume a match. If a match is already visible, teach its HUD in Setup. '
                    'Game input is stopped.')
        self.window_hint.set(hint)
        if hint:
            self.window_warning.pack(fill='x', padx=16, pady=(0, 8), after=self.banner)
        else:
            self.window_warning.pack_forget()

    def supervise(self):
        if self.supervisor is None:
            return
        try:
            self.supervisor.step()
        except Exception as error:        # a failed grab (window covered, moved) waits for the next step
            self.hotkey_status.set(f'Unattended: {error}'[:120])

    def show(self, image, tracks=(), items=None):
        """The annotated view, with the scene's tracks drawn at preview size (still legible)."""
        self.last_view = image, tracks, items
        scale = self.inspector.fit(image.shape[1], image.shape[0])
        small = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        if tracks:
            draw_scene(small, tracks, items, scale=scale)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), master=self.canvas)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    @staticmethod
    def describe(state):
        world, intent = state['world'], state['intent']
        lines = [f"mode {world['mode']}   playing {state['playing'][0]} ({state['playing'][1]})",
                 f"credits {world['credits']}  reserved {world['reserved']}  power {world['power']}",
                 f"tab {world['tab']}   structures {world['structures']}",
                 f"units queued {world['units']}", '', 'LANES']
        for lane, view in world['lanes'].items():
            progress = f" {view['progress']:.0%}" if view['progress'] is not None else ''
            age = f"{view['age']:.1f}s" if view['age'] is not None else 'never seen'
            lines.append(f"  {lane:<10} {view['status']:<9} {view['item'] or '':<16}{progress:<6} age {age}"
                         f"   avail {','.join(view['available'])[:60]}")
        if state.get('army'):
            army = state['army']
            lines += ['', f"ARMY posture {army['posture']}  combat units built {army['size']}  "
                          f"at home {army['available']}  attack size {army.get('attack_size')}  "
                          f"waves {army['waves']}  camera {army['camera']}",
                      f"  attack: {army['attack']}", f"  threat: {army['threat']}"]
        units = state.get('units')
        if units and (units['ours'] or units.get('patches') or units.get('enemy_patches') or any(units['enemy'].values())):
            ours = ', '.join(f'{count} {name}' for name, count in sorted(units['ours'].items())) or 'none recognized'
            lines += [f"SCENE ours: {ours}  health {units['health']}",
                      f"  enemy recognized: {units['enemy']['unit']} units, {units['enemy']['building']} buildings",
                      f"  unconfirmed color patches: {units.get('patches', 0)} ours, {units.get('enemy_patches', 0)} enemy",
                      f"  boxes: units green, buildings teal, enemy red/orange; unlabeled = unconfirmed",
                      f"  names with ? are guesses; % is health; seconds mark unseen tracks",
                      f"  tracked {units.get('tracked')}  scan {units.get('ms')} ms"
                      + (f"  ERROR {units['error']}" if units.get('error') else '')]
        if units and units.get('detector'):
            detector = units['detector']
            lines += [f"  detector: {detector['where'] or 'starting'}  {detector['ms']} ms"
                      + (f"  ({detector['error'][:80]})" if detector.get('error') else '')]
        if units and units.get('pairs') is not None:
            lines += [f"  census pairs saved for the detector: {units['pairs']}"]
        intel = state.get('intel')
        if intel:
            looks = ', '.join(f"{k} {v['age']:.0f}s ago ({v['enemy_objects']} enemy, {v['own_objects']} ours)"
                              for k, v in intel['looks'].items()) or 'none yet'
            lines += [f"INTEL enemy base {intel['enemy_base'] or 'not found'}  strength {intel['strength']}  "
                      f"scout {'out' if intel['scout'] and not intel['scout'].get('found') else intel['scout'] and 'found it'}",
                      f"  looks: {looks}"]
        radar = world.get('radar')
        if radar:
            lines += [f"RADAR online {radar['online']}  explored {radar.get('explored')}  "
                      f"enemy dots {len(radar.get('enemy') or [])}"]
        lines += ['', f"INTENT focus {intent.get('focus')}  objective {intent['objective']}  posture {intent.get('posture')}  "
                      f"attack size {intent.get('attack_size') or 'rule'}  intel request {intent.get('intel')}",
                  f"  production {intent['production']}", f"  provenance {intent['provenance']}"]
        if state['tasks']:
            lines += ['', f"TASKS active {json.dumps(state['tasks']['active'])}",
                      f"  recent {json.dumps(state['tasks']['recent'][-4:])}"]
        if state['systemone']:
            lines += ['', f"SYSTEM ONE {json.dumps(state['systemone'])[:300]}"]
        lines += ['', f"APM {state['apm']}  counts {state['counts']}  pointer {state['pointer']}",
                  f"tick ms {state['timings_ms']}", '', 'EVENTS']
        lines += [f"  {json.dumps(event)[:110]}" for event in state['events'][-8:]]
        return '\n'.join(lines)


def main(args):
    from desktop.target import enable_dpi
    enable_dpi()
    lock = single_instance()
    if lock is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(TITLE, 'The SystemX RA2 Agent is already running. Use the open window '
                                    '(check the taskbar), or close it before starting another.')
        root.destroy()
        return
    root = tk.Tk()
    controller = Controller(root, args)
    root.mainloop()
    sys.exit(controller.exit_code)
