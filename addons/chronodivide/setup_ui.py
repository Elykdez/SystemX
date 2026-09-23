"""Setup window: pick the game's window and client profile, find or mark the game view, check what
is recognized, and teach what is not, the way the Minesweeper desktop player is calibrated.

1. Choose the profile (Chrono Divide, or a starter for the original game) and the game's window.
2. Capture: the window comes to the front for a moment and is captured once. Start a match first:
   Find game looks for the sidebar's HUD anchors; drag around the game view instead when they
   are not found.
3. Check shows what is read: the sidebar's cameos, credits, power, the play/menu state, and the
   scene's units and buildings. Unknown cameos are red.
4. Teach what is wrong or unknown: click a cameo and choose its item (in its normal, available
   look), enter the credits shown, take the HUD look from this frame, or pick our and the
   enemy's house colors by clicking a unit or building of that side. Save the profile (taught
   templates go to a new profile; the bundled ones are never overwritten).
5. Use this setup: the controller's next Start plays this window with this profile.

The setup sends no input to the game; it only captures its window.
"""

import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from desktop.capture import Capture
from desktop.target import DesktopPause
from .calibration import Setup, find_game, load_setup, rank_windows, save_setup
from .client import ClientProfile
from .perception import colors as house
from .perception.colors import HouseColor
from .perception.modes import ModeReader
from .perception.reader import Perception
from .perception.scene import Tracker, confirmed, draw_scene, scan
from .perception.sidebar import annotate
from .window import Rect, bring_to_front
from .world import World, load_knowledge

PREVIEW = (900, 520)
BUNDLED = ('chronodivide', 'ra2-original')


class SetupWindow:
    def __init__(self, master, profile=None, setup=None, on_done=None, capture=None):
        self.top = tk.Toplevel(master)
        self.top.title('SystemX RA2 Agent: setup')
        self.on_done = on_done
        self.grab = capture or self._capture_window
        self.setup = setup or load_setup() or Setup()
        self.profile = profile or ClientProfile.load(self.setup.profile)
        self.items = load_knowledge()['items']
        self.frame = self.target = self.rect = None
        self.candidates = []
        self.readers = None           # (catalog, credits, modes) of the profile, with what was taught
        self.readings, self.selected, self.picking = [], None, None
        self.taught = set()
        self.scale, self.drag = 1.0, None
        self.status = tk.StringVar(value='Choose the profile and the game window, start a match, then Capture.')
        self._build()
        self._load_readers()
        self.refresh_windows()

    # -- widgets --------------------------------------------------------------
    def _build(self):
        top = self.top
        row = ttk.Frame(top, padding=6)
        row.pack(fill='x')
        ttk.Label(row, text='Profile').pack(side='left')
        self.profile_name = tk.StringVar(value=self.profile.slug)
        self.profiles = ttk.Combobox(row, textvariable=self.profile_name, values=ClientProfile.available(),
                                     width=22, state='readonly')
        self.profiles.pack(side='left', padx=4)
        self.profiles.bind('<<ComboboxSelected>>', lambda e: self.safe(self.change_profile))
        ttk.Button(row, text='Save profile', command=lambda: self.safe(self.save_profile)).pack(side='left')
        self.about = tk.StringVar()
        ttk.Label(row, textvariable=self.about, foreground='#555').pack(side='left', padx=8)
        row = ttk.Frame(top, padding=(6, 0))
        row.pack(fill='x')
        ttk.Label(row, text='Window').pack(side='left')
        self.window_text = tk.StringVar()
        self.windows = ttk.Combobox(row, textvariable=self.window_text, width=70, state='readonly')
        self.windows.pack(side='left', padx=4, fill='x', expand=True)
        ttk.Button(row, text='Refresh', command=lambda: self.safe(self.refresh_windows)).pack(side='left')
        ttk.Button(row, text='Capture', command=lambda: self.safe(self.capture)).pack(side='left', padx=4)
        self.canvas = tk.Canvas(top, width=PREVIEW[0], height=PREVIEW[1], bg='#202428', highlightthickness=0)
        self.canvas.pack(padx=6, pady=4)
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.motion)
        self.canvas.bind('<ButtonRelease-1>', lambda e: self.safe(lambda: self.release(e)))
        row = ttk.Frame(top, padding=(6, 0))
        row.pack(fill='x')
        ttk.Button(row, text='Find game', command=lambda: self.safe(self.find)).pack(side='left')
        ttk.Button(row, text='Check', command=lambda: self.safe(self.check)).pack(side='left', padx=4)
        ttk.Label(row, text='or drag around the game view in the preview', foreground='#555').pack(side='left', padx=6)
        row = ttk.Frame(top, padding=(6, 4))
        row.pack(fill='x')
        ttk.Label(row, text='Cameo').pack(side='left')
        self.item = tk.StringVar(value='(empty slot)')
        ttk.Combobox(row, textvariable=self.item, width=20, state='readonly',
                     values=['(empty slot)', *sorted(self.items)]).pack(side='left', padx=2)
        ttk.Button(row, text='Teach cameo', command=lambda: self.safe(self.teach_cameo)).pack(side='left')
        ttk.Label(row, text='   Credits shown').pack(side='left')
        self.credits = tk.StringVar()
        ttk.Entry(row, textvariable=self.credits, width=8).pack(side='left', padx=2)
        ttk.Button(row, text='Teach credits', command=lambda: self.safe(self.teach_credits)).pack(side='left')
        ttk.Button(row, text='Teach HUD look', command=lambda: self.safe(self.teach_hud)).pack(side='left', padx=6)
        row = ttk.Frame(top, padding=(6, 0))
        row.pack(fill='x')
        self.colors = tk.StringVar()
        ttk.Label(row, textvariable=self.colors).pack(side='left')
        row = ttk.Frame(top, padding=(6, 0))
        row.pack(fill='x')
        ttk.Button(row, text='Pick our color', command=lambda: self.pick('own')).pack(side='left', padx=4)
        ttk.Button(row, text='Add enemy color', command=lambda: self.pick('enemy')).pack(side='left')
        ttk.Button(row, text='Add ally color', command=lambda: self.pick('ally')).pack(side='left', padx=4)
        ttk.Button(row, text='Clear other sides', command=lambda: self.safe(self.clear_sides)).pack(side='left')
        ttk.Button(row, text='Red vs blue', command=lambda: self.safe(self.default_colors)).pack(side='left', padx=4)
        row = ttk.Frame(top, padding=(6, 4))
        row.pack(fill='x')
        self.build_off_ally = tk.BooleanVar()
        ttk.Checkbutton(row, text='Build off Ally enabled in match', variable=self.build_off_ally,
                        command=self.team_settings).pack(side='left')
        ttk.Button(row, text='Use this setup', command=lambda: self.safe(self.use)).pack(side='right')
        ttk.Label(top, textvariable=self.status, wraplength=PREVIEW[0], padding=6, justify='left').pack(fill='x')
        self._show_profile()

    def safe(self, function):
        try:
            function()
        except (DesktopPause, ValueError, OSError) as error:
            self.status.set(str(error))

    def _show_profile(self):
        p = self.profile
        self.about.set(f"{'input: pointer lock' if p.input == 'locked' else 'input: absolute'}; "
                       f"{'drawn in a page' if p.embedded else 'fills its window'}; "
                       f"{'verified: ' + p.verified if p.verified else 'unverified'}")
        own = p.own.name if p.own.hue is None else f'{p.own.name} (hue {p.own.hue:.0f})'
        enemies = ', '.join(c.name if c.hue is None else f'{c.name} (hue {c.hue:.0f})' for c in p.enemies)
        allies = ', '.join(c.name for c in p.allies)
        self.colors.set(f'Our color: {own}    Enemy colors: {enemies}    Allies: {allies or "none"}')
        self.build_off_ally.set(p.raw.get('team_play', {}).get('build_off_ally', False))

    def _load_readers(self):
        self.readers = (self.profile.cameos(), self.profile.credits(), None)
        self.taught = set()

    # -- window and view --------------------------------------------------------
    def refresh_windows(self):
        self.candidates = rank_windows(self.profile)
        self.windows['values'] = [c.label for c in self.candidates]
        if self.candidates:
            saved = (self.setup.window or {}).get('hwnd')
            index = next((i for i, c in enumerate(self.candidates) if c.target.hwnd == saved), 0)
            self.windows.current(index)

    def change_profile(self):
        self.profile = ClientProfile.load(self.profile_name.get())
        self._load_readers()
        self._show_profile()
        self.refresh_windows()
        if self.frame is not None:
            self.find(quiet=True)

    def _capture_window(self, target):
        bring_to_front(target)
        time.sleep(0.3)
        capture = Capture(target)
        try:
            return capture.read(foreground=False)
        finally:
            capture.close()
            self.top.lift()

    def capture(self):
        index = self.windows.current()
        if index < 0:
            raise ValueError('Choose the game window first (Refresh if it is not listed).')
        self.target = self.candidates[index].target
        self.frame = self.grab(self.target)
        self.rect, self.readings, self.selected = None, [], None
        self.find(quiet=True)

    def find(self, quiet=False):
        if self.frame is None:
            raise ValueError('Capture the game window first.')
        found = find_game(self.frame.pixels, self.profile)
        if found is None:
            self.draw()
            if not quiet:
                raise ValueError('The HUD anchors are not on screen. Start a match (the sidebar must show) and '
                                 'Capture again, or drag around the game view.')
            self.status.set('Captured. No match sidebar found: start a match and Capture again, or drag around '
                            'the game view.')
            return
        self.rect = found.rect
        self.check()
        self.status.set(f'Game view {found.size[0]}x{found.size[1]} at ({found.rect.x}, {found.rect.y}); '
                        f'anchors {found.anchors}. ' + self.status.get())

    def press(self, event):
        self.drag = (event.x, event.y)

    def motion(self, event):
        if self.drag is not None and self.frame is not None and self.picking is None:
            self.canvas.delete('drag')
            self.canvas.create_rectangle(*self.drag, event.x, event.y, outline='#f6df45', tags='drag')

    def release(self, event):
        start, self.drag = self.drag, None
        self.canvas.delete('drag')
        if self.frame is None or start is None:
            return
        x, y = round(event.x / self.scale), round(event.y / self.scale)
        if self.picking is not None:
            self.take_color(x, y)
        elif abs(event.x - start[0]) > 8 and abs(event.y - start[1]) > 8:
            x0, y0 = round(min(start[0], event.x) / self.scale), round(min(start[1], event.y) / self.scale)
            width, height = round(abs(event.x - start[0]) / self.scale), round(abs(event.y - start[1]) / self.scale)
            if width < 320 or height < 240:
                raise ValueError('That is too small for a game view; drag around all of it, sidebar included.')
            self.rect = Rect(x0, y0, width, height)
            self.check()
        elif self.rect is not None:
            self.select_slot(x - self.rect.x, y - self.rect.y)

    def view(self):
        if self.rect is None:
            raise ValueError('Find the game view first (Find game, or drag around it).')
        r = self.rect
        return self.frame.pixels[r.y:r.bottom, r.x:r.right]

    # -- reading and teaching ----------------------------------------------------
    def perception(self):
        catalog, credits, modes = self.readers
        layout = self.profile.layout(self.rect.width, self.rect.height)
        return Perception(layout, catalog, credits, modes or self.profile.modes(layout), profile=self.profile)

    def check(self):
        canvas = self.view()
        previous = self.profile.use_colors()
        try:
            perception = self.perception()
            world = World(clock=lambda: 0.0)
            mode, self.readings, lane = perception.read(canvas, world, 0.0)
            tactical = perception.layout.tactical.tuple()
            tracks = Tracker(clock=lambda: 0.0).update(scan(canvas, tactical), 0.0, region=tactical)
        finally:
            house.use(*previous)
        self.layout, self.tracks = perception.layout, tracks
        snapshot = world.snapshot()
        known = sum(1 for r in self.readings if r.item)
        unknown = sum(1 for r in self.readings if r.state == 'unknown')
        units = sum(1 for t in tracks if confirmed(t) and t.kind != 'building')
        buildings = sum(1 for t in tracks if confirmed(t) and t.kind == 'building')
        patches = sum(1 for t in tracks if not confirmed(t))
        self.status.set(f"{mode['mode'].upper()} (anchors {mode['scores']}).  Credits {snapshot['credits']}, "
                        f"power {(snapshot['power'] or {}).get('level')}, tab {lane}: {known} cameos known, "
                        f"{unknown} unknown.  Scene: {units} recognized units, {buildings} buildings, "
                        f"{patches} unconfirmed color patches.  "
                        + ('Teach what is unknown or wrong, then Use this setup.' if mode['mode'] == 'hud' else
                           'Not in a match: start one, then Capture again.'))
        self.draw()

    def select_slot(self, x, y):
        layout = self.profile.layout(self.rect.width, self.rect.height)
        for index, box in enumerate(layout.grid.boxes()):
            if box.contains((x, y)):
                self.selected = index
                reading = self.readings[index] if index < len(self.readings) else None
                self.status.set(f'Cameo slot {index + 1} selected ({reading.item or reading.state if reading else "?"}). '
                                'Choose its item and Teach cameo (it must show its normal, available look).')
                self.draw()
                return

    def teach_cameo(self):
        if self.selected is None:
            raise ValueError('Click a cameo slot in the preview first.')
        layout = self.profile.layout(self.rect.width, self.rect.height)
        crop = layout.grid.box(self.selected).crop(self.view())
        item = None if self.item.get() == '(empty slot)' else self.item.get()
        self.readers[0].teach(item, crop)
        self.taught.add('cameos')
        self.check()

    def teach_credits(self):
        text = self.credits.get().strip()
        if not text.isdigit():
            raise ValueError('Enter the credits exactly as the game shows them.')
        layout = self.profile.layout(self.rect.width, self.rect.height)
        if not self.readers[1].learn(layout.regions['credits'].crop(self.view()), int(text)):
            raise ValueError('The readout does not split into that many digits here: check the value and the view.')
        self.taught.add('digits')
        self.check()

    def teach_hud(self):
        layout = self.profile.layout(self.rect.width, self.rect.height)
        modes = ModeReader.from_frame(self.view(), anchors={k: b.tuple() for k, b in layout.anchors.items()},
                                      menu=layout.menu.tuple())
        self.readers = (self.readers[0], self.readers[1], modes)
        self.taught.add('hud')
        self.check()

    def pick(self, side):
        if self.rect is None:
            self.status.set('Find the game view first.')
            return
        self.picking = side
        label = dict(own='friendly owned', enemy='hostile', ally='teammate')[side]
        self.status.set(f"Click on a {label} unit or building in the preview.")

    def take_color(self, x, y):
        side, self.picking = self.picking, None
        patch = self.frame.pixels[max(0, y - 7):y + 8, max(0, x - 7):x + 8]
        name = simpledialog.askstring('House color', 'Name this color (for example green):', parent=self.top) or \
            f'{side}-color'
        color = HouseColor.sample(name, patch)
        enemies = [c for c in self.profile.enemies if c.name != name]
        allies = [c for c in self.profile.allies if c.name != name]
        if side == 'own':
            self.profile.set_colors(own=color, enemies=enemies, allies=allies)
        elif side == 'ally':
            self.profile.set_colors(enemies=enemies, allies=allies + [color])
        else:
            self.profile.set_colors(enemies=enemies + [color], allies=allies)
        self.taught.add('colors')
        self._show_profile()
        self.check()

    def default_colors(self):
        self.profile.set_colors(HouseColor('red'), [HouseColor('blue')], [])
        self.taught.add('colors')
        self._show_profile()
        if self.rect is not None:
            self.check()

    def clear_sides(self):
        self.profile.set_colors(enemies=[], allies=[])
        self.taught.add('colors')
        self._show_profile()

    def team_settings(self):
        self.profile.raw['team_play'] = {'build_off_ally': self.build_off_ally.get()}
        self.taught.add('team_play')

    def save_profile(self):
        if not self.taught:
            raise ValueError('Nothing was taught; the profile is unchanged.')
        profile = self.profile
        if profile.slug in BUNDLED:
            name = simpledialog.askstring('New profile', 'Name for the taught profile:', parent=self.top)
            if not name:
                return
            profile = profile.derived(name)
            profile.set_colors(self.profile.own, self.profile.enemies, self.profile.allies)
        catalog, credits, modes = self.readers
        path = profile.save(catalog if 'cameos' in self.taught else None, credits if 'digits' in self.taught else None,
                            modes if 'hud' in self.taught else None)
        self.profile, self.taught = profile, set()
        self.profiles['values'] = ClientProfile.available()
        self.profile_name.set(profile.slug)
        self._show_profile()
        self.status.set(f'Profile saved: {path}')

    def use(self):
        if self.target is None or self.rect is None:
            raise ValueError('Capture the game and find its view first.')
        if self.taught and not messagebox.askyesno('Unsaved teaching', 'What you taught is not saved in a profile '
                                                   'and will be lost. Use this setup anyway?', parent=self.top):
            return
        candidate = next((c for c in self.candidates if c.target.hwnd == self.target.hwnd), None)
        self.setup.profile = self.profile.slug if self.profile.slug in ClientProfile.available() else str(self.profile.path)
        self.setup.remember(self.target, candidate.process if candidate else '', self.rect, self.frame.geometry)
        save_setup(self.setup)
        if self.on_done is not None:
            self.on_done(self.profile, self.setup, self.target)
        self.top.destroy()

    # -- preview --------------------------------------------------------------------
    def draw(self):
        if self.frame is None:
            return
        image = self.frame.pixels.copy()
        if self.rect is not None:
            r = self.rect
            view = image[r.y:r.bottom, r.x:r.right]
            layout = self.profile.layout(r.width, r.height)
            if self.readings:
                annotate(view, layout, self.readings)
            for name in ('credits', 'power', 'radar', 'tactical'):
                box = layout.regions[name]
                cv2.rectangle(view, (box.x, box.y), (box.x + box.w - 1, box.y + box.h - 1), (255, 180, 0), 1)
            for box in layout.anchors.values():
                cv2.rectangle(view, (box.x, box.y), (box.x + box.w - 1, box.y + box.h - 1), (255, 0, 255), 1)
            if getattr(self, 'tracks', None):
                draw_scene(view, self.tracks, self.items, now=0.0)
            if self.selected is not None:
                box = layout.grid.box(self.selected)
                cv2.rectangle(view, (box.x - 2, box.y - 2), (box.x + box.w + 1, box.y + box.h + 1), (69, 223, 246), 2)
            cv2.rectangle(image, (r.x, r.y), (r.right - 1, r.bottom - 1), (69, 223, 246), 2)
        self.scale = min(PREVIEW[0] / image.shape[1], PREVIEW[1] / image.shape[0], 1.0)
        small = cv2.resize(image, (round(image.shape[1] * self.scale), round(image.shape[0] * self.scale)),
                           interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), master=self.canvas)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
