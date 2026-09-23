"""Shared Tk presentation for the game inspectors; no capture or input code."""

import ctypes
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk


BG = '#0f1722'
PANEL = '#192532'
TEXT = '#e4edf3'
MUTED = '#a2b3bf'
ACCENT = '#69bafa'
STATUS = {'idle': '#25384c', 'running': '#15506c', 'paused': '#873b39',
          'ready': '#275c48', 'warning': '#674e25'}


def theme(root):
    root.configure(background=BG)
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', background=BG, foreground=TEXT, font=('Segoe UI', 10))
    style.configure('TFrame', background=BG)
    style.configure('TLabel', background=BG, foreground=TEXT)
    style.configure('Muted.TLabel', foreground=MUTED)
    style.configure('TButton', background='#263c50', foreground=TEXT, padding=(10, 6), borderwidth=0)
    style.map('TButton', background=[('active', '#355570'), ('disabled', PANEL)],
              foreground=[('disabled', '#738390')])
    style.configure('TEntry', fieldbackground=PANEL, foreground=TEXT, insertcolor=TEXT)
    style.configure('TCombobox', fieldbackground=PANEL, foreground=TEXT, arrowcolor=TEXT)
    style.map('TCombobox', fieldbackground=[('readonly', PANEL)],
              foreground=[('readonly', TEXT)], selectbackground=[('readonly', '#263c50')])
    for name in ('TRadiobutton', 'TCheckbutton'):
        style.configure(name, background=BG, foreground=TEXT)
        style.map(name, background=[('active', PANEL)])
    style.configure('Vertical.TScrollbar', background='#31485a', troughcolor=PANEL, borderwidth=0)
    root.option_add('*TCombobox*Listbox.background', PANEL)
    root.option_add('*TCombobox*Listbox.foreground', TEXT)
    root.option_add('*TCombobox*Listbox.selectBackground', '#355570')


class Inspector:
    def __init__(self, root, subtitle, badge):
        if sys.platform == 'win32':
            # Set identity before the window is shown so Explorer does not group it under Python.
            shell32 = ctypes.OleDLL('shell32')
            shell32.SetCurrentProcessExplicitAppUserModelID.argtypes = [ctypes.c_wchar_p]
            shell32.SetCurrentProcessExplicitAppUserModelID('SystemX.AIInspector')
            root.iconbitmap(str(Path(__file__).resolve().parents[1] / 'systemx-icon.ico'))
        theme(root)
        header = ttk.Frame(root, padding=(16, 12, 16, 8))
        header.pack(fill='x')
        self.badge = tk.StringVar(value=badge)
        tk.Label(header, textvariable=self.badge, bg='#203850', fg=TEXT,
                 font=('Segoe UI', 9), padx=12, pady=7).pack(side='right', padx=(12, 0))
        ttk.Label(header, text='SystemX / AI Inspector', font=('Segoe UI', 17, 'bold')).pack(anchor='w')
        ttk.Label(header, text=subtitle, style='Muted.TLabel').pack(anchor='w', pady=(4, 0))
        self.controls = ttk.Frame(root, padding=(16, 0, 16, 6))
        self.controls.pack(fill='x')
        self.status = tk.StringVar(value='Ready to configure.')
        self.banner = tk.Label(root, textvariable=self.status, bg=STATUS['idle'], fg=TEXT,
                               font=('Segoe UI', 10, 'bold'), anchor='w', justify='left', padx=10, pady=8)
        self.banner.pack(fill='x', padx=16, pady=(2, 8))
        self.body = ttk.Frame(root, padding=(16, 0, 16, 0))
        self.body.pack(fill='both', expand=True)
        self.body.columnconfigure(0, weight=3)
        self.body.rowconfigure(0, weight=1)
        self.preview = tk.Frame(self.body, bg=PANEL)
        self.preview.grid(row=0, column=0, sticky='nsew', padx=(0, 12))
        tk.Label(self.preview, text='OBSERVATION', bg=PANEL, fg=MUTED, anchor='w',
                 font=('Segoe UI', 9, 'bold'), padx=10, pady=8).pack(fill='x')
        self.canvas = tk.Canvas(self.preview, width=640, height=440, bg='#0a111a', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.create_text(20, 25, text='Connect a game to inspect its visible state.',
                                fill=MUTED, anchor='nw', font=('Segoe UI', 10))
        self.panel = tk.Frame(self.body, bg=PANEL)
        self.panel.grid(row=0, column=1, sticky='nsew')
        self.body.columnconfigure(1, weight=2)
        tk.Label(self.panel, text='INSPECTOR', bg=PANEL, fg=TEXT, anchor='w',
                 font=('Segoe UI', 11, 'bold'), padx=12, pady=10).pack(fill='x')
        scroll = ttk.Scrollbar(self.panel, orient='vertical')
        scroll.pack(side='right', fill='y')
        self.text = tk.Text(self.panel, width=34, height=12, bg=PANEL, fg=TEXT,
                            insertbackground=TEXT, selectbackground='#355570', relief='flat',
                            font=('Consolas', 10), wrap='word', padx=12, pady=4,
                            yscrollcommand=scroll.set, state='disabled')
        self.text.pack(fill='both', expand=True)
        scroll.configure(command=self.text.yview)
        self.text.tag_configure('heading', foreground=ACCENT)
        ttk.Label(root, text='System 0: visible evidence   >   System 1: bounded choices   >   Controller: verified actions',
                  style='Muted.TLabel', font=('Segoe UI', 9), padding=(16, 8)).pack(fill='x')
        self.stacked = None
        self.body.bind('<Configure>', self.layout)
        root.bind('<Configure>', lambda e: self.banner.configure(wraplength=max(240, e.width - 52))
                  if e.widget == root else None, add='+')

    def layout(self, event):
        stacked = event.width < 900
        if stacked == self.stacked:
            return
        self.stacked = stacked
        self.body.columnconfigure(1, weight=0 if stacked else 2)
        self.body.rowconfigure(1, weight=1 if stacked else 0)
        self.preview.grid_configure(padx=0 if stacked else (0, 12), pady=(0, 8) if stacked else 0)
        self.panel.grid_configure(row=1 if stacked else 0, column=0 if stacked else 1)

    def describe(self, text):
        position = self.text.yview()[0]
        self.text.configure(state='normal')
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', text)
        for number, line in enumerate(text.splitlines(), 1):
            if line and line.split()[0].isupper():
                self.text.tag_add('heading', f'{number}.0', f'{number}.end')
        self.text.configure(state='disabled')
        self.text.yview_moveto(position)

    def fit(self, width, height):
        available_w = self.canvas.winfo_width()
        available_h = self.canvas.winfo_height()
        if available_w <= 1 or available_h <= 1:
            available_w, available_h = 640, 440
        return min(available_w / width, available_h / height, 1.0)
