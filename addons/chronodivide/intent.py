"""Standing intent written by System One (or rule defaults) and read by reflexes every tick.

System One answers short, bounded questions: the persistent workflow focus after an incident,
what the money is spent on (objective), how many tanks it needs before it attacks (attack size), what to find
out about the battlefield next (intel: scouting and camera looks), and what each production
lane builds next. Each answer updates standing intent with provenance, so one quick
decision steers many fast, deterministic inputs afterwards. The questions, their words and
their rule defaults are in the System One profile (playbook.py).
"""

from dataclasses import dataclass, field
import threading
import time


@dataclass
class StandingIntent:
    focus: str = 'build'                             # persistent workflow priority, changed by incidents
    focus_event: tuple | None = None                 # last incident considered by the model
    objective: str = 'build_army'
    posture: str = 'gather'
    intel: str | None = None                          # a requested look or scout, until the army does it
    resources: str = 'keep_home'                       # whether to send a verified miner to a known field
    attack_size: int | None = None                    # tanks at home before an attack; None: the rule default
    production: dict = field(default_factory=dict)    # lane -> item id
    provenance: dict = field(default_factory=dict)    # field -> (source, time)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, name, value, source, key=None):
        with self.lock:
            if key is None:
                setattr(self, name, value)
                self.provenance[name] = (source, time.monotonic())
            else:
                getattr(self, name)[key] = value
                self.provenance[f'{name}:{key}'] = (source, time.monotonic())

    def view(self):
        with self.lock:
            return dict(focus=self.focus, focus_event=self.focus_event,
                        objective=self.objective, posture=self.posture, intel=self.intel,
                        resources=self.resources,
                        attack_size=self.attack_size, production=dict(self.production),
                        provenance={k: v[0] for k, v in self.provenance.items()})
