"""System One's side of the player, read from profiles/startlux-ra2-soviet.json (see its "about").

The player measures facts and carries out decisions. The profile holds how to play: the
questions put to System One (instructions, option words, the facts each is asked about), the words
of every fact, each question's rule default (a first-match table over the facts' classes),
and the playbook numbers for the choices never put to System One. The code names only the option
ids it acts on (a posture of 'attack' sends waves) and the facts it can measure.
"""

import json
from pathlib import Path

PROFILE = Path(__file__).with_name('profiles') / 'startlux-ra2-soviet.json'
# Option ids the player acts on: a profile must offer each of them.
ACTS_ON = {'objective': ('expand_economy', 'build_army', 'tech_up'),
           'posture': ('gather', 'attack', 'defend'),
           'intel': ('scout', 'look_enemy_base', 'look_at_army', 'nothing')}


class Playbook:
    def __init__(self, data):
        if data.get('version') != 1:
            raise ValueError('Unsupported System One profile.')
        self.data = data
        self.path = None
        self.systemone, self.questions, self.words = data['systemone'], data['questions'], data['facts']
        for qid, ids in ACTS_ON.items():
            missing = set(ids) - set(self.questions[qid]['options'])
            if missing:
                raise ValueError(f'The System One profile question {qid} lacks options {sorted(missing)}.')
        for qid, question in self.questions.items():
            for rule in question.get('rules', ()):
                if rule['choose'] not in question['options']:
                    raise ValueError(f'A rule of {qid} chooses {rule["choose"]}, which is not an option.')
        for option in self.questions['attack_size']['options'].values():
            if not isinstance(option, dict) or not isinstance(option.get('tanks'), int):
                raise ValueError('Every attack_size option needs a number of tanks.')

    def __getitem__(self, section):
        """A playbook section: the numbers of one part of the plan."""
        return self.data['playbook'][section]

    # -- questions -------------------------------------------------------------
    def option(self, qid, option):
        """An option's entry: its words, and values such as tanks."""
        entry = self.questions[qid]['options'][option]
        return {'words': entry} if isinstance(entry, str) else entry

    def options(self, qid):
        return list(self.questions[qid]['options'])

    def question(self, qid, options):
        """{qid: question} offering these options, in this order, in the profile's words."""
        spec = self.questions[qid]
        return {qid: {'type': spec['type'], 'instructions': spec['instructions'],
                      'criteria': {key: self.option(qid, key)['words'] for key in options}}}

    def state(self, qid, facts):
        """The facts this question is asked about (only those: others confuse the model)."""
        return {name: facts[name] for name in self.questions[qid]['facts']}

    def rule(self, qid, options, classes):
        """The rule default: the first rule whose option is offered and whose conditions hold
        (a condition names a fact class, or a list of acceptable ones)."""
        for rule in self.questions[qid].get('rules', ()):
            if rule['choose'] in options and all(
                    classes.get(fact) in (want if isinstance(want, list) else [want])
                    for fact, want in rule.get('when', {}).items()):
                return rule['choose']
        return next(iter(options))

    # -- fact words ------------------------------------------------------------
    def say(self, family, fact, key, **values):
        """The words of one class of a fact, with its numbers filled in."""
        return self.words[family][fact][key].format(**values)

    def level(self, family, fact, value):
        """(class, words template) of the first level of a fact that holds for the value
        (at_least: value >= limit, within: value <= limit, neither: always)."""
        for level in self.words[family][fact]['levels']:
            if 'at_least' in level and value < level['at_least']:
                continue
            if 'within' in level and value > level['within']:
                continue
            return level['name'], level['words']
        raise ValueError(f'No level of {family}.{fact} holds for {value}.')


def load(path=PROFILE):
    book = Playbook(json.loads(Path(path).read_text(encoding='utf-8')))
    book.path = Path(path)
    return book


def knowledge(book):
    """Resolve faction knowledge beside the selected strategy, preserving the Soviet default."""
    from .world import load_knowledge
    name = book.data.get('knowledge')
    return load_knowledge((book.path or PROFILE).parent / name) if name else load_knowledge()


_default = None


def default():
    """The profile shipped with the player, loaded once."""
    global _default
    if _default is None:
        _default = load()
    return _default
