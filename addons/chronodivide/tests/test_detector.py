"""The unit detector: its network and export, labels from real frame pairs, and running it in the
server or in the player. No trained model is needed here."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from addons.chronodivide.perception.detector import Detection, Detector, side_of
from addons.chronodivide.perception.unitboxes import enemy_objects, our_units
from addons.chronodivide.tools import detector as tool

torch = pytest.importorskip('torch')
FIXTURES = Path(__file__).with_name('fixtures')


def frame(name):
    return cv2.imread(str(FIXTURES / name))


@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    """An untrained network in its exported form, with its metadata beside it."""
    from addons.chronodivide.perception.detector_net import Exported, UnitNet
    folder = tmp_path_factory.mktemp('model')
    path = folder / 'detector-units.pt'
    torch.manual_seed(0)
    torch.jit.save(torch.jit.script(Exported(UnitNet().eval())), str(path))
    import hashlib
    meta = dict(name='ra2-units', threshold=0.4, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.json').write_text(json.dumps(meta))
    return path


def test_the_exported_network_takes_a_frame_and_gives_ranked_boxes(exported):
    model = torch.jit.load(str(exported))
    view = frame('army-selected.png')[:736, :850]
    with torch.inference_mode():
        rows = model(torch.from_numpy(np.ascontiguousarray(view))).numpy()
    assert rows.shape == (128, 6)
    assert np.all(np.diff(rows[:, 4]) <= 1e-6) and set(np.unique(rows[:, 5])) <= {0.0, 1.0}
    assert np.all(rows[:, 2] >= rows[:, 0]) and np.all(rows[:, 3] >= rows[:, 1])


def test_training_targets_mark_unit_centers_and_never_ignore_a_labeled_unit():
    heat, wh, off, mask, skip = tool.targets(np.array([[40, 40, 72, 64, 1]], np.float32),
                                             np.array([[0, 0, 120, 120]], np.float32), (128, 128))
    assert heat[1].max() == 1.0 and heat[0].max() == 0.0 and mask.sum() == 1
    iy, ix = np.argwhere(mask)[0]
    assert (iy, ix) == (13, 14) and tuple(wh[:, iy, ix]) == (8.0, 6.0) and skip[iy, ix] == 0 and skip[0, 0] == 1


def test_a_unit_is_still_only_when_its_sprite_looks_the_same():
    canvas = frame('army-selected.png')
    unit = our_units(canvas)[0]
    assert tool.still(canvas, canvas.copy(), unit.box)
    moved = np.roll(canvas, 6, axis=1)
    assert not tool.still(canvas, moved, unit.box)


def test_pairs_need_an_unselected_view_from_the_same_camera():
    selected = frame('army-selected.png')
    assert tool.pair_labels(selected, selected) is None                  # the other view is selected too
    other = frame('base-reactors.png')                                   # another place: the camera "moved"
    assert tool.pair_labels(selected, other) is None
    labels = tool.label(selected)
    assert labels is not None and len(labels[0]) == len(our_units(selected))


def test_a_detection_belongs_to_the_side_whose_color_it_holds():
    ours = our_units(frame('army-selected.png'))
    assert side_of(frame('army-selected.png'), ours[0].box) == 'ours'
    enemy = [u for u in enemy_objects(frame('enemy-base-siege.png')) if u.kind == 'unit']
    assert side_of(frame('enemy-base-siege.png'), enemy[0].box) == 'enemy'
    assert side_of(frame('base-reactors.png'), (5, 700, 40, 730)) is None       # plain ground


def test_the_detector_runs_in_the_player_without_a_server(exported, monkeypatch):
    detector = Detector(exported, server=None)
    canvas = frame('army-selected.png')
    box = our_units(canvas)[0].box
    monkeypatch.setattr(detector, '_rows', lambda view: [[box[0], box[1], box[2], box[3], 0.9, 1.0],
                                                         [5, 700, 40, 730, 0.9, 0.0], [0, 0, 5, 5, 0.1, 0.0]])
    found = detector.detect(canvas, (0, 0, 850, 736))
    assert found == [Detection(tuple(box), 'ours', 'vehicle', 0.9)]    # plain ground and a weak peak dropped
    monkeypatch.undo()
    assert isinstance(detector.detect(canvas, (0, 0, 850, 736)), list) and detector.where == 'local'


class FakeClient:
    """The SystemX server as the detector sees it: its vision folder and what it has loaded."""
    folder, loaded, reloads, detects = None, {}, 0, 0

    def __init__(self, url, timeout=None):
        pass

    def vision(self):
        return dict(folder=str(FakeClient.folder), models=list(FakeClient.loaded.values()))

    def vision_reload(self):
        FakeClient.reloads += 1
        meta = json.loads((FakeClient.folder / 'detector-units.json').read_text())
        FakeClient.loaded = {meta['name']: dict(name=meta['name'], sha256=meta['sha256'])}
        return self.vision()

    def detect(self, name, view, min_score=None):
        FakeClient.detects += 1
        return []


def test_the_model_is_handed_to_a_local_server_once_and_runs_there(exported, tmp_path, monkeypatch):
    import systemone_client
    monkeypatch.setattr(systemone_client, 'SystemOneClient', FakeClient)
    FakeClient.folder, FakeClient.loaded, FakeClient.reloads = tmp_path / 'vision', {}, 0
    detector = Detector(exported, server='http://127.0.0.1:8000')
    assert detector.detect(frame('base-reactors.png'), (0, 0, 850, 736)) == []
    assert detector.where == 'server' and FakeClient.reloads == 1 and FakeClient.detects == 1
    assert (tmp_path / 'vision' / 'detector-units.pt').read_bytes() == exported.read_bytes()
    again = Detector(exported, server='http://127.0.0.1:8000')          # same model already loaded: no copy
    again.detect(frame('base-reactors.png'), (0, 0, 850, 736))
    assert again.where == 'server' and FakeClient.reloads == 1
    assert Detector(exported, server='http://192.168.1.5:8000').server is None   # only a server on this machine
