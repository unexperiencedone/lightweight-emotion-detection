import numpy as np
from emotion_edge.labels import CANON, TEXT_LABELS, SPEECH_LABELS, FACE_LABELS, MAPS
from emotion_edge.fusion.fuse import Modality, fuse, to_canonical


def onehot(labels, name, strength=6.0):
    z = np.zeros(len(labels)); z[labels.index(name)] = strength
    return z


def test_maps_rows_sum_to_one():
    for m in MAPS.values():
        assert np.allclose(m.sum(1), 1)


def test_agreement_is_confident():
    f = fuse([Modality("text", onehot(TEXT_LABELS, "anger")), Modality("speech", onehot(SPEECH_LABELS, "angry")),
              Modality("face", onehot(FACE_LABELS, "angry"))])
    assert f.label == "anger" and f.state == "confident"


def test_incongruence_flagged_as_conflict():
    f = fuse([Modality("text", onehot(TEXT_LABELS, "joy", 7)), Modality("speech", onehot(SPEECH_LABELS, "angry", 7))])
    assert f.state == "conflict" and "text->joy" in f.explanation


def test_blend_vs_ambiguous():
    # same-modality split between two V/A-compatible emotions -> blend
    z = np.full(6, -5.0); z[TEXT_LABELS.index("joy")] = z[TEXT_LABELS.index("love")] = 3.0
    assert fuse([Modality("text", z)]).state == "blend"
    z = np.full(6, -5.0); z[TEXT_LABELS.index("joy")] = z[TEXT_LABELS.index("anger")] = 3.0
    assert fuse([Modality("text", z)]).state == "ambiguous"


def test_flat_is_uncertain():
    assert fuse([Modality("text", np.zeros(6))]).state == "uncertain"


def test_missing_modality_and_low_quality_discount():
    t = Modality("text", onehot(TEXT_LABELS, "sadness"))
    bad_face = Modality("face", onehot(FACE_LABELS, "happy"), quality=0.05)    # tiny/blurry face must not override text
    assert fuse([t, bad_face]).label == "sadness"
    assert fuse([t, None]).label == "sadness"


def test_text_cannot_veto_classes_it_cannot_express():
    # text has no "neutral"; confident neutral from speech+face must survive a flat-ish text opinion
    f = fuse([Modality("text", np.zeros(6)), Modality("speech", onehot(SPEECH_LABELS, "neutral")),
              Modality("face", onehot(FACE_LABELS, "neutral"))])
    assert f.label == "neutral"


def test_temperature_changes_confidence():
    z = onehot(TEXT_LABELS, "fear", 4)
    hot = fuse([Modality("text", z, temperature=3.0)]); cold = fuse([Modality("text", z, temperature=1.0)])
    assert hot.p_top1 < cold.p_top1
