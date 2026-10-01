# Running the speech and face tracks (needs datasets; not bundled)

    # speech: RAVDESS (Zenodo 1188976, Audio_Speech_Actors_01-24.zip) or CREMA-D (github.com/CheyneyComputerScience/CREMA-D)
    PYTHONPATH=. python -m emotion_edge.speech.train --dataset ravdess --root data/ravdess --out artifacts/speech
    # face: FER2013 fer2013.csv (Kaggle) -- or FER+ labels
    PYTHONPATH=. python -m emotion_edge.vision.train --csv data/fer2013.csv --out artifacts/face --epochs 60

Both write `results.json` (accuracy, macro-F1, confusion, int8 size/accuracy, temperature) and `logits.npz`
(validation + test logits, used to calibrate and to tune fusion thresholds on real data).
