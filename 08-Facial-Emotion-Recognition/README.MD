# Real-Time Facial Emotion Detection (MediaPipe + OpenCV)

A simple webcam application that detects **Happy, Sad, Surprised, Scared,
Sleepy, and Neutral** facial expressions in real time, using
[MediaPipe Face Mesh](https://developers.google.com/mediapipe) landmarks and
[OpenCV](https://opencv.org/).

## How it works

1. **Calibration (3 seconds):** when the program starts, it asks you to keep
   a neutral face and stay still. During this time it collects several
   frames and averages them into a `baseline` (your neutral-face reference).
2. **Live detection:** every subsequent frame is compared **as a ratio** to
   that baseline, instead of being checked against fixed numbers.

> Instead of fixed numbers, ratio-based multipliers relative to the user's
> own neutral face are used. This way it also works across different face
> shapes and different cameras.

Press **`r`** at any time to reset and redo the calibration (useful if
lighting changes or a different person sits down). Press **`q`** to quit.

## Raw metrics

| Metric | What it measures | Landmarks used |
|---|---|---|
| `eye` | Vertical eyelid opening, normalized by face height | 159 (upper lid) - 145 (lower lid) |
| `brow` | Eyebrow-to-eye distance, normalized by face width | 65 (brow) - 159 (eye) |
| `mopen` | Vertical mouth opening, normalized by face height | 13 (upper lip) - 14 (lower lip) |
| `mwidth` | Mouth width, normalized by face width | 61 (left corner) - 291 (right corner) |
| `drop` | How far the mouth corners sag below the mouth's vertical center, normalized by face height | corner average - (13, 14) average |

## Real values measured during testing

The following two examples were captured live from the webcam (from the
debug line shown on screen):

| State | eye | brow | mopen | mwidth | drop |
|---|---|---|---|---|---|
| Neutral face | 0.044 | 0.118 | 0.000 | 0.380 | -0.014 |
| Surprised expression (attempted) | 0.046 | 0.157 | 0.107 | 0.372 | -0.004 |

You can capture your own values the same way (via the on-screen debug line)
and update the code's thresholds to match your own facial expressions and
camera setup.

From these two rows the resulting ratios are:
- `brow_ratio` = 0.157 / 0.118 ≈ **1.33**
- `mopen_ratio` (baseline mopen ≈ 0, so divided by the 0.01 fallback) ≈ **10.7**
- `eye_ratio` = 0.046 / 0.044 ≈ **1.05**

**Bug found from this data:** the code originally required `brow_ratio >
1.35`; the real surprised expression only reached 1.33, so it fell just
short of the threshold and was classified as "Neutral". The threshold was
lowered to **1.20**.

**Sleepy bug:** the original threshold was `eye_ratio < 0.55`, which
required the eyes to be almost fully closed. A half-closed / squinting eye
never crossed it. The threshold was loosened to **0.75**, so a ≥25% drop in
eye opening relative to neutral now counts as sleepy.

## Current thresholds

| Emotion | Condition | Reasoning |
|---|---|---|
| **Sleepy** | `eye_ratio < 0.75` | Eyes are at least 25% more closed than neutral (covers squinting, not just fully shut) |
| **Surprised** | `brow_ratio > 1.20` **and** `mopen_ratio > 1.5` **and** `mopen_abs > 0.03` | Eyebrows clearly raised **and** mouth actually (absolutely) open |
| **Scared** | `brow_ratio > 1.20` **and** `eye_ratio > 1.15` | Eyebrows raised, eyes widened, but mouth not as open as in "surprised" |
| **Sad** | `drop_delta > 0.008` **and** `mwidth_ratio < 1.05` | Mouth corners sag below neutral, mouth not stretched wide |
| **Happy** | `mwidth_ratio > 1.15` | Mouth is at least 15% wider than neutral |
| **Neutral** | none of the above | — |

The `mopen_abs > 0.03` guard exists because, when the neutral-face `mopen`
baseline is near zero, the ratio (`mopen_ratio`) can spike from a tiny
absolute change (e.g. going from 0.005 to 0.01 already looks like "+100%").
The absolute threshold prevents this from causing false positives.

## Tuning it for yourself

Run the script, note the debug values (`eye`, `brow`, `mopen`, `mwidth`,
`drop`) shown on screen for different expressions, divide them by your
`baseline` to get the ratios, and compare against the table above. If an
emotion is still misclassified, nudge the corresponding threshold (`1.20`,
`0.75`, `1.5`, etc.) slightly past the ratio you actually measured.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python duygu_tanima.py
```

1. Wait for the 3-second calibration to finish (keep a neutral face).
2. Try different expressions and watch the detected emotion + debug values
   on screen.
3. Press `r` to recalibrate, `q` to quit.

## Notes / Limitations

- Works with one face at a time (`max_num_faces = 1`).
- Thresholds were tuned on a single person / camera setup and are meant as
  a starting point — recalibrate for your own face if results feel off.
- This is a simple geometric heuristic (distances between landmarks), not a
  trained ML emotion classifier, so it can be fooled by unusual expressions,
  glasses, facial hair, or poor lighting.