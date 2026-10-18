# WM-811K Wafer Defect Classification: Results Report

## Setup
- Data: WM-811K labelled wafer maps, resized to 32x32 (nearest neighbour), 9 classes (8 defect types + `none`).
- Split: lot-aware ~80/10/10 (`StratifiedGroupKFold`), so no wafer lot appears in more than one split. Train 138,376 / val 17,292 / test 17,282.
- Identical for every model: class-weighted cross-entropy (label smoothing 0.05), AdamW + OneCycle, up to 20 epochs, early stopping on validation macro-F1, one test evaluation.
- Primary metric: **macro-F1**. Accuracy is misleading because ~85% of wafers are `none`.

## 1. Baseline: ten architectures

| Rank | Model | Macro-F1 | Balanced acc | Accuracy | Params |
|---|---|---|---|---|---|
| 1 | transfer_resnet18 | 87.97% | 91.24% | 96.96% | 11.2M |
| 2 | googlenet | 87.31% | 90.58% | 97.03% | 0.55M |
| 3 | alexnet | 85.38% | 88.84% | 96.45% | 7.0M |
| 4 | vgg | 84.92% | 89.34% | 96.46% | 4.0M |
| 5 | resnet | 80.53% | 78.42% | 96.38% | 11.2M |
| 6 | cnn | 79.26% | 81.52% | 96.11% | 94K |
| 7 | lstm | 78.95% | 80.41% | 95.32% | 0.56M |
| 8 | autoencoder | 77.79% | 80.65% | 96.03% | 184K |
| 9 | transformer | 71.36% | 73.11% | 93.83% | 1.2M |
| 10 | rnn | 60.46% | 59.91% | 93.14% | 143K |

Full per-class F1 for all ten models is in `results/summary.md`.

Observations:
- Convolutional models with pretraining or multi-scale features (ResNet-18 transfer, GoogLeNet) lead. GoogLeNet matches the leader with 20x fewer parameters.
- Sequence models (RNN, Transformer) treat the map as a sequence and lose 2D structure. They fail on `Scratch` and, for the RNN, `Near-full`.
- Accuracy spans only 93-97% while macro-F1 spans 60-88%, which confirms macro-F1 is the discriminating metric.

## 2. Improvements on the top four models

Changes ("V2"): independent random rotation/reflection per wafer, plus 8-way dihedral test-time augmentation (TTA).

| Model | Baseline F1 | V2 F1 | Baseline acc | V2 acc |
|---|---|---|---|---|
| transfer_resnet18 | 87.97% | 87.86% | 96.96% | 97.19% |
| googlenet | 87.31% | 86.71% | 97.03% | 96.94% |
| alexnet | 85.38% | 87.34% | 96.45% | 97.27% |
| vgg | 84.92% | 82.43% | 96.46% | 96.23% |

V2 alone gives mixed results: a clear gain for AlexNet, roughly neutral for ResNet-18 and GoogLeNet, and a loss for VGG. These are single-seed runs, so differences of about 1 point should not be over-interpreted.

## 3. Ensemble and per-class decision weights

Probabilities from the V2 models were averaged, then one multiplicative weight per class was tuned by coordinate ascent on **validation** macro-F1. The configuration was selected on validation and evaluated once on test.

Selected: **GoogLeNet + ResNet-18 transfer, averaged, with tuned class weights.**

| System | Macro-F1 | Balanced acc | Accuracy |
|---|---|---|---|
| Best single baseline (transfer_resnet18) | 87.97% | 91.24% | 96.96% |
| **Selected ensemble + thresholds** | **88.69%** | 87.44% | **97.56%** |

Per-class F1, baseline best vs. selected ensemble:

| Class | Baseline (resnet18) | Ensemble |
|---|---|---|
| Center | 0.91 | 0.92 |
| Donut | 0.92 | 0.96 |
| Edge-Loc | 0.82 | 0.83 |
| Edge-Ring | 0.97 | 0.97 |
| Loc | 0.76 | 0.81 |
| Near-full | 0.97 | 0.93 |
| Random | 0.89 | 0.86 |
| Scratch | 0.68 | 0.71 |
| none | 0.99 | 0.99 |

Notes:
- Gains concentrate in `Loc`, `Donut` and `Scratch`; `Near-full` and `Random` got worse, likely because those classes have few test samples, so per-class F1 is noisy.
- Balanced accuracy fell (91.2% to 87.4%). The tuned weights favour precision on rare classes, which raises macro-F1 and accuracy at the cost of recall.
- A three-model ensemble (AlexNet + GoogLeNet + ResNet-18) scored 88.82% on test but ranked lower on validation, so it was not chosen. Selecting it for its test score would leak the test set into model selection.
- The gain over the baseline (+0.7 points macro-F1) is modest and from a single seed and split; it should be reported as such.

## 4. Native 64x64 input (the largest improvement)

Same V2 recipe, but trained on 64x64 maps (nearest-neighbour resize) instead of 32x32, for the two strongest architectures.

| Model | 32x32 baseline F1 | 64x64 V2 F1 | 64x64 balanced acc | 64x64 acc |
|---|---|---|---|---|
| transfer_resnet18 | 87.97% | 89.06% | 91.21% | 97.78% |
| googlenet | 87.31% | **89.53%** | 92.33% | 97.69% |

Both models improve, and `Scratch` F1 in particular: 0.68 to 0.71 (ResNet-18) and 0.64 to 0.77 (GoogLeNet). This supports the hypothesis that thin defects were being damaged by 32x32 downsizing. GoogLeNet needed batch size 64 instead of 256 to fit in 4 GB of GPU memory, so it is not a perfectly like-for-like comparison.

## 5. Final system: 64x64 ensemble + class weights

GoogLeNet + ResNet-18 (64x64, V2) probabilities averaged, then per-class weights tuned on validation macro-F1 only. The configuration was chosen on validation and scored once on test.

| System | Macro-F1 | Balanced acc | Accuracy |
|---|---|---|---|
| Best baseline (transfer_resnet18, 32x32) | 87.97% | 91.24% | 96.96% |
| Best single model (googlenet, 64x64, V2) | 89.53% | 92.33% | 97.69% |
| Ensemble, plain average | 90.70% | 93.27% | 97.97% |
| **Ensemble + tuned class weights (selected)** | **90.97%** | 90.42% | **98.22%** |

| Class | Baseline (resnet18) | Final system |
|---|---|---|
| Center | 0.91 | 0.93 |
| Donut | 0.92 | 0.92 |
| Edge-Loc | 0.82 | 0.87 |
| Edge-Ring | 0.97 | 0.98 |
| Loc | 0.76 | 0.86 |
| Near-full | 0.97 | 0.93 |
| Random | 0.89 | 0.86 |
| Scratch | 0.68 | 0.83 |
| none | 0.99 | 0.99 |

Notes:
- Total improvement over the best baseline: +3.0 points macro-F1 (87.97% to 90.97%) and +1.3 points accuracy.
- Validation macro-F1 for the selected system was 92.81% against 90.97% on test, so the tuned class weights are somewhat fitted to the validation set. The untuned average (90.70% F1, 93.27% balanced accuracy) is a more conservative alternative with nearly the same F1 and better balanced accuracy.
- `Near-full` and `Random` are lower than the baseline; both are rare classes, so their per-class F1 is noisy.

The ranking of all 16 trained models is in `results/summary.md`. The `googlenet_v2_64` train time there includes laptop suspend time and should be ignored.

## 6. Limitations and future work
- Single seed and single split: no confidence intervals. Repeating over several seeds would show whether differences of about 1 point (including most of Section 2) are real.
- Threshold weights were fit on the validation set, which also drove early stopping, so validation scores are slightly optimistic (see Section 5).
- Only two architectures were run at 64x64. The other models, and larger resolutions, were not tried.
- The 64x64 GoogLeNet used a different batch size from the other runs.

## Reproduction
- Baseline: `./run_all.sh`
- V2 models: `V2=1 TAG=_v2 .venv/bin/python algorithms/<script>.py`
- Ensemble: `.venv/bin/python ensemble.py _v2` (writes `results/ensemble_v2.json`)
- 64x64: `python prep_data.py 64`, then `WAFER_DATA=full_dataset_64.npz V2=1 TAG=_v2_64 python algorithms/06_transfer_resnet.py` and the same for `04_googlenet.py` with `BATCH_SIZE=64`, then `python ensemble.py _v2_64`
- Summary table: `python compare.py`
