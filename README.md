# Wafer Map Defect Classification (WM-811K)

Mini Project for the course Deep Learning (23ELC364), Amrita Vishwa Vidyapeetham.

Comparison of ten deep learning architectures for classifying semiconductor wafer-map defect patterns, followed by experiments to push macro-F1 as high as possible.

**Result:** the final system (GoogLeNet + ResNet-18 ensemble on 64x64 maps with tuned class weights) reaches **90.97% macro-F1 / 98.22% accuracy** on a lot-held-out test set, up from 87.97% / 96.96% for the best baseline model. The weakest class, `Scratch`, improved from 0.68 to 0.83 F1.

Full write-up: [`results/REPORT.md`](results/REPORT.md). Ranking of all 16 trained models: [`results/summary.md`](results/summary.md).

## Task and evaluation
- Data: WM-811K, 172,950 labelled wafer maps, 9 classes (Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none).
- ~85% of wafers are `none`, so **macro-F1** is the primary metric; accuracy is reported but is not discriminating.
- Lot-aware 80/10/10 split (no wafer lot appears in more than one of train/val/test). Every model uses the same split, loss (class-weighted, label smoothing), optimiser and epoch budget.

## Models
| # | Script | Architecture |
|---|---|---|
| 1 | `algorithms/01_cnn.py` | Small CNN |
| 2 | `algorithms/02_alexnet.py` | AlexNet-style |
| 3 | `algorithms/03_vgg.py` | VGG-style |
| 4 | `algorithms/04_googlenet.py` | GoogLeNet / Inception-style |
| 5 | `algorithms/05_resnet.py` | ResNet-18 from scratch |
| 6 | `algorithms/06_transfer_resnet.py` | ImageNet ResNet-18, fine-tuned |
| 7 | `algorithms/07_rnn.py` | RNN |
| 8 | `algorithms/08_lstm.py` | LSTM |
| 9 | `algorithms/09_autoencoder.py` | Autoencoder pre-training + classifier |
| 10 | `algorithms/10_transformer.py` | Vision-transformer-style |

## Results at a glance (test set)
| System | Macro-F1 | Accuracy |
|---|---|---|
| Worst baseline (RNN) | 60.46% | 93.14% |
| Best baseline (ResNet-18 transfer, 32x32) | 87.97% | 96.96% |
| + augmentation, TTA, 64x64 input (GoogLeNet) | 89.53% | 97.69% |
| + ensemble and tuned class weights (final) | **90.97%** | **98.22%** |

## Repository layout
```
algorithms/    one training script per architecture + common.py (shared data, training, evaluation)
prep_data.py   builds data/full_dataset[_64].npz from the raw WM-811K pickle
compare.py     ranks all trained models -> results/summary.md
ensemble.py    ensembles saved probabilities and tunes class weights on validation
run_all.sh     trains all ten baselines and prints the comparison
results/       per-model JSON metrics, summary.md, REPORT.md, saved probabilities, ensemble results
checkpoints/   trained weights (only the two final models are tracked in git)
logs/          training / preprocessing logs
legacy/        first-iteration CNN, inference script and early metrics (kept for history)
```

## Reproducing
1. Install: `pip install -r requirements.txt` (a CUDA GPU is strongly recommended; about 4 GB was enough with the batch sizes used).
2. Get the data: download the WM-811K dataset (`LSWMD.pkl`, e.g. from Kaggle "WM-811K wafer map") into `data/`.
3. Prepare: `python prep_data.py` (32x32) and `python prep_data.py 64` (64x64).
4. Baselines: `./run_all.sh`
5. Improved models: `V2=1 TAG=_v2 python algorithms/06_transfer_resnet.py`, and with 64x64 data: `WAFER_DATA=full_dataset_64.npz V2=1 TAG=_v2_64 python algorithms/06_transfer_resnet.py` (use `BATCH_SIZE=64` for `04_googlenet.py`).
6. Ensemble: `python ensemble.py _v2_64`, then `python compare.py`.

Environment switches (all optional): `V2=1` per-sample augmentation + 8-way test-time augmentation + saved probabilities; `TAG` result-name suffix; `WAFER_DATA` dataset file; `BATCH_SIZE` (default 256); `PATIENCE` early-stopping patience.

## Caveats
Single seed and single split, so differences of about 1 point between individual models are within noise. The tuned class weights were fitted on the validation set; the untuned ensemble average (90.70% macro-F1) is a more conservative figure. See the report for details.
