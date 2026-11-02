# Fed-MU: Modality-Preserving Federated Unlearning

[![ci](https://github.com/Yousuf-Hozaifa/fedmu-implementation/actions/workflows/ci.yml/badge.svg)](https://github.com/Yousuf-Hozaifa/fedmu-implementation/actions)

A lightweight research implementation of **"Federated Unlearning for Decentralised Multimodal Medical AI Systems"**:
remove a patient's data from a federated multimodal (image + report) model **without full retraining**, and verify that it is gone.

> Research code, not a service. It trains models with PyTorch and is run from the command line (CPU for the toy setup, a GPU for larger runs). There is no hosted demo.

## Idea in one paragraph
A multimodal model (CLIP vision encoder + DistilBERT text encoder + a 2-layer cross-modal projection) is wrapped with three LoRA adapters: **A_v** (vision), **A_l** (language) and **A×** (the cross-modal projection). Federated nodes train the adapters (FedAvg, with selective aggregation). To unlearn a patient, **LoRA-Subtraction** removes that patient's contribution from the adapters while a preservation term protects the other modalities (measured by a modality-preservation score, MPS). Deletion is then checked in three layers: membership-inference attack (MIA), reconstruction test, and KL divergence, plus an identity probe (E6).

## Repository map
| Path | What it does |
|---|---|
| `models/multimodal_model.py` | `FedMUModel` + `CrossModalProjection` (used by the whole pipeline); `MultimodalModel` (larger LLaVA/ViT-L backbone, not yet wired into LoRA) |
| `models/lora_model.py`, `lora_layer.py` | `FedMULoRAModel`: attaches A_v / A_l / A× and exposes gradient-norm queries |
| `federated/` | node training, data splitter, FedAvg server, selective aggregation |
| `unlearning/algorithms/lora_subtraction.py` | the unlearning algorithm |
| `unlearning/verification/` | MPS evaluator, MIA, reconstruction test, KL verifier |
| `experiments/e2_rho/` | gradient concentration ratio ρ(p) analysis (E2) |
| `experiments/e6_probe/` | identity probe (E6) |
| `experiments/pretrain.py` | federated pre-training |
| `data/` | synthetic toy patients, IU chest X-ray loader, node splits |
| `configs/fedmu_config.yaml` | all hyperparameters |
| `tests/` | static consistency tests (run in CI) |

## Quick start (CPU, synthetic toy data)
```bash
git clone https://github.com/Yousuf-Hozaifa/fedmu-implementation.git
cd fedmu-implementation
python3.11 -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
python utils/smoke_test.py          # libraries load
python -m pytest -q tests           # static consistency checks (no GPU/models needed)
```
Run everything **from the repo root**. The first model run downloads CLIP and DistilBERT from Hugging Face (about 400 MB).

```bash
python utils/test_pipeline.py              # Step 2: forward pass on toy data
python utils/test_lora.py                  # Step 3: LoRA adapters
python experiments/e2_rho/run_e2.py        # Step 4: rho(p) analysis
python experiments/pretrain.py             # Step 5: federated pre-training -> models/checkpoints/
python unlearning/run_unlearning.py        # Step 6: LoRA-Subtraction
python unlearning/run_verification.py      # Step 7: MIA / reconstruction / KL
```
Synthetic images are generated deterministically on first use (they are git-ignored), so a fresh clone works.

## Real data (Indiana University chest X-rays, Kaggle)
Set `data.dataset_path`, `image_folder` and `reports_file` in `configs/fedmu_config.yaml` (defaults point to `/kaggle/input/...`). `experiments/pretrain.py` switches to this dataset automatically when `dataset_path` exists.

## Results
_Add your measured numbers here from your own runs (MPS, ΔLoss, MIA AUC, KL ratio, identity-probe score, ρ(p) statistics). This repository does not currently include result files or figures._

## Known limitations
- **Labels:** the IU loader uses all-zero 14-class labels (the dataset has none), so supervised pathology performance is not meaningful on real data.
- **Large-model mode:** `MultimodalModel` (LLaVA-1.5-7B / ViT-L) is not yet compatible with `FedMULoRAModel`; all pipeline scripts use `FedMUModel`.
- **Hyperparameters** (see the history in the config) were tuned on the toy setup.
- Running `pretrain.py` on IU data overwrites `data/toy_patients/metadata.json`; restore it with `git checkout data/toy_patients/metadata.json` before toy runs.
- `configs/fedmu_config.yaml` says 2000 patients, while the committed toy metadata and splits cover 50. Delete `data/splits/` to re-split after changing the size.

## Citation / status
Work in progress; the associated paper is not yet published.
