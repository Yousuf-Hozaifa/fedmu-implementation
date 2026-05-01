# Fed-MU: Modality-Preserving Federated Unlearning

A lightweight research implementation of the Fed-MU proposal:
**"Federated Unlearning for Decentralised Multimodal Medical AI Systems"**

## Research Goal
Implement verified, efficient deletion of patient-specific data from federated
multimodal medical AI models, without full retraining.

## Status
- [x] Step 1: Environment setup
- [x] Step 2: Foundational understanding + toy pipeline
- [x] Step 3: LoRA fine-tuning
- [x] Step 4: Gradient concentration ratio ρ(p)
- [x] Step 5: LoRA-Subtraction algorithm
- [x] Step 6: Federated protocol
- [ ] Step 7: Experiments E1–E6
- [ ] Step 8: Analysis

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/fedmu-implementation.git
cd fedmu-implementation
python3.11 -m venv fedmu_env
source fedmu_env/bin/activate
pip install -r requirements.txt
python utils/smoke_test.py
```

## Hardware
- Local: CPU only, 8GB RAM (uses tiny model variants)
- Colab: A100 GPU (uses LLaVA-1.5-7B — see `configs/fedmu_config.yaml`)
