# PRECISE-AS: Official PyTorch Implementation

- **Venue**: MICCAI 2025
- **Paper**: PRECISE-AS: Personalized Reinforcement Learning for Efficient Point-of-Care Echocardiography in Aortic Stenosis Diagnosis 
- **Authors**: Armin Saadat, Nima Hashemi, Hooman Vaseli, Michael Y Tsang, Christina Luong, Michiel Van de Panne, Teresa SM Tsang, Purang Abolmaesumi
- **Institution(s)**: University of British Columbia, Vancouver General Hospital


<p align="center">
<a href="https://arxiv.org/abs/2509.02898" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2503.15784-b31b1b.svg?style=flat" /></a>
    [License](https://img.shields.io/badge/license-MIT-blue.svg)(LICENSE)
</p>

<img width="1037" height="506" alt="Screenshot 2025-09-27 at 11 35 23 AM" src="https://github.com/user-attachments/assets/9cee0020-668b-4c95-90ab-38551f4792b6" />


## Installation
To install and run this project locally, please follow these steps:

conda >= 23.11.0

```
git clone repo_address
conda create --name afa python=3.9 pip
conda activate afa
python -m pip install poetry
cd path_to_repo
poetry install
```

## train
```
python run.py --config_path ./configs/default.yaml --save_dir ./logs/run-temp --train
```
Note: In the config file, you can set WANDB_MODE=offline to avoid logging in to WANDB.


## evaluate
```
python run.py --config_path ./configs/default.yaml --save_dir ./logs/eval-run-temp --evaluate
```
Note: In the config file, you can set WANDB_MODE=offline to avoid logging in to WANDB.

