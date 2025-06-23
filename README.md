# Active Feature Acquisition

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


## Installation
To install and run this project locally, follow these steps:

I used conda=23.11.0

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

## evaluate
```
python run.py --config_path ./configs/default.yaml --save_dir ./logs/eval-run-temp --evaluate
```
