import os
import yaml
import shutil
import argparse
from attributedict.collections import AttributeDict

from afa.engine import Engine
from afa.utils import create_logger, set_seed

# set random seeds
set_seed(seed=42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="Path to the YAML config file")
    parser.add_argument('--save_dir', default='./logs/run-temp', help='Directory to save config and model checkpoint')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', default=False,
                       help='Run training on train-set and evaluation on the validation-set')
    group.add_argument('--evaluate', action='store_true', default=False,
                       help='Only run testing - ensure the checkpoint path is provided in the config file')
    args = parser.parse_args()

    # Create the save directory
    exist_ok = True if 'temp' in args.save_dir else False
    os.makedirs(args.save_dir, exist_ok=exist_ok)

    # Copy the provided config file into save_dir
    shutil.copyfile(args.config_path, os.path.join(args.save_dir, "config.yaml"))

    # Create the logger
    logger = create_logger(name=args.save_dir)

    # process config
    with open(args.config_path) as f:
        config = AttributeDict(yaml.load(f, Loader=yaml.FullLoader))

    # Create the engine
    engine = Engine(config=config, logger=logger, save_dir=args.save_dir)

    if args.train:
        engine.train()
    elif args.evaluate:
        engine.evaluate()

# python run.py --config_path ./configs/default.yaml --save_dir ./logs/temp_run --train
# python run.py --config_path ./configs/default.yaml --save_dir ./logs/temp_run_eval --evaluate
