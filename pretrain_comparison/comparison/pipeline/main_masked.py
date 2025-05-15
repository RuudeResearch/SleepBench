import argparse

import sys
sys.path.append('../')
from train.train_MAE_masked import main_worker


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    main_worker(None, None, args.config)
    #end the job
    print("Job finished")
