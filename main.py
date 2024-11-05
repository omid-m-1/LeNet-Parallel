import torch
from torch.multiprocessing import Process
import torch.multiprocessing as mp
from torch.profiler import profile, ProfilerActivity

from util.model import LeNet
from util.util import train_model, evaluate_model
from util.util import setup_and_train as dist_train

import numpy as np
import random
import argparse
import os

if __name__ == "__main__":
    # Command-line arguments for model configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_rank', type=int, default=2, help='Max Number of GPUs')
    parser.add_argument('--kernel', default='Custom', help='valid options: [Custom, PyTorch]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for optimizer')
    parser.add_argument('--mixed', default='float32', help='valid options: [float32, mixed_tc, mixed_amp]')
    parser.add_argument('--use_ddp', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    # Hyperparameters
    max_rank = args.max_rank
    kernel = args.kernel
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    mixed = args.mixed
    test = args.test
    use_ddp = args.use_ddp

    # Set random seed
    seed_value = 42
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True

    # Launch a separate process for each GPU
    mp.set_start_method('spawn')
    processes = []
    ranks = torch.cuda.device_count()
    ranks = min(ranks, max_rank)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for rank in range(ranks):
            p = Process(target=dist_train, args=(kernel, rank, ranks, batch_size, epochs, learning_rate, mixed, use_ddp, test, seed_value))
            p.start()
            processes.append(p)

        # Ensure all processes complete before finishing
        for p in processes:
            p.join()

    print(prof.key_averages().table(row_limit=20))
