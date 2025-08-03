import argparse

import torch

from dataset import StockDataset
from utils import DatasetGenerator


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Path of the generator checkpoint')
parser.add_argument('--output_path', required=True, help='Path of the output .npy file')
parser.add_argument('--delta_path', default='', help='Path of the file containing the list of deltas for conditional generation')
parser.add_argument('--dataset', default="D05.SI", help='dataset to use for normalization')
parser.add_argument('--size', default=1000, help='Size of the dataset to generate in case of unconditional generation')
opt = parser.parse_args()

# If an unknown option is provided for the dataset, then don't use any normalization
dataset = StockDataset(opt.dataset) if opt.dataset else None

model = torch.load(opt.checkpoint_path)
generator = DatasetGenerator(generator=model, dataset=dataset) #Using default params

delta_list = [float(line) for line in open(opt.delta_path)] if opt.delta_path != '' else None

# Size is ignored if delta_list is not None: it is inferred as the length of the list of deltas
generator.generate_dataset(outfile=opt.output_path, delta_list=delta_list, size=opt.size)
