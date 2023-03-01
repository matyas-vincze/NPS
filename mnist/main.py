import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import dataclasses
import argparse
import wandb
import json
import pprint
import os

from data import MNISTData
from model import MnistSequentialNPS, ModelArgs
from utils import set_seed, custom_argparser, save_transformations
from train import train_epoch

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
# basic args
parser.add_argument("--cuda", type=bool, help="Enable CUDA")
parser.add_argument("--seed", type=int, help="Random seed")
parser.add_argument("--load_model", type=bool, help="Load model")
parser.add_argument("--model_path", type=str, help="Model path")
parser.add_argument("--wandb", type=bool, help="Enable wandb logging")
# training args
parser.add_argument("--n_epochs", type=int, help="Number of epochs to train for.")
parser.add_argument("--batch_size", type=int, help="Batch size for training and testing.")
parser.add_argument('--lr', type=float, help='Learning rate')
# model args
parser.add_argument("--n_operations", type=int, help="Number of operations.")
parser.add_argument("--dim_slot_embed", type=int, help="Dimension of slot embeddings.")
parser.add_argument("--dim_rule_embed", type=int, help="Dimension of rule embeddings.")
parser.add_argument("--dropout_prob", type=float, help="Dropout probability.")

args = ModelArgs(**custom_argparser())

if args.wandb:
    wandb.init(
        project = "NeuralProductionSystem",
        config = dataclasses.asdict(args)
    )


def main(args: ModelArgs):
    
    print(f'\n# {"-" * 5} SETUP {"-" * 5} #\n')

    # print args
    print("Configuration:")
    pprint.pprint(dataclasses.asdict(args))
    print("\n")

    # define and print device
    if args.cuda:
        assert torch.cuda.is_available() == True, "CUDA not available, please run with --cuda=False"
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Using device: {device}")

    # set and print seed
    set_seed(args.seed, args.cuda)
    print(f"Using seed: {args.seed}\n")

    # load/download data and print dims
    train_data = MNISTData(args, root='./data/', train=True)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_data = MNISTData(args, root='./data/', train=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Data loaded: {len(train_data)} train examples, {len(test_data)} test examples")

    # define model, loss, and optimizer
    model = MnistSequentialNPS(args).to(device)
    metric = BCELoss()
    optim = Adam(model.parameters(), lr=args.lr)

    # load model if specified
    loss_min = float('inf')
    results_path = os.path.dirname(os.path.realpath(__file__)) + '/res'

    if args.load_model:
        model_path = results_path + '/' + args.model_path
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_min = checkpoint['loss']
        print(f"Loaded model from {model_path} with loss {loss_min}")
    else:
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        print("Training from scratch")

    print(f'\n# {"-" * 5} TRAIN {"-" * 5} #\n')

    # train model
    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch + 1}/{args.n_epochs}")
        
        loss, selected_rules_count, selected_slots_count = train_epoch(model, train_dataloader, optim, metric, device)
        print(selected_rules_count, selected_slots_count)
        
        if args.wandb:
            wandb.log({"loss": loss})

        if loss < loss_min:
            loss_min = loss
            print('...saving new best model checkpoint...')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss_min
            }, results_path + '/' + args.model_path)
            
    print(f'\n# {"-" * 5} TEST {"-" * 5} #\n')
            
    # save transformation examples (first of each batch)
    save_transformations(model, test_dataloader, device, num_samples=5)
        
    print(f"Model, transformed images and attention counts saved to {results_path}")           

if __name__ == '__main__':
    main(args)
