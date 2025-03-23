import sys
import numpy as np
import torch
import os
import pickle
import logging
from pathlib import Path
from deepchem.models.torch_models.protac_model import DeepPROTAC
print(DeepPROTAC)

Path('log').mkdir(parents=True, exist_ok=True)
Path('model').mkdir(parents=True, exist_ok=True)

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from protacloader import PROTACSet, collater
from prepare_data import GraphData

BATCH_SIZE = 1
EPOCH = 30
TRAIN_RATE = 0.8
LEARNING_RATE = 0.0001
root = "data"
TRAIN_NAME = "test"
logging.basicConfig(filename="log/"+TRAIN_NAME+".log", filemode="w", level=logging.DEBUG)

def main():
    # Load data
    ligase_ligand = GraphData("ligase_ligand", root)
    ligase_pocket = GraphData("ligase_pocket", root)
    target_ligand = GraphData("target_ligand", root)
    target_pocket = GraphData("target_pocket", root)
    with open(os.path.join(target_pocket.processed_dir, "smiles.pkl"), "rb") as f:
        smiles = pickle.load(f)
    with open('name.pkl', 'rb') as f:
        name_list = pickle.load(f)
    label = torch.load(os.path.join(target_pocket.processed_dir, "label.pt"))

    # Create dataset
    protac_set = PROTACSet(
        name_list,
        ligase_ligand, 
        ligase_pocket, 
        target_ligand, 
        target_pocket, 
        smiles, 
        label,
    )

    # Split dataset into train and test
    data_size = len(protac_set)
    train_size = int(data_size * TRAIN_RATE)
    test_size = data_size - train_size
    logging.info(f"all data: {data_size}")
    logging.info(f"train data: {train_size}")
    logging.info(f"test data: {test_size}")
    train_dataset = torch.utils.data.Subset(protac_set, range(train_size))
    test_dataset = torch.utils.data.Subset(protac_set, range(train_size, data_size))

    # Create DataLoader
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collater, drop_last=False, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collater, drop_last=False)

    # Initialize model
    model = DeepPROTAC()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(f'runs/{TRAIN_NAME}')

    # Train the model
    model.fit(trainloader, nb_epoch=EPOCH)

    torch.save(model, f"model/workingmodels.pt")
if __name__ == "__main__":
    Path('log').mkdir(exist_ok=True)
    Path('model').mkdir(exist_ok=True)
    main()