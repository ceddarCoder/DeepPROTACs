import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence


# def collater(data_list):
#     batch = {}
#     name = [x["name"] for x in data_list]
#     ligase_ligand = [x["ligase_ligand"] for x in data_list]
#     ligase_pocket = [x["ligase_pocket"] for x in data_list]
#     target_ligand = [x["target_ligand"] for x in data_list]
#     target_pocket = [x["target_pocket"] for x in data_list]
#     smiles = [torch.tensor(x["smiles"]) for x in data_list]
#     smiles_length = [len(x["smiles"]) for x in data_list]
#     label = [x["label"] for x in data_list]

#     batch["name"] = name
#     batch["ligase_ligand"] = Batch.from_data_list(ligase_ligand)
#     batch["ligase_pocket"] = Batch.from_data_list(ligase_pocket)
#     batch["target_ligand"] = Batch.from_data_list(target_ligand)
#     batch["target_pocket"] = Batch.from_data_list(target_pocket)
#     batch["smiles"] = torch.nn.utils.rnn.pad_sequence(smiles, batch_first=True)
#     batch["smiles_length"] = smiles_length
#     batch["label"]=torch.tensor(label)
#     return batch
from torch_geometric.data import Batch

from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence

def collater(batch):
    """Collate function to prepare batches for PyTorch Geometric."""
    # Extract items from the batch
    ligase_ligand = [item['ligase_ligand'] for item in batch]
    ligase_pocket = [item['ligase_pocket'] for item in batch]
    target_ligand = [item['target_ligand'] for item in batch]
    target_pocket = [item['target_pocket'] for item in batch]
    smiles = [torch.tensor(item['smiles']) for item in batch]  # Convert each SMILES to a tensor
    smiles_length = [len(item['smiles']) for item in batch]  # Get lengths of SMILES sequences
    labels = [item['label'] for item in batch]  # This is a list of integers

    # Collate Data objects using PyTorch Geometric's Batch.from_data_list
    ligase_ligand = Batch.from_data_list(ligase_ligand)
    ligase_pocket = Batch.from_data_list(ligase_pocket)
    target_ligand = Batch.from_data_list(target_ligand)
    target_pocket = Batch.from_data_list(target_pocket)

    # Pad smiles sequences to the same length
    smiles = pad_sequence(smiles, batch_first=True, padding_value=0)  # Pad with 0

    # Convert labels to a tensor
    labels = torch.tensor(labels)  # Convert list of integers to a tensor

    # Prepare inputs as a dictionary
    inputs = {
        'ligase_ligand': ligase_ligand,
        'ligase_pocket': ligase_pocket,
        'target_ligand': target_ligand,
        'target_pocket': target_pocket,
        'smiles': smiles,
        'smiles_length': smiles_length,  # Include smiles_length
    }

    # Weights can be None if not used
    weights = None

    return inputs, labels, weights

class PROTACSet(Dataset):
    def __init__(self, name, ligase_ligand, ligase_pocket, target_ligand, target_pocket, smiles, label):
        super().__init__()
        self.name = name
        self.ligase_ligand = ligase_ligand
        self.ligase_pocket = ligase_pocket
        self.target_ligand = target_ligand
        self.target_pocket = target_pocket
        self.smiles = smiles
        self.label = label

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        # Calculate the length of the SMILES sequence
        smiles_length = len(self.smiles[idx])

        sample = {
            "name": self.name[idx],
            "ligase_ligand": self.ligase_ligand[idx],
            "ligase_pocket": self.ligase_pocket[idx],
            "target_ligand": self.target_ligand[idx],
            "target_pocket": self.target_pocket[idx],
            "smiles": self.smiles[idx],
            "smiles_length": smiles_length,  # Include smiles_length
            "label": self.label[idx],
        }
        return sample

