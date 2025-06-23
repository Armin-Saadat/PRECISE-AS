import os.path
import torch
from torch.utils.data import Dataset


class ASTomSimDataset(Dataset):
    def __init__(self, dataset_root: str = 'data/ASTom/processed', split: str = "train"):
        if split in ['val', 'test']:
            data_dict = torch.load(os.path.join(dataset_root, 'sim_data.pth'))
            self.similarities = data_dict['similarities'][split] # [n, 4, 40]
            self.labels = data_dict['labels'][split] # [n]
            self.dopplers = data_dict['dopplers'][split] # [n, 3]

        elif split in ['train']: # for training the RL, not the classifier.
            data_dict = torch.load(os.path.join(dataset_root, 'sim_data_ge2.pth'))
            similarities = data_dict['similarities'][split]  # [n, 4, 40]
            # Intra-View Permutation
            _0_1_2_3 = similarities[:, [0, 1, 2, 3], :].clone()
            _1_0_2_3 = similarities[:, [1, 0, 2, 3], :].clone()
            _0_1_3_2 = similarities[:, [0, 1, 3, 2], :].clone()
            _1_0_3_2 = similarities[:, [1, 0, 3, 2], :].clone()
            self.similarities = torch.cat((_0_1_2_3, _1_0_2_3, _1_0_3_2, _0_1_3_2), dim=0)
            self.labels = data_dict['labels'][split].repeat(4)
            self.dopplers = data_dict['dopplers'][split].repeat(4, 1)

    def __len__(self):
        return self.similarities.shape[0]

    def __getitem__(self, idx):
        return self.similarities[idx], self.labels[idx], None, self.dopplers[idx]


if __name__ == "__main__":
    dataset_root = '../../data/ASTom/processed/'

    # # ################### ASTOMSimDataset #########################
    sim_dataset = ASTomSimDataset(dataset_root=dataset_root, split="train")
    print(sim_dataset.dopplers.shape)