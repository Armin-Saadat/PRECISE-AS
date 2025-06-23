import os
import pandas as pd
import time
from colorlog import ColoredFormatter
import logging
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, \
    balanced_accuracy_score, mean_absolute_error
import torch
import numpy as np
import random
from einops import rearrange


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For extra determinism (sometimes at a performance cost):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def powerset(lst):
    n = len(lst)
    for i in range(2 ** n):
        subset = [lst[j] for j in range(n) if (i & (1 << j)) > 0]
        if not subset:
            continue
        yield tuple(subset)


def get_classification_metrics(gt: list, pred: list):
    bACC = balanced_accuracy_score(gt, pred)
    F1 = f1_score(gt, pred, average='weighted')
    MAE = mean_absolute_error(gt, pred)
    ACC = accuracy_score(gt, pred)
    # recall = recall_score(gt, pred, average=None)
    # precision = precision_score(gt, pred, average=None)
    conf_matrix = confusion_matrix(gt, pred)

    return {'bACC': bACC, 'F1': F1, 'MAE': MAE, 'ACC': ACC, 'Confusion Matrix': conf_matrix}


def plot_confusion_matrix(conf_matrix, dataset_name, save_path=None):
    plt.figure(figsize=(8, 8))
    if dataset_name == "UCI":
        labels = ['normal', 'CAD']
    elif dataset_name in ['AS-Tom', 'AS-Tom-Sim']:
        labels = ['no AS', 'early AS', 'significant AS']
    else:
        raise Exception("Unsupported dataset name.")

    y_labels = labels.copy()
    y_labels.reverse()
    reordered_conf_matrix = np.flipud(conf_matrix)

    sns.heatmap(reordered_conf_matrix, annot=True, annot_kws={"size": 28}, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=y_labels)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xlabel('Predicted Labels', fontsize=17, labelpad=16)
    plt.ylabel('True Labels', fontsize=17, labelpad=16)
    # plt.title('Train Set', fontsize=20, pad=24)
    plt.savefig(save_path) if save_path else plt.show()


def read_uci_data(dataset_path: str) -> pd.DataFrame:
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
            'thal', 'label']
    df1 = pd.read_csv(os.path.join(dataset_path, 'processed.cleveland.data'), header=None, na_values='?', names=cols)
    df1['center'] = 0
    df2 = pd.read_csv(os.path.join(dataset_path, 'processed.va.data'), header=None, na_values='?', names=cols)
    df2['center'] = 1
    df3 = pd.read_csv(os.path.join(dataset_path, 'processed.switzerland.data'), header=None, na_values='?', names=cols)
    df3['center'] = 2
    df4 = pd.read_csv(os.path.join(dataset_path, 'processed.hungarian.data'), header=None, na_values='?', names=cols)
    df4['center'] = 3
    df = pd.concat([df1, df2, df3, df4]).dropna()
    df['label'] = df['label'].values >= 1

    return df


def create_logger(name: str) -> logging.Logger:
    """
    Creates a custom logger

    :param name: str, name for the logger
    :return: A custom logging.Logger object
    """

    # Define a new level for the logger
    def _info_important(self, msg, *args, **kwargs):
        self.log(logging.INFO + 1, msg, *args, **kwargs)

    formatter = ColoredFormatter(
        "%(log_color)s[%(name)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white,bold',
            'INFOIMPORTANT': 'cyan,bold',
            'WARNING': 'yellow',
            'ERROR': 'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = [logging.FileHandler(os.path.join(name, 'output.log')), ch]
    logger.propagate = False

    logging.addLevelName(logging.INFO + 1, 'INFOIMPORTANT')
    logging.Logger.info_important = _info_important

    return logger


def get_action_demography(file_path, dataset_name):
    episodes_list = {}
    episodes_set = {}
    y_preds, y_trues = [], []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            y_pred = int(line.split('- ')[5].replace(" ", "").removeprefix('y_pred'))
            y_true = int(line.split('- ')[6].replace(" ", "").removeprefix('y_true'))
            result = 'correct' if y_pred == y_true else 'incorrect'
            episode = tuple(ast.literal_eval(line.split('- ')[4].replace(" ", "").removeprefix('Actions')))

            y_preds.append(y_pred)
            y_trues.append(y_true)

            # order of actions matter
            if episode in episodes_list:
                episodes_list[episode][result] += 1
            else:
                episodes_list[episode] = {'correct': 1, 'incorrect': 0} if result == 'correct' else {
                    'correct': 0, 'incorrect': 1}

            # order of actions does not matter
            episode = tuple(set(episode))
            if episode in episodes_set:
                episodes_set[episode][result] += 1
            else:
                episodes_set[episode] = {'correct': 1, 'incorrect': 0} if result == 'correct' else {'correct': 0,
                                                                                                    'incorrect': 1}
        print('Number of patients:', i + 1, end="\n\n")
        print('Considering order of actions')
        for key, value in episodes_list.items():
            print(key, "\t", value)
        print('------------------------------------------', end="\n\n")
        print('Not considering order of actions')
        for key, value in episodes_set.items():
            print(key, "\t", value)

        classification_metrics = get_classification_metrics(y_trues, y_preds)
        for k, v in classification_metrics.items():
            print(f'{k}: {v}')
        # plot_confusion_matrix(classification_metrics['Confusion Matrix'], dataset_name=dataset_name)
        # plot_confusion_matrix(classification_metrics['Confusion Matrix'],
        #                       dataset_name=dataset_name,
        #                       save_path='C:/Users/arminsdt.stu/Desktop/conf_matrix.pdf')


def mask_patches(images, patch_size, mask_prob=0.5, patch_idx=None):
    """
    images: (B, C, H, W)
    patch_size: int, height/width of each patch
    mask_prob: float, probability of masking each patch
    patch_idx: list, containing the indices of the patches to mask
    returns:
      - patch_mask: (B, N) with 1 => patch is masked out, 0 => not masked
      - masked_images: (B, 1, H, W) grayscale, with masked patches zeroed out

    """

    B, C, H, W = images.shape
    # Number of patches horizontally and vertically
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    N = num_patches_h * num_patches_w  # total patches per image

    # Create a random [B, N] binary mask
    if patch_idx is not None:
        patch_mask = torch.zeros((B, N))
        patch_mask[:, patch_idx] = 1
    else:
        patch_mask = torch.bernoulli(torch.full((B, N), mask_prob))

    # Reshape (patchify) from [B, 1, H, W] -> [B, N, 1, patch_size, patch_size]
    patches = rearrange(
        images,
        'b c (ph p1) (pw p2) -> b (ph pw) c p1 p2',
        p1=patch_size, p2=patch_size
    )

    # Zero out the patches where patch_mask == 1
    #    patch_mask=1 => masked out => patch becomes 0
    #    patch_mask=0 => keep original patch
    mask_expanded = patch_mask.view(B, N, 1, 1, 1)  # shape: (B, N, 1, 1, 1)

    pixel_mask = torch.ones_like(patches)
    patches = patches * (1 - mask_expanded)  # sets to zero where mask=1
    pixel_mask = 1 - (pixel_mask * (1 - mask_expanded))

    # Rearrange patches back to [B, 1, H, W]
    masked_images = rearrange(
        patches,
        'b (ph pw) c p1 p2 -> b c (ph p1) (pw p2)',
        ph=num_patches_h,
        pw=num_patches_w
    )

    pixel_mask = rearrange(
        pixel_mask,
        'b (ph pw) c p1 p2 -> b c (ph p1) (pw p2)',
        ph=num_patches_h,
        pw=num_patches_w
    )

    return patch_mask, pixel_mask, masked_images


def mask_as_study(studies, mask_prob=0.5, video_idx=None):
    """
    studies: (B, N_videos, N_similarities)
    mask_prob: float, probability of masking each video
    video_idx: list, containing the indices of the videos to mask
    returns:
      - mask_idx: (B, N_videos) with 1 => video is masked out, 0 => not masked
      - binary_mask: (B, N_videos, N_similarities), masked values are set to 1, unmasked values set to 0.
      - masked_studies: (B, N_videos, N_similarities), with masked videos zeroed out, others keep the original value
    """

    B, N_vid, N_sim = studies.shape
    device = studies.device

    # Create a random [B, N] binary mask
    if video_idx is not None:
        mask_idx = torch.zeros((B, N_vid)).to(device)
        mask_idx[:, video_idx] = 1
    else:
        mask_idx = torch.bernoulli(torch.full((B, N_vid), mask_prob)).to(device)

    # ToDo: attend to this later
    # studies with plax or psax count less than 2 have zero embeddings, which would create similarity tensors of 0.5, and should be masked.
    mask_idx = (mask_idx.bool() | (studies == 0.5).all(dim=2).bool()).float()

    # Zero out the videos where mask == 1
    #    mask=1 => masked out => video becomes 0
    #    mask=0 => keep original value
    mask_expanded = mask_idx.view(B, N_vid, 1)  # shape: (B, N, 1)
    masked_studies = studies.clone() * (1 - mask_expanded)

    binary_mask = 1 - (torch.ones_like(studies).to(device) * (1 - mask_expanded))

    return mask_idx, binary_mask, masked_studies


class Profiler:
    def __init__(self, name: str = None, logger: logging.Logger = None):
        self.name = name
        self.logger = logger
        self.times = []
        self.record = False

    def __enter__(self):
        self.record = True
        self.start = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.record:
            self.times.append(time.time() - self.start)

        return False

    def skip(self):
        self.record = False

    def reset(self):
        self.times = []

    def log(self):
        times_sum = np.sum(self.times)
        times_mean = times_sum / len(self.times) if len(self.times) > 0 else 0
        self.logger.warning(
            f'{self.name}: Avg Time {times_mean:.4f} - N_Calls {len(self.times)} - Total Time {times_sum:.1f}')
