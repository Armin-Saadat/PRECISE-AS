import torch
import torch.nn as nn

from afa.classifiers import ProtoASNetClassifier
from afa.datasets import ASTomSimDataset
from afa.utils import mask_as_study

ACTION_2_VIDEO_IDX = {0: [], 1: [0], 2: [1], 3: [2], 4: [3]}
LABELS_STR_2_INT = {'No AS': 0, 'Early AS': 1, 'Significant AS': 2}


class ProtoASSimMax(nn.Module):
    def __init__(self, n_videos=4, logger=None):
        super().__init__()

        self.protoAS_model = ProtoASNetClassifier()
        self.n_videos = n_videos

    def forward(self, studies):
        """
        studies: (B, N_videos, N_similarities)
        """
        max_similarity, _ = torch.max(studies, dim=1)  # shape: [1, N_similarities]

        # logits shape: [1, 4], but the last one is aleatoric uncertainty
        logits = self.protoAS_model.last_layer(max_similarity)
        _, y_pred = torch.max(logits[:, :-1], dim=1)

        return y_pred

    def predict(self, state: torch.tensor, action_history: list):
        """
            state: torch.tensor, shape: [N_videos, N_similarities]
            action_history: list
        """
        assert 0 not in action_history, f"0 is a termination action, so the process should have been terminated."

        # predict 0 when no data is acquired
        if not action_history:
            return 0

        observed_videos_idx = set([idx for a in action_history for idx in ACTION_2_VIDEO_IDX[a]])
        masked_videos_idx = list(set(range(self.n_videos)) - observed_videos_idx)

        with torch.no_grad():
            _, _, masked_studies = mask_as_study(state.unsqueeze(0), mask_prob=0.0, video_idx=masked_videos_idx)
            y_pred = self.forward(studies=masked_studies)

        return y_pred.squeeze(0).item()

    def validate(self, data: ASTomSimDataset, phase: str = 'validation'):
        pass


if __name__ == '__main__':
    model = ProtoASSimMax(n_videos=4, )

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_nontrainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Trainable parameters:", total_trainable_params)
    print("Non-trainable parameters:", total_nontrainable_params)
