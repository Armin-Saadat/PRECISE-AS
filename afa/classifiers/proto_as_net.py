import torch
import torch.nn as nn

ACTION_2_VIDEO_IDX = {0: [], 1: [0], 2: [1], 3: [2], 4: [3]}
LABELS_STR_2_INT = {'No AS': 0, 'Early AS': 1, 'Significant AS': 2}


class ProtoASNetClassifier(nn.Module):
    def __init__(self, num_prototypes=40, num_classes=4, logger=None, **kwargs):
        super(ProtoASNetClassifier, self).__init__()

        self.num_prototypes = num_prototypes
        self.num_classes = num_classes

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)  # do not use bias
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, 256), requires_grad=False)

        self.cosine_similarity = nn.CosineSimilarity(dim=2)

        self.logger = logger

    def forward(self, features_extracted):
        similarity = self.cosine_similarity(
            features_extracted, self.prototypes.squeeze().unsqueeze(0)
        )  # shape (N, P)
        similarity = (similarity + 1) / 2.0  # normalizing to [0,1] for positive reasoning

        # logits = self.last_layer(similarity)

        return similarity

    def predict(self, state: torch.tensor, action_history: list):
        """
            state: torch.tensor, shape: [N_videos, N_similarities]
            action_history: list
        """

        assert 0 not in action_history, f"0 is a termination action, so the process should have been terminated."

        # predict 0 when no data is acquired
        if not action_history:
            return 0
        actions_sorted = tuple(sorted(set(action_history)))

        with torch.no_grad():
            max_similarity = torch.zeros((1, self.num_prototypes))
            for a in actions_sorted:
                # forward input shape: [1, 40, 256], similarity shape: [1, 40]
                similarity = self.forward(state[ACTION_2_VIDEO_IDX[a]])
                max_similarity = torch.max(max_similarity, similarity)

            # logits shape: [1, 4], but the last one is aleatoric uncertainty
            logits = self.last_layer(max_similarity)
            _, y_pred = torch.max(logits[:, :-1], dim=1)

        return y_pred.squeeze(0).item()


if __name__ == '__main__':
    import os

    model = ProtoASNetClassifier()
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'checkpoints/proto_as_net.pth')))
    for name, _ in model.named_parameters():
        print(name)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_nontrainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Trainable parameters:", total_trainable_params)
    print("Non-trainable parameters:", total_nontrainable_params)
