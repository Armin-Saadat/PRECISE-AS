import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from afa.utils import mask_as_study

ACTION_2_VIDEO_IDX = {0: [], 1: [0], 2: [1], 3: [2], 4: [3]}
LABELS_STR_2_INT = {'No AS': 0, 'Early AS': 1, 'Significant AS': 2}


class ProtoASSimFormer(nn.Module):
    def __init__(self, *, n_videos=4, n_sims=40, n_classes=3, d_model=64, n_layers=6, n_heads=8, d_mlp=256, d_head=8,
                 pool='cls', dropout=0.1, emb_dropout=0.1, logger=None):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.n_videos = n_videos

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(n_sims),
            nn.Linear(n_sims, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, n_videos + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(d_model, n_layers, n_heads, d_head, d_mlp, dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, studies, mask_idx=None):
        """
        studies: (B, N_videos, N_similarities)
        mask_idx: Optional[Tensor] of shape (B, N_videos),
                    where 1 => video is masked out, 0 => video is not masked out.
                    If None, no masking is applied.
        """
        x = self.to_patch_embedding(studies)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        #    patch_mask: shape (B, N), we want shape (B, N+1) after prepending CLS
        if mask_idx is not None:
            # Prepend a zero column for the CLS token (so CLS is not masked)
            mask_idx = F.pad(mask_idx, (1, 0), value=0)  # shape => (B, N+1)
            # We want a 2D attention mask: (N+1) x (N+1) for each batch
            # If row i is masked => entire row = True.
            # If row i is unmasked => only unmasked columns => ~col_mask_bool.
            row_mask = mask_idx.unsqueeze(-1).bool()  # (B, N+1, 1)
            col_mask = mask_idx.unsqueeze(-2).bool()  # (B, 1, N+1)
            attn_mask_2d = (row_mask | (~col_mask)).float()  # (B, N+1, N+1)
            # Finally, expand a "heads" dimension or keep it as (B, 1, N+1, N+1) so it can broadcast:
            attn_mask_2d = attn_mask_2d.unsqueeze(1)  # (B, 1, N+1, N+1)
        else:
            attn_mask_2d = None

        x, attn = self.transformer(x, attn_mask=attn_mask_2d)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        pred_label = self.mlp_head(x)

        return pred_label, attn

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
            mask_idx, _, masked_studies = mask_as_study(state.unsqueeze(0), mask_prob=0.0, video_idx=masked_videos_idx)
            pred_labels, attn = self.forward(studies=masked_studies, mask_idx=mask_idx)
            _, y_pred = torch.max(pred_labels, dim=1)

        return y_pred.squeeze(0).item()



############### Code for General Transformer ###################
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, attn_mask=None, **kwargs):
        return self.fn(self.norm(x), attn_mask=attn_mask, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, dropout=0.):
        super().__init__()
        inner_dim = d_head * n_heads
        project_out = not (n_heads == 1 and d_head == d_model)

        self.n_heads = n_heads
        self.scale = d_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(d_model, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, d_model),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, attn_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if attn_mask is not None:
            # attn_mask shape is expected to be (B, 1, N, N) or broadcastable to (B, h, N, N)
            dots = dots.masked_fill(attn_mask == 0, float('-inf'))

        attn = self.attend(dots)
        out = torch.matmul(self.dropout(attn), v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class Transformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_head, d_mlp, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(d_model, Attention(d_model, n_heads, d_head, dropout=dropout)),
                PreNorm(d_model, FeedForward(d_model, d_mlp, dropout=dropout))
            ]))

    def forward(self, x, attn_mask=None):
        for attn, ff in self.layers:
            attn_out, attn_w = attn(x, attn_mask=attn_mask)
            x = attn_out + x
            x = ff(x) + x
        return x, attn_w


##################################################################
if __name__ == '__main__':
    model = ProtoASSimFormer(
        n_videos=4,
        n_sims=40,
        n_classes=3,
        d_model=64,
        n_layers=6,
        n_heads=8,
        d_mlp=256,
        d_head=8,
        pool='cls',
        dropout=0.1,
        emb_dropout=0.1)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_nontrainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Trainable parameters:", total_trainable_params)
    print("Non-trainable parameters:", total_nontrainable_params)
