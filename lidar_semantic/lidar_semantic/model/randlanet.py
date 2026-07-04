"""
RandLA-Net — PyTorch implementation matching the semantickitti_tsunghan checkpoint.

Architecture verified against checkpoint key names and tensor shapes.

Encoder: 4 DilatedResBlocks, d_out=[32,128,256,512], block-then-downsample.
Decoder: decoder_0 + 4 decoder_blocks with NN upsampling.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import KDTree


# ── Offline preprocessing helpers ─────────────────────────────────────────────

def knn_query(points: np.ndarray, k: int) -> np.ndarray:
    """K nearest neighbours (self excluded). Returns (N, k) int64."""
    tree = KDTree(points)
    _, idx = tree.query(points, k=k + 1)
    return idx[:, 1:].astype(np.int64)


def random_downsample(points: np.ndarray, ratio: int):
    """Random 1/ratio subset. Returns (sub_points, kept_idx)."""
    N = len(points)
    N_sub = max(1, N // ratio)
    kept = np.random.choice(N, N_sub, replace=False)
    return points[kept], kept.astype(np.int64)


# ── Building blocks ───────────────────────────────────────────────────────────

class _BN(nn.Module):
    """BN wrapper matching checkpoint key structure: self.bn = BatchNorm1d."""
    def __init__(self, d: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x):
        return self.bn(x)


class SharedMLP(nn.Module):
    """Conv1d(k=1) + optional _BN + optional LeakyReLU.
    Checkpoint key structure: self.conv, self.bn (a _BN instance).
    """
    def __init__(self, in_ch: int, out_ch: int, bn: bool = True, activation: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1, bias=not bn)
        self.bn   = _BN(out_ch) if bn else None
        self.act  = nn.LeakyReLU(0.2) if activation else None

    def forward(self, x):          # x: [B, C, N]
        x = self.conv(x)
        if self.bn  is not None: x = self.bn(x)
        if self.act is not None: x = self.act(x)
        return x


class AttPooling(nn.Module):
    """Attentive pooling over K neighbours.

    Checkpoint keys:
        fc  — Conv1d(in_ch, in_ch, 1, bias=False)   [no BN, no act]
        mlp — SharedMLP(in_ch, out_ch)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.fc  = nn.Conv1d(in_ch, in_ch, 1, bias=False)
        self.mlp = SharedMLP(in_ch, out_ch)

    def forward(self, x: torch.Tensor, N: int, K: int) -> torch.Tensor:
        """
        x: [B, in_ch, N*K]  (neighbours flattened into the last dim)
        Returns: [B, out_ch, N]
        """
        B, C, NK = x.shape
        scores = self.fc(x).reshape(B, C, N, K)    # [B, C, N, K]
        scores = torch.softmax(scores, dim=-1)

        x_nk   = x.reshape(B, C, N, K)
        agg    = (x_nk * scores).sum(dim=-1)        # [B, C, N]
        return self.mlp(agg)                        # [B, out_ch, N]


class _LFA(nn.Module):
    """Local Feature Aggregation (two rounds of LSE + attentive pooling).

    Checkpoint keys live under 'lfa.*':
        mlp1, att_pooling_1, mlp2, att_pooling_2
    """
    def __init__(self, d_feat: int, d_out: int, K: int):
        super().__init__()
        self._K    = K
        d4         = d_out // 4   # d_out//4
        d2         = d_out // 2   # d_out//2 = 2*d4

        self.mlp1          = SharedMLP(10, d4)           # LSE (geo→d4), reused for both rounds
        self.att_pooling_1 = AttPooling(d2, d4)          # round-1 pool: in=d4+d4, out=d4

        self.mlp2          = SharedMLP(d4, d4)           # per-point feature MLP on att1_out
        self.att_pooling_2 = AttPooling(d2, d2)          # round-2 pool: in=d4+d4, out=d2

    @staticmethod
    def _geo_desc(pts: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        """Build 10-dim relative geometry descriptor for each (point, neighbour) pair.

        pts:     [B, N, 3]
        knn_idx: [B, N, K]
        Returns: [B, 10, N*K]
        """
        B, N, K = knn_idx.shape
        # Gather neighbour positions
        flat = knn_idx.reshape(B, -1)                               # [B, N*K]
        p_j  = pts.gather(1, flat.unsqueeze(-1).expand(-1, -1, 3)) # [B, N*K, 3]
        p_i  = pts.unsqueeze(2).expand(-1, -1, K, -1)              # [B, N, K, 3]
        p_i  = p_i.reshape(B, N * K, 3)
        diff = p_i - p_j                                            # [B, N*K, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True)               # [B, N*K, 1]
        desc = torch.cat([p_i, p_j, diff, dist], dim=-1)           # [B, N*K, 10]
        return desc.permute(0, 2, 1)                                # [B, 10, N*K]

    def forward(self, pts: torch.Tensor, x1: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        """
        pts:     [B, N, 3]
        x1:      [B, d_feat, N]   (output of the block's mlp1)
        knn_idx: [B, N, K]
        Returns: [B, d_out//2, N]
        """
        B, _, N = x1.shape
        K = knn_idx.shape[-1]

        geo = self._geo_desc(pts, knn_idx)      # [B, 10, N*K]

        # Shared LSE encoding (used for both rounds)
        lse = self.mlp1(geo)                        # [B, d4, N*K]

        # Round 1: concat(lse, x1_expanded_over_K) → att_pool → d4
        x1_exp  = x1.unsqueeze(-1).expand(-1, -1, -1, K).reshape(B, -1, N * K)
        att1_in  = torch.cat([lse, x1_exp], dim=1)   # [B, d2, N*K]
        att1_out = self.att_pooling_1(att1_in, N, K)  # [B, d4, N]

        # Round 2: per-point feature MLP, then concat(lse, x2_expanded_over_K) → att_pool → d2
        x2      = self.mlp2(att1_out)                 # [B, d4, N]
        x2_exp  = x2.unsqueeze(-1).expand(-1, -1, -1, K).reshape(B, -1, N * K)
        att2_in  = torch.cat([lse, x2_exp], dim=1)   # [B, d2, N*K]  (reuse lse)
        att2_out = self.att_pooling_2(att2_in, N, K)  # [B, d2, N]

        return att2_out


class DilatedResBlock(nn.Module):
    """One encoder stage.

    Checkpoint keys:
        mlp1     — pre-projection SharedMLP(d_in → d_out//4)
        lfa      — _LFA submodule
        mlp2     — post-aggregation SharedMLP(d_out//2 → d_out)
        shortcut — SharedMLP(d_in → d_out)
    """
    def __init__(self, d_in: int, d_out: int, K: int):
        super().__init__()
        self.mlp1     = SharedMLP(d_in,     d_out // 4)
        self.lfa      = _LFA(d_in,          d_out, K)
        self.mlp2     = SharedMLP(d_out // 2, d_out)
        self.shortcut = SharedMLP(d_in,     d_out)
        self.act      = nn.LeakyReLU(0.2)

    def forward(self, pts: torch.Tensor, x: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        """
        pts:     [B, N, 3]
        x:       [B, d_in, N]
        knn_idx: [B, N, K]
        Returns: [B, d_out, N]
        """
        x1  = self.mlp1(x)                    # [B, d_out//4, N]
        agg = self.lfa(pts, x1, knn_idx)      # [B, d_out//2, N]
        y   = self.mlp2(agg)                  # [B, d_out,    N]
        sc  = self.shortcut(x)                # [B, d_out,    N]
        return self.act(y + sc)


# ── Full network ──────────────────────────────────────────────────────────────

class RandLANet(nn.Module):
    """
    RandLA-Net matching semantickitti_tsunghan(_1d).tar checkpoint.

    Forward interface (caller must precompute KNN + downsampling):
        pts_stages:    list of 5 tensors [B, N_i, 3], i=0..4
                       N_0=full, N_1=N_0/4, N_2=N_1/4, N_3=N_2/4, N_4=N_3/4
        knn_list:      list of 4 tensors [B, N_i, K]  (KNN at encoder level i)
        down_idx_list: list of 4 tensors [B, N_{i+1}] (kept indices at each step)

    Encoder: block-then-downsample, saves features BEFORE each downsample.
    Decoder: decoder_0 → dec[0]: N4→N2 → dec[1]: N2→N1 → dec[2]: N1→N0 →
             dec[3]: N0→N0 (refinement using enc_feats[0] again).
    """

    def __init__(
        self,
        d_in:          int   = 3,
        num_classes:   int   = 19,
        num_neighbors: int   = 16,
        dropout:       float = 0.5,
        **_ignored,              # swallow legacy kwargs from old interface
    ):
        super().__init__()
        d_out = [32, 128, 256, 512]

        # Input embedding
        self.fc0 = SharedMLP(d_in, 8)

        # Encoder
        enc_in = [8] + d_out
        self.dilated_res_blocks = nn.ModuleList([
            DilatedResBlock(enc_in[i], d_out[i], num_neighbors)
            for i in range(4)
        ])

        # Bottleneck decoder
        self.decoder_0 = SharedMLP(512, 512)

        # Decoder blocks (concat upsampled + skip → output)
        self.decoder_blocks = nn.ModuleList([
            SharedMLP(512 + 256, 256),   # [0]: N4→N2, skip enc[2]=256
            SharedMLP(256 + 128, 128),   # [1]: N2→N1, skip enc[1]=128
            SharedMLP(128 + 32,  32),    # [2]: N1→N0, skip enc[0]=32
            SharedMLP(32  + 32,  32),    # [3]: N0→N0, skip enc[0]=32 again
        ])

        # Classification head
        self.fc1 = SharedMLP(32, 64)
        self.fc2 = SharedMLP(64, 32)
        self.dp  = nn.Dropout(p=dropout)
        self.fc3 = SharedMLP(32, num_classes, bn=False, activation=False)
        # fc3 needs bias: override conv bias
        self.fc3.conv = nn.Conv1d(32, num_classes, 1, bias=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _nn_upsample(x: torch.Tensor,
                     src_pts: torch.Tensor,
                     tgt_pts: torch.Tensor) -> torch.Tensor:
        """
        Nearest-neighbour feature propagation from src to tgt resolution.

        x:       [B, C, N_src]
        src_pts: [B, N_src, 3]
        tgt_pts: [B, N_tgt, 3]
        Returns: [B, C, N_tgt]
        """
        dists  = torch.cdist(tgt_pts, src_pts)   # [B, N_tgt, N_src]
        nn_idx = dists.argmin(dim=-1)            # [B, N_tgt]
        return x.gather(2, nn_idx.unsqueeze(1).expand(-1, x.shape[1], -1))

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, pts_stages, knn_list, down_idx_list):
        """
        pts_stages:    list [B,N_i,3]  i=0..4
        knn_list:      list [B,N_i,K]  i=0..3
        down_idx_list: list [B,N_{i+1}] i=0..3
        Returns: [B, num_classes, N_0]
        """
        # fc0
        x = self.fc0(pts_stages[0].permute(0, 2, 1))   # [B, 8, N_0]

        # Encoder: block → save → downsample
        enc_feats = []
        for i in range(4):
            x = self.dilated_res_blocks[i](pts_stages[i], x, knn_list[i])
            enc_feats.append(x)                          # [B, d_out[i], N_i]
            kept = down_idx_list[i]                      # [B, N_{i+1}]
            x = x.gather(2, kept.unsqueeze(1).expand(-1, x.shape[1], -1))

        # x = [B, 512, N_4] (bottleneck)

        # Bottleneck decoder
        x = self.decoder_0(x)                           # [B, 512, N_4]

        # dec[0]: N_4 → N_2 (skip enc_feats[2] @ N_2)
        x_up = self._nn_upsample(x, pts_stages[4], pts_stages[2])
        x = self.decoder_blocks[0](torch.cat([x_up, enc_feats[2]], dim=1))

        # dec[1]: N_2 → N_1 (skip enc_feats[1] @ N_1)
        x_up = self._nn_upsample(x, pts_stages[2], pts_stages[1])
        x = self.decoder_blocks[1](torch.cat([x_up, enc_feats[1]], dim=1))

        # dec[2]: N_1 → N_0 (skip enc_feats[0] @ N_0)
        x_up = self._nn_upsample(x, pts_stages[1], pts_stages[0])
        x = self.decoder_blocks[2](torch.cat([x_up, enc_feats[0]], dim=1))

        # dec[3]: N_0 → N_0, refinement (skip enc_feats[0] again)
        x = self.decoder_blocks[3](torch.cat([x, enc_feats[0]], dim=1))

        # Classification head
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dp(x)
        x = self.fc3(x)
        return x                                        # [B, num_classes, N_0]

    # ── Weight loading ────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = 'cpu', **kwargs):
        model = cls(**kwargs)
        ckpt  = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f'[RandLANet] Missing keys ({len(missing)}): {missing[:5]}')
        if unexpected:
            print(f'[RandLANet] Unexpected keys ({len(unexpected)}): {unexpected[:5]}')
        return model.to(device).eval()
