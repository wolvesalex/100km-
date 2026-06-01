# gnn_imager_models.py
# -*- coding: utf-8 -*-
"""
Models for D-region imaging.

(2026-01, updated)
核心增强点：
- Path obs encoder: multi-scale conv + Transformer, keep point-wise features
- Observation -> grid "splat" using precomputed bilinear idx/weights
- 显式两阶段多尺度反演（coarse->fine），每阶段都重新融合观测
- UNet backbone 支持自动 pad 到 2^depth 的倍数并 crop 回来（解决 H/W 不整齐导致的上采样对齐问题）
- 输出 obs_coverage (wsum map) 供 loss 做 near-path 加权

(本次额外修正)
- coarse 阶段对 obs_wsum 的池化由 avg_pool 改为 sum pooling（用 avg_pool * k*k 实现），
  避免 coverage 强度在 coarse 网格被“稀释”，让 log1p(wsum) 在 coarse 阶段仍有效。
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def bilinear_splat(
    point_feats: torch.Tensor,     # (Npts, C)
    idx4: torch.Tensor,            # (Npts, 4) int64 flatten indices
    w4: torch.Tensor,              # (Npts, 4) float
    valid: torch.Tensor,           # (Npts,) uint8/bool
    H: int,
    W: int,
    eps: float = 1e-6,
    return_wsum: bool = True,
):
    """
    Splat point features to grid using bilinear weights.
    Return:
      - grid_feats: (C, H, W)
      - wsum_map:  (H, W)   (optional)
    """
    assert point_feats.dim() == 2
    Npts, C = point_feats.shape
    device = point_feats.device

    idx4 = idx4.to(device=device, dtype=torch.long)
    w4 = w4.to(device=device, dtype=point_feats.dtype)
    valid = valid.to(device=device)
    if valid.dtype != torch.bool:
        valid = valid > 0

    flatN = H * W
    grid_flat = torch.zeros((flatN, C), device=device, dtype=point_feats.dtype)
    wsum = torch.zeros((flatN, 1), device=device, dtype=point_feats.dtype)

    v = valid.unsqueeze(-1).to(point_feats.dtype)  # (N,1)

    for k in range(4):
        ik = idx4[:, k]  # (N,)
        wk = (w4[:, k:k+1] * v)  # (N,1)
        grid_flat.index_add_(0, ik, point_feats * wk)
        wsum.index_add_(0, ik, wk)

    grid_flat = grid_flat / (wsum + eps)
    grid = grid_flat.view(H, W, C).permute(2, 0, 1).contiguous()  # (C,H,W)

    if not return_wsum:
        return grid

    wsum_map = wsum.view(H, W).contiguous()  # (H,W)
    return grid, wsum_map


def tv_loss_2d(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Total variation loss for (B,C,H,W) or (C,H,W)
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    assert x.dim() == 4
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        my = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        dx = dx * mx
        dy = dy * my
    return 0.5 * (dx.mean() + dy.mean())


def _pad_to_multiple_2d(x: torch.Tensor, multiple_h: int, multiple_w: int):
    """
    Pad (B,C,H,W) to H%multiple_h==0 and W%multiple_w==0 with zeros.
    Return padded_x, (pad_left, pad_right, pad_top, pad_bottom)
    """
    assert x.dim() == 4
    B, C, H, W = x.shape
    target_h = int(math.ceil(H / multiple_h) * multiple_h)
    target_w = int(math.ceil(W / multiple_w) * multiple_w)
    pad_h = target_h - H
    pad_w = target_w - W
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    xp = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], mode="constant", value=0.0)
    return xp, (pad_left, pad_right, pad_top, pad_bottom)


def _crop_like(x: torch.Tensor, ref_h: int, ref_w: int, pads):
    """Inverse of _pad_to_multiple_2d: crop back to (ref_h, ref_w)."""
    pad_left, pad_right, pad_top, pad_bottom = pads
    if pad_left == pad_right == pad_top == pad_bottom == 0:
        return x
    return x[:, :, pad_top:pad_top + ref_h, pad_left:pad_left + ref_w].contiguous()


# -------------------------------------------------------------------
# Path observation encoder (stronger, keeps point-wise features)
# -------------------------------------------------------------------

class MultiScaleConv1D(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dw3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.dw5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.dw9 = nn.Conv1d(d_model, d_model, kernel_size=9, padding=4, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Npath,L,d)
        """
        xt = x.transpose(1, 2)  # (Npath,d,L)
        y = (F.gelu(self.dw3(xt)) + F.gelu(self.dw5(xt)) + F.gelu(self.dw9(xt))) / 3.0
        y = self.pw(y).transpose(1, 2)  # (Npath,L,d)
        x = x + self.dropout(y)
        x = self.norm(x)
        return x


class PathObsEncoderV2(nn.Module):
    """
    输入: path_seq_feats (Npath,L,Cin), path_static (Npath,Cs)
    输出:
      - point_emb: (Npath,L,Cp) 逐点特征（用于投影到网格）
      - path_emb:  (Npath,Cg)   路径全局特征（可选）
    """
    def __init__(
        self,
        seq_in_ch: int,
        static_dim: int,
        d_model: int = 96,
        point_dim: int = 64,
        nhead: int = 6,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.in_proj = nn.Linear(seq_in_ch, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        self.msconv = MultiScaleConv1D(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.film = nn.Linear(d_model, d_model * 2)

        self.point_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, point_dim),
        )

        self.path_pool = nn.AdaptiveAvgPool1d(1)
        self.path_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, seq: torch.Tensor, path_static: torch.Tensor):
        Np, L, _ = seq.shape
        x = self.in_proj(seq)  # (Np,L,d)

        if L <= self.pos.size(1):
            x = x + self.pos[:, :L, :]
        else:
            pos = F.interpolate(self.pos.transpose(1, 2), size=L, mode="linear", align_corners=False).transpose(1, 2)
            x = x + pos

        x = self.msconv(x)
        x = self.encoder(x)  # (Np,L,d)

        s = self.static_proj(path_static)  # (Np,d)
        gamma_beta = self.film(s)          # (Np,2d)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)         # (Np,1,d)
        beta = beta.unsqueeze(1)
        x = x * (1.0 + torch.tanh(gamma)) + beta

        point_emb = self.point_out(x)      # (Np,L,point_dim)

        x_pool = self.path_pool(x.transpose(1, 2)).squeeze(-1)  # (Np,d)
        path_emb = self.path_out(x_pool)
        return point_emb, path_emb


# -------------------------------------------------------------------
# UNet backbone (2D, regular grid) with robust upsampling alignment
# -------------------------------------------------------------------

class ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Down2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = ConvBlock2D(in_ch, out_ch, dropout=dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up2D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock2D(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)

        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)

        if dh > 0 or dw > 0:
            x = F.pad(x, [max(0, dw // 2), max(0, dw - dw // 2), max(0, dh // 2), max(0, dh - dh // 2)])

        if dh < 0 or dw < 0:
            h0 = (x.size(2) - skip.size(2)) // 2
            w0 = (x.size(3) - skip.size(3)) // 2
            x = x[:, :, h0:h0 + skip.size(2), w0:w0 + skip.size(3)].contiguous()

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetBackbone2D(nn.Module):
    def __init__(self, in_ch: int, base: int = 64, dropout: float = 0.1, depth: int = 3):
        super().__init__()
        assert depth in (2, 3), "This implementation supports depth=2 or 3."
        self.depth = int(depth)
        self.pad_multiple = 2 ** self.depth

        self.inc = ConvBlock2D(in_ch, base, dropout=dropout)
        self.down1 = Down2D(base, base * 2, dropout=dropout)
        self.down2 = Down2D(base * 2, base * 4, dropout=dropout)

        if self.depth == 3:
            self.down3 = Down2D(base * 4, base * 8, dropout=dropout)
            self.bot = ConvBlock2D(base * 8, base * 16, dropout=dropout)

            self.up2 = Up2D(base * 16, base * 4, base * 4, dropout=dropout)  # to H/4, skip x2
            self.up1 = Up2D(base * 4, base * 2, base * 2, dropout=dropout)   # to H/2, skip x1
            self.up0 = Up2D(base * 2, base, base, dropout=dropout)           # to H,   skip x0
        else:
            self.bot = ConvBlock2D(base * 4, base * 8, dropout=dropout)
            self.up1 = Up2D(base * 8, base * 2, base * 2, dropout=dropout)   # to H/2, skip x1
            self.up0 = Up2D(base * 2, base, base, dropout=dropout)           # to H,   skip x0

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        xp, pads = _pad_to_multiple_2d(x, self.pad_multiple, self.pad_multiple)

        x0 = self.inc(xp)
        x1 = self.down1(x0)
        x2 = self.down2(x1)

        if self.depth == 3:
            x3 = self.down3(x2)
            xb = self.bot(x3)

            d2 = self.up2(xb, x2)
            d1 = self.up1(d2, x1)
            d0 = self.up0(d1, x0)
        else:
            xb = self.bot(x2)
            d1 = self.up1(xb, x1)
            d0 = self.up0(d1, x0)
            d2 = x2

        d0 = _crop_like(d0, H, W, pads)
        return {"d0": d0, "d1": d1, "d2": d2}


# -------------------------------------------------------------------
# Main model: explicit coarse->fine + iterative refinement
# -------------------------------------------------------------------

class DRegionImagerUNetRefiner(nn.Module):
    def __init__(
        self,
        path_seq_in_ch: int,
        path_static_dim: int,
        h_mid: float,
        h_half: float,
        b_mid: float,
        b_half: float,
        delta_h_scale: float = 2.0,
        delta_beta_scale: float = 0.03,
        obs_point_dim: int = 64,
        obs_grid_dim: int = 64,
        unet_base: int = 64,
        refine_steps: int = 3,
        coarse_factor: int = 4,
        coarse_steps: int = 1,
        unet_depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.refine_steps = int(refine_steps)
        self.coarse_steps = int(coarse_steps)
        self.total_steps = self.coarse_steps + self.refine_steps
        self.coarse_factor = int(coarse_factor)

        self.register_buffer("h_mid", torch.tensor([h_mid], dtype=torch.float32))
        self.register_buffer("h_half", torch.tensor([h_half], dtype=torch.float32))
        self.register_buffer("b_mid", torch.tensor([b_mid], dtype=torch.float32))
        self.register_buffer("b_half", torch.tensor([b_half], dtype=torch.float32))

        self.h_min = float(h_mid - h_half)
        self.h_max = float(h_mid + h_half)
        self.b_min = float(b_mid - b_half)
        self.b_max = float(b_mid + b_half)

        self.delta_h_scale = float(delta_h_scale)
        self.delta_beta_scale = float(delta_beta_scale)

        self.obs_encoder = PathObsEncoderV2(
            seq_in_ch=path_seq_in_ch,
            static_dim=path_static_dim,
            d_model=96,
            point_dim=obs_point_dim,
            nhead=6,
            num_layers=2,
            dropout=dropout,
            max_len=256,
        )

        self.obs_grid_proj = nn.Sequential(
            nn.Conv2d(obs_point_dim, obs_grid_dim, 1),
            nn.GroupNorm(8, obs_grid_dim),
            nn.GELU(),
        )

        step_ch = 8
        self.step_embed = nn.Embedding(self.total_steps, step_ch)
        self.step_proj = nn.Linear(step_ch, step_ch)

        unet_in_ch = 6 + obs_grid_dim + 2 + step_ch

        self.unet_coarse = UNetBackbone2D(unet_in_ch, base=unet_base, dropout=dropout, depth=unet_depth)
        self.unet_fine = UNetBackbone2D(unet_in_ch, base=unet_base, dropout=dropout, depth=unet_depth)

        self.head_full = nn.Conv2d(unet_base, 2, 1)
        self.head_half = nn.Conv2d(unet_base * 2, 2, 1)
        self.head_quarter = nn.Conv2d(unet_base * 4, 2, 1)

        self.fuse_w = nn.Parameter(torch.ones(2, 3))  # [h, b] x [full, half, quarter]

    def _norm_params(self, hp: torch.Tensor, be: torch.Tensor):
        hp_n = (hp - self.h_mid) / (self.h_half + 1e-6)
        be_n = (be - self.b_mid) / (self.b_half + 1e-6)
        return hp_n, be_n

    def _make_step_map(self, B: int, H: int, W: int, step_id: int, device):
        step_idx = torch.full((B,), int(step_id), device=device, dtype=torch.long)
        step_vec = self.step_embed(step_idx)
        step_vec = self.step_proj(step_vec)
        step_map = step_vec.view(B, -1, 1, 1).expand(B, -1, H, W)
        return step_map

    def _predict_delta(
        self,
        unet: nn.Module,
        hp_curr,
        be_curr,
        hp_prior,
        be_prior,
        grid_xy_norm,
        obs_grid,
        obs_wsum,
        grid_valid_mask,
        step_id: int
    ):
        B, H, W = hp_curr.shape
        hp_curr_n, be_curr_n = self._norm_params(hp_curr, be_curr)
        hp_prior_n, be_prior_n = self._norm_params(hp_prior, be_prior)

        base = torch.stack([hp_curr_n, be_curr_n, hp_prior_n, be_prior_n], dim=1)  # (B,4,H,W)
        step_map = self._make_step_map(B, H, W, step_id, device=hp_curr.device)

        wsum_feat = torch.log1p(obs_wsum)  # (B,1,H,W)
        inp = torch.cat([base, grid_xy_norm, obs_grid, wsum_feat, grid_valid_mask, step_map], dim=1)

        feats = unet(inp)
        d_full = self.head_full(feats["d0"])  # (B,2,H,W)
        d_half = self.head_half(feats["d1"])
        d_qua = self.head_quarter(feats["d2"])

        d_half_u = F.interpolate(d_half, size=(H, W), mode="bilinear", align_corners=False)
        d_qua_u = F.interpolate(d_qua, size=(H, W), mode="bilinear", align_corners=False)

        d_stack = torch.stack([d_full, d_half_u, d_qua_u], dim=2)  # (B,2,3,H,W)

        w = F.softmax(self.fuse_w, dim=-1).view(1, 2, 3, 1, 1)
        d = (d_stack * w).sum(dim=2)  # (B,2,H,W)

        d = torch.tanh(d)
        return d

    def forward(
        self,
        hp_prior: torch.Tensor,
        be_prior: torch.Tensor,
        grid_xy_norm: torch.Tensor,
        grid_valid_mask: torch.Tensor,
        path_seq_feats: torch.Tensor,
        path_static_feats: torch.Tensor,
        path_point_grid_idx: torch.Tensor,
        path_point_grid_w: torch.Tensor,
        path_point_valid: torch.Tensor,
    ):
        device = hp_prior.device

        if hp_prior.dim() == 2:
            hp_prior = hp_prior.unsqueeze(0)
            be_prior = be_prior.unsqueeze(0)
        B, H, W = hp_prior.shape

        if grid_xy_norm.dim() == 3:
            grid_xy_norm = grid_xy_norm.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        if grid_valid_mask.dim() == 2:
            grid_valid_mask = grid_valid_mask.unsqueeze(0).expand(B, -1, -1).contiguous()
        grid_valid_mask = grid_valid_mask.to(device=device).float().unsqueeze(1)  # (B,1,H,W)

        if path_seq_feats.dim() == 3:
            path_seq_feats = path_seq_feats.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        if path_static_feats.dim() == 2:
            path_static_feats = path_static_feats.unsqueeze(0).expand(B, -1, -1).contiguous()

        if path_point_valid.dim() == 2:
            path_point_valid = path_point_valid.unsqueeze(0).expand(B, -1, -1).contiguous()

        if path_point_grid_idx.dim() == 3:
            path_point_grid_idx = path_point_grid_idx.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        if path_point_grid_w.dim() == 3:
            path_point_grid_w = path_point_grid_w.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

        B, P, L, Cin = path_seq_feats.shape
        seq_flat = path_seq_feats.view(B * P, L, Cin)
        stat_flat = path_static_feats.view(B * P, -1)

        point_emb_flat, _ = self.obs_encoder(seq_flat, stat_flat)  # (B*P,L,Cp)
        Cp = point_emb_flat.size(-1)
        point_emb = point_emb_flat.view(B, P, L, Cp)

        obs_grids = []
        obs_wsums = []
        for b in range(B):
            pf = point_emb[b].reshape(P * L, Cp)

            idx4 = path_point_grid_idx[b].reshape(P * L, 4)
            w4 = path_point_grid_w[b].reshape(P * L, 4)
            valid = path_point_valid[b].reshape(P * L)

            g, wsum_map = bilinear_splat(pf, idx4, w4, valid, H=H, W=W, return_wsum=True)
            obs_grids.append(g.unsqueeze(0))
            obs_wsums.append(wsum_map.unsqueeze(0))

        obs_grid = torch.cat(obs_grids, dim=0)  # (B,Cp,H,W)
        obs_wsum = torch.cat(obs_wsums, dim=0).unsqueeze(1)  # (B,1,H,W)

        obs_grid = self.obs_grid_proj(obs_grid)

        # ---- Stage 0: coarse ----
        hp_curr = hp_prior
        be_curr = be_prior

        coarse_up_hp = hp_prior
        coarse_up_be = be_prior

        if self.coarse_factor > 1:
            k = int(self.coarse_factor)

            hp_p_c = F.avg_pool2d(hp_prior.unsqueeze(1), kernel_size=k, stride=k).squeeze(1)
            be_p_c = F.avg_pool2d(be_prior.unsqueeze(1), kernel_size=k, stride=k).squeeze(1)
            hp_c = hp_p_c.clone()
            be_c = be_p_c.clone()

            obs_c = F.avg_pool2d(obs_grid, kernel_size=k, stride=k)

            # ---- FIX: sum pooling for wsum (avg_pool * k*k) ----
            wsum_c = F.avg_pool2d(obs_wsum, kernel_size=k, stride=k) * float(k * k)

            mask_c = F.interpolate(grid_valid_mask, size=hp_c.shape[-2:], mode="nearest")
            xy_c = F.interpolate(grid_xy_norm, size=hp_c.shape[-2:], mode="bilinear", align_corners=False)

            for t in range(self.coarse_steps):
                d = self._predict_delta(
                    self.unet_coarse,
                    hp_c, be_c, hp_p_c, be_p_c,
                    xy_c, obs_c, wsum_c, mask_c,
                    step_id=t
                )
                dh = d[:, 0] * self.delta_h_scale
                db = d[:, 1] * self.delta_beta_scale
                hp_c = torch.clamp(hp_c + dh, min=self.h_min, max=self.h_max)
                be_c = torch.clamp(be_c + db, min=self.b_min, max=self.b_max)

            dh_c = (hp_c - hp_p_c).unsqueeze(1)
            db_c = (be_c - be_p_c).unsqueeze(1)
            dh_u = F.interpolate(dh_c, size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
            db_u = F.interpolate(db_c, size=(H, W), mode="bilinear", align_corners=False).squeeze(1)

            hp_curr = torch.clamp(hp_prior + dh_u, min=self.h_min, max=self.h_max)
            be_curr = torch.clamp(be_prior + db_u, min=self.b_min, max=self.b_max)

            coarse_up_hp = hp_curr
            coarse_up_be = be_curr

        # ---- Stage 1: fine iterative refinement ----
        hp_steps = []
        be_steps = []
        for t in range(self.refine_steps):
            step_id = self.coarse_steps + t
            d = self._predict_delta(
                self.unet_fine,
                hp_curr, be_curr, hp_prior, be_prior,
                grid_xy_norm, obs_grid, obs_wsum, grid_valid_mask,
                step_id=step_id
            )
            dh = d[:, 0] * self.delta_h_scale
            db = d[:, 1] * self.delta_beta_scale
            hp_curr = torch.clamp(hp_curr + dh, min=self.h_min, max=self.h_max)
            be_curr = torch.clamp(be_curr + db, min=self.b_min, max=self.b_max)
            hp_steps.append(hp_curr)
            be_steps.append(be_curr)

        return {
            "hprime_pred": hp_curr,
            "beta_pred": be_curr,
            "hprime_steps": hp_steps,
            "beta_steps": be_steps,
            "hprime_coarse_up": coarse_up_hp,
            "beta_coarse_up": coarse_up_be,
            "obs_wsum": obs_wsum,
        }


# -------------------------------------------------------------------
# 下面保留 EnhancedGNN（不删）
# -------------------------------------------------------------------

def wrap_phase_deg_to_sincos(phase_deg: torch.Tensor) -> torch.Tensor:
    rad = phase_deg * math.pi / 180.0
    return torch.stack([torch.sin(rad), torch.cos(rad)], dim=-1)


class LearnableGeometricAttention(nn.Module):
    def __init__(self, grid_dim: int, path_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.grid_proj = nn.Linear(grid_dim, hidden_dim)
        self.path_proj = nn.Linear(path_dim, hidden_dim)
        total_input_dim = hidden_dim * 2 + 3
        layers = []
        current_dim = total_input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.weight_net = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, grid_feats, grid_coords, path_feats, path_coords, knn_idx=None, eps: float = 1e-8):
        N = grid_feats.size(0)
        P_total = path_feats.size(0)
        grid_proj = self.grid_proj(grid_feats)

        if knn_idx is not None:
            path_feats_knn = path_feats[knn_idx]
            path_coords_knn = path_coords[knn_idx]
            path_proj = self.path_proj(path_feats_knn)
            grid_proj_exp = grid_proj.unsqueeze(1).expand(-1, path_proj.size(1), -1)
            grid_coords_exp = grid_coords.unsqueeze(1).expand(-1, path_coords_knn.size(1), -1)
            rel_coords = grid_coords_exp - path_coords_knn
            distance = torch.norm(rel_coords, dim=-1, keepdim=True)
            combined = torch.cat([grid_proj_exp, path_proj, rel_coords, distance], dim=-1)
            raw = self.weight_net(combined).squeeze(-1)
            w = F.softmax(raw, dim=-1)
            w = self.dropout(w)
            return w

        path_proj = self.path_proj(path_feats)
        grid_proj_exp = grid_proj.unsqueeze(1).expand(-1, P_total, -1)
        path_proj_exp = path_proj.unsqueeze(0).expand(N, -1, -1)
        grid_coords_exp = grid_coords.unsqueeze(1).expand(-1, P_total, -1)
        path_coords_exp = path_coords.unsqueeze(0).expand(N, -1, -1)
        rel_coords = grid_coords_exp - path_coords_exp
        distance = torch.norm(rel_coords, dim=-1, keepdim=True)
        combined = torch.cat([grid_proj_exp, path_proj_exp, rel_coords, distance], dim=-1)
        raw = self.weight_net(combined).squeeze(-1)
        w = F.softmax(raw, dim=-1)
        w = self.dropout(w)
        return w


class MultiScaleGraphConvLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, num_scales: int = 3, scale_factors: list = None):
        super().__init__()
        if scale_factors is None:
            scale_factors = [1.0, 0.5, 0.25]
        self.num_scales = num_scales
        self.scale_factors = scale_factors
        self.lin_self = nn.Linear(dim, dim)
        self.lin_msg_scales = nn.ModuleList([nn.Linear(dim + 1, dim) for _ in range(num_scales)])
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, multi_edge_index, multi_edge_attr=None):
        x_self = self.lin_self(x)
        scale_outputs = []
        for scale_idx in range(self.num_scales):
            edge_index = multi_edge_index[scale_idx]
            src, dst = edge_index[0], edge_index[1]
            if multi_edge_attr is not None and multi_edge_attr[scale_idx] is not None:
                edge_attr = multi_edge_attr[scale_idx]
                edge_attr_scaled = edge_attr * self.scale_factors[scale_idx]
                edge_feats = torch.cat([x[src], edge_attr_scaled], dim=-1)
            else:
                zeros = torch.zeros((src.numel(), 1), device=x.device, dtype=x.dtype)
                edge_feats = torch.cat([x[src], zeros], dim=-1)

            msg = self.lin_msg_scales[scale_idx](edge_feats)
            agg = torch.zeros_like(x)
            count = torch.zeros(x.size(0), device=x.device, dtype=torch.float32)
            agg.index_add_(0, dst, msg)
            count.index_add_(0, dst, torch.ones(dst.size(0), device=x.device, dtype=torch.float32))
            count = count.clamp(min=1.0).unsqueeze(-1)
            scale_outputs.append(agg / count)

        scale_weights = F.softmax(self.scale_weights, dim=0)
        x_multi = torch.zeros_like(x)
        for i, out_i in enumerate(scale_outputs):
            x_multi = x_multi + scale_weights[i] * out_i

        out = x_self + x_multi
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out


class IntraPathSelfAttention(nn.Module):
    def __init__(self, in_dim: int, num_heads: int = 4, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        assert in_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.q_proj = nn.Linear(in_dim, in_dim)
        self.k_proj = nn.Linear(in_dim, in_dim)
        self.v_proj = nn.Linear(in_dim, in_dim)
        self.out_proj = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, in_dim) * 0.02)

    def forward(self, x):
        P, L, d = x.shape
        x = x + self.pos_encoding[:, :L, :]
        q = self.q_proj(x).view(P, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(P, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(P, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(P, L, d)
        out = self.out_proj(out)
        return x + out


class EnhancedPathEncoder(nn.Module):
    def __init__(self, in_ch: int, emb_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.initial_proj = nn.Linear(in_ch, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=7, padding=3)
        self.intra_attention = IntraPathSelfAttention(hidden, num_heads=4, dropout=dropout, max_len=100)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.final_proj = nn.Linear(hidden, emb_dim)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_feats: torch.Tensor):
        P, L, _ = seq_feats.shape
        x = self.initial_proj(seq_feats)
        x = self.norm1(x)
        x_t = x.transpose(1, 2)
        c1 = F.gelu(self.conv1(x_t))
        c2 = F.gelu(self.conv2(x_t))
        c3 = F.gelu(self.conv3(x_t))
        x_conv = (c1 + c2 + c3) / 3.0
        x_conv = x_conv.transpose(1, 2)
        x = x + self.dropout(x_conv)
        x = self.norm2(x)
        x_attn = self.intra_attention(x)
        x_pooled = self.adaptive_pool(x_attn.transpose(1, 2)).squeeze(-1)
        path_emb = self.final_proj(x_pooled)
        return path_emb, x_attn


class EnhancedCrossAttention(nn.Module):
    def __init__(self, grid_dim: int, path_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert grid_dim % num_heads == 0
        self.num_heads = num_heads
        self.grid_dim = grid_dim
        self.path_dim = path_dim
        self.head_dim = grid_dim // num_heads

        self.geom_attention = LearnableGeometricAttention(grid_dim=grid_dim, path_dim=path_dim, hidden_dim=64, num_layers=2, dropout=dropout)
        self.grid_norm = nn.LayerNorm(grid_dim)
        self.q_proj = nn.Linear(grid_dim, grid_dim)
        self.path_norm = nn.LayerNorm(path_dim)
        self.k_proj = nn.Linear(path_dim, grid_dim)
        self.v_proj = nn.Linear(path_dim, grid_dim)
        self.out_proj = nn.Linear(grid_dim, grid_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, grid_x, grid_coords, path_point_feats, path_coords, knn_idx=None):
        N = grid_x.size(0)
        P_total = path_point_feats.size(0)

        grid_norm = self.grid_norm(grid_x)
        q = self.q_proj(grid_norm).view(N, self.num_heads, self.head_dim)

        path_norm = self.path_norm(path_point_feats)
        k_all = self.k_proj(path_norm).view(P_total, self.num_heads, self.head_dim)
        v_all = self.v_proj(path_norm).view(P_total, self.num_heads, self.head_dim)

        if knn_idx is not None:
            w = self.geom_attention(grid_x, grid_coords, path_point_feats, path_coords, knn_idx=knn_idx)
            k_knn = k_all[knn_idx]
            v_knn = v_all[knn_idx]
            k_knn = k_knn.permute(0, 2, 1, 3).contiguous()
            v_knn = v_knn.permute(0, 2, 1, 3).contiguous()
            scores = torch.einsum("nhd,nhkd->nhk", q, k_knn) * self.scale
            bias = torch.log(w.clamp(min=1e-6)).unsqueeze(1)
            scores = scores + bias
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.einsum("nhk,nhkd->nhd", attn, v_knn)
            out = out.reshape(N, self.grid_dim)
            out = self.out_proj(out)
            out = self.dropout(out)
            return grid_x + out

        w = self.geom_attention(grid_x, grid_coords, path_point_feats, path_coords, knn_idx=None)
        k = k_all.permute(1, 0, 2).contiguous()
        v = v_all.permute(1, 0, 2).contiguous()
        qh = q.permute(1, 0, 2).contiguous()
        scores = torch.matmul(qh, k.transpose(-2, -1)) * self.scale
        bias = torch.log(w.clamp(min=1e-6)).unsqueeze(0)
        scores = scores + bias
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(1, 0, 2).contiguous().view(N, self.grid_dim)
        out = self.out_proj(out)
        out = self.dropout(out)
        return grid_x + out


class DRegionImagerEnhancedGNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        path_seq_in_ch: int,
        path_static_dim: int,
        hidden_dim: int = 256,
        path_emb_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.1,
        num_cross_attn_layers: int = 2,
        num_heads: int = 8,
        num_scales: int = 3,
        scale_factors: list = None,
    ):
        super().__init__()
        if scale_factors is None:
            scale_factors = [1.0, 0.5, 0.25]

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.path_encoder = EnhancedPathEncoder(in_ch=path_seq_in_ch, emb_dim=path_emb_dim, hidden=128, dropout=dropout)
        self.path_fuse = MLP(path_emb_dim + path_static_dim, hidden=128, out_dim=hidden_dim, depth=3, dropout=dropout)

        self.point_proj = nn.Linear(128, hidden_dim)
        self.point_ctx_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cross_attn_layers = nn.ModuleList([
            EnhancedCrossAttention(grid_dim=hidden_dim, path_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_cross_attn_layers)
        ])

        self.graph_layers = nn.ModuleList([
            MultiScaleGraphConvLayer(hidden_dim, dropout=dropout, num_scales=num_scales, scale_factors=scale_factors)
            for _ in range(num_layers)
        ])

        self.fusion_weights = nn.Parameter(torch.ones(num_cross_attn_layers + 1))
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        self.hprime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        grid_coords: torch.Tensor,
        multi_edge_index: list,
        path_seq_feats: torch.Tensor,
        path_static_feats: torch.Tensor,
        path_coords: torch.Tensor,
        multi_edge_attr: list = None,
        grid_to_path_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.node_encoder(node_feats)
        path_emb, point_feats = self.path_encoder(path_seq_feats)
        path_ctx = self.path_fuse(torch.cat([path_emb, path_static_feats], dim=-1))

        P, L, _ = point_feats.shape
        point_feats_flat = point_feats.reshape(P * L, -1)
        point_feats_proj = self.point_proj(point_feats_flat)

        path_ctx_rep = path_ctx.unsqueeze(1).expand(P, L, -1).reshape(P * L, -1)
        point_feats_for_attn = self.point_ctx_fuse(torch.cat([point_feats_proj, path_ctx_rep], dim=-1))

        cross_outputs = [x]
        for cross_attn in self.cross_attn_layers:
            x = cross_attn(x, grid_coords, point_feats_for_attn, path_coords, knn_idx=grid_to_path_idx)
            cross_outputs.append(x)

        fusion_w = F.softmax(self.fusion_weights, dim=0)
        x_fused = torch.zeros_like(x)
        for i, out_i in enumerate(cross_outputs):
            x_fused = x_fused + fusion_w[i] * out_i
        x = self.fusion_norm(x_fused)

        for graph_layer in self.graph_layers:
            x = graph_layer(x, multi_edge_index, multi_edge_attr)

        delta_hprime = self.hprime_head(x)
        delta_beta = self.beta_head(x)
        return torch.cat([delta_hprime, delta_beta], dim=-1)