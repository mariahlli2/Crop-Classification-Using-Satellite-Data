import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep  = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand  = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand  = torch.floor(rand + keep)
        return x * rand / keep


class ChannelAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(2, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=2)
        mx  = x.amax(dim=2)
        return torch.sigmoid(self.mlp(avg) + self.mlp(mx)).unsqueeze(2)


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7):
        super().__init__()
        self.ch = ChannelAttention1D(channels, reduction)
        self.sp = SpatialAttention1D(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.ch(x) * self.sp(x)


class CBAMResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 dropout: float = 0.10, cbam_reduction: int = 8,
                 drop_path: float = 0.0):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.cbam      = CBAM1D(out_ch, reduction=cbam_reduction)
        self.drop_path = DropPath(drop_path)
        self.skip      = (nn.Sequential(
                              nn.Conv1d(in_ch, out_ch, 1, bias=False),
                              nn.BatchNorm1d(out_ch),
                          ) if in_ch != out_ch else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.drop_path(self.cbam(self.block(x)))
        return self.relu(branch + self.skip(x))


class ALPE(nn.Module):
    def __init__(self, n_timesteps: int, d_model: int):
        super().__init__()
        self.register_buffer("pe", self._sinusoidal(n_timesteps, d_model))
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    @staticmethod
    def _sinusoidal(T: int, dm: int) -> torch.Tensor:
        pe  = torch.zeros(T, dm)
        pos = torch.arange(T, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dm, 2, dtype=torch.float)
                        * (-np.log(10000.0) / dm))
        pe[:, 0::2] = torch.sin(pos * div)
        if dm % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        return pe

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        B, T = mask.shape
        pe = self.pe.unsqueeze(0).expand(B, -1, -1)
        pe = pe * (~mask.bool()).unsqueeze(-1).expand_as(pe)
        return self.conv1d(pe.permute(0, 2, 1)).permute(0, 2, 1)


class TAHead(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_timesteps: int,
                 attn_drop: float = 0.05, ff_drop: float = 0.05):
        super().__init__()
        self.alpe  = ALPE(n_timesteps, d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_head,
                                            batch_first=True,
                                            dropout=attn_drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(ff_drop),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop  = nn.Dropout(attn_drop)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            x = x + self.alpe(mask)
        kpm = mask.bool() if mask is not None else None
        attn_out, _ = self.attn(x, x, x, key_padding_mask=kpm)
        x = self.norm1(x + self.drop(attn_out))
        return self.norm2(x + self.ff(x))


class TemporalMaskAug(nn.Module):
    def __init__(self, max_mask_ratio: float = 0.05):
        super().__init__()
        self.max_ratio = max_mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, T, C = x.shape
        n_mask = int(T * self.max_ratio * torch.rand(1).item())
        if n_mask == 0:
            return x
        idx = torch.randperm(T, device=x.device)[:n_mask]
        x = x.clone()
        x[:, idx, :] = 0.0
        return x


class CovariateMLP(nn.Module):
    def __init__(self, n_covariates: int, hidden: int = 48, out_dim: int = 32,
                 dropout1: float = 0.10, dropout2: float = 0.05):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(n_covariates, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout1),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
        )
        self.skip = nn.Linear(n_covariates, out_dim, bias=True)

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(cov)) + self.skip(cov)


class CrossGate(nn.Module):
    def __init__(self, spec_dim: int, cov_dim: int):
        super().__init__()
        self.gate_s = nn.Linear(cov_dim,  spec_dim, bias=True)
        self.gate_c = nn.Linear(spec_dim, cov_dim,  bias=True)

    def forward(self, s: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        gs = torch.sigmoid(self.gate_s(c))   # (B, spec_dim)
        gc = torch.sigmoid(self.gate_c(s))   # (B, cov_dim)
        return torch.cat([s * gs, c * gc], dim=1)


class AttentionCNN(nn.Module):

    def __init__(
        self,
        n_classes:       int,
        n_bands:         int   = 10,
        n_timesteps:     int   = 36,
        n_stages:        int   = 3,          
        n_head:          int   = 4,
        cnn_kernel:      int   = 3,
        mlp_hidden:      int   = 256,
        n_covariates:    int   = 8,
        cov_mlp_hidden:  int   = 48,
        cov_out_dim:     int   = 32,
        use_covariates:  bool  = True,
        # architecture knobs
        d_model:         int   = 64,
        cbam_reduction:  int   = 8,
        drop_path_rate:  float = 0.0,
        aug_mask_ratio:  float = 0.0,
        attn_drop:       float = 0.10,   
    ):
        super().__init__()
        self.use_covariates = use_covariates
        self.aug = TemporalMaskAug(aug_mask_ratio)
        dp = [drop_path_rate * i / 2 for i in range(3)]
        self.block0 = CBAMResBlock(n_bands,     d_model,     cnn_kernel,
                                   0.15, cbam_reduction, dp[0])
        self.pool0  = nn.MaxPool1d(2, stride=2)

        self.block1 = CBAMResBlock(d_model,     d_model * 2, cnn_kernel,
                                   0.15, cbam_reduction, dp[1])
        self.pool1  = nn.MaxPool1d(2, stride=2)

        self.block2 = CBAMResBlock(d_model * 2, d_model * 2, cnn_kernel,
                                   0.15, cbam_reduction, dp[2])

        T_out = n_timesteps // 4
        self.ta = TAHead(d_model * 2, n_head, T_out,
                         attn_drop=attn_drop, ff_drop=0.05)

        spec_dim = d_model * 2 * 2        
        
        if use_covariates and n_covariates > 0:
            self.cov_mlp  = CovariateMLP(n_covariates,
                                         hidden=cov_mlp_hidden,
                                         out_dim=cov_out_dim,
                                         dropout1=0.25,
                                         dropout2=0.15)
            self.crossgate = CrossGate(spec_dim, cov_out_dim)
            feat_dim = spec_dim + cov_out_dim
        else:
            self.cov_mlp   = None
            self.crossgate = None
            feat_dim = spec_dim

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(mlp_hidden // 2, n_classes),
        )

    def forward(
        self,
        x:          torch.Tensor,
        mask:       torch.Tensor = None,
        covariates: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.aug(x)
        x = x.permute(0, 2, 1)
        x = self.pool0(self.block0(x))
        x = self.pool1(self.block1(x))
        x = self.block2(x)

        if mask is not None:
            mask_down = mask[:, ::4]
            T_out = x.shape[2]
            mask_down = mask_down[:, :T_out]
        else:
            mask_down = None

        x = self.ta(x.permute(0, 2, 1), mask_down)
        x = x.permute(0, 2, 1)

        spec = torch.cat([x.mean(2), x.amax(2)], dim=1)

        if self.use_covariates and self.cov_mlp is not None \
                and covariates is not None:
            cov  = self.cov_mlp(covariates)
            feat = self.crossgate(spec, cov)
        else:
            feat = spec

        return self.classifier(feat)

    def get_n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


MCTNet = AttentionCNN


if __name__ == "__main__":
    B, T, C = 8, 36, 10
    x    = torch.randn(B, T, C)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, -6:] = True
    cov  = torch.randn(B, 8)

    for n_cls, state in [(5, "Arkansas"), (6, "California")]:
        m = AttentionCNN(n_classes=n_cls, n_bands=C, n_timesteps=T,
                         n_covariates=8, use_covariates=True)
        m.eval();  out_e = m(x, mask, cov)
        m.train(); out_t = m(x, mask, cov)
        print(f"[{state}]  params={m.get_n_params():,}  "
              f"output shape: {out_e.shape}")