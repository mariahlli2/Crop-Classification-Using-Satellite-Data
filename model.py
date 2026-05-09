

import torch
import torch.nn as nn
import numpy as np


class ALPE(nn.Module): 
    def __init__(self, n_timesteps: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.T = n_timesteps
        self.register_buffer("pe", self._sinusoidal(n_timesteps, d_model))
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    @staticmethod
    def _sinusoidal(T, dm):
        pe = torch.zeros(T, dm)
        pos = torch.arange(T, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dm, 2, dtype=torch.float)
                        * (-np.log(10000.0) / dm))
        pe[:, 0::2] = torch.sin(pos * div)
        if dm % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, mask):
        mask = mask.bool()
        B, T = mask.shape
        pe = self.pe.unsqueeze(0).expand(B, -1, -1)
        m = (~mask).unsqueeze(-1).expand_as(pe)
        pe = pe * m
        pe = pe.permute(0, 2, 1)
        pe = self.conv1d(pe)
        return pe.permute(0, 2, 1)


class TransformerSubModule(nn.Module):
    
    def __init__(self, d_model, n_head, use_alpe=True, n_timesteps=36):
        super().__init__()
        self.use_alpe = use_alpe
        if use_alpe:
            self.alpe = ALPE(n_timesteps, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        if self.use_alpe and mask is not None:
            x = x + self.alpe(mask)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask.bool() if mask is not None else None)
        x = self.norm1(x + self.dropout(attn_out))
        x = x + self.ff(x)
        return self.norm2(x)


class CNNSubModule(nn.Module):    
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + self.skip(x))


class CTFusion(nn.Module):
    
    def __init__(self, d_model, n_head, cnn_kernel, use_alpe=True, n_timesteps=36, pool_size=2):
        super().__init__()
        self.transformer = TransformerSubModule(d_model, n_head, use_alpe=use_alpe, n_timesteps=n_timesteps)
        self.cnn = CNNSubModule(d_model, d_model, kernel=cnn_kernel)
        self.proj = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
        self.pool = nn.MaxPool1d(pool_size, stride=pool_size)

    def forward(self, x, mask=None):
        t_out = self.transformer(x, mask)
        c_out = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        fused = torch.cat([t_out, c_out], dim=-1)
        fused = self.proj(fused.permute(0, 2, 1))
        fused = self.pool(fused)
        return fused.permute(0, 2, 1)


class MCTNet(nn.Module):

    def __init__(self, n_classes, n_bands=10, n_timesteps=36, n_stages=3, 
                 n_head=5, cnn_kernel=3, mlp_hidden=64,
                 n_covariates=0, cov_mlp_hidden=32, use_covariates=True):
        super().__init__()
        self.use_covariates = use_covariates
        self.n_covariates = n_covariates
        
  
        self.stages = nn.ModuleList()
        T = n_timesteps
        for i in range(n_stages):
            use_alpe = (i == 0)
            self.stages.append(
                CTFusion(
                    d_model=n_bands,
                    n_head=n_head,
                    cnn_kernel=cnn_kernel,
                    use_alpe=use_alpe,
                    n_timesteps=T,
                    pool_size=2,
                )
            )
            T = T // 2
            
        self.global_max = nn.AdaptiveMaxPool1d(1)

        if use_covariates and n_covariates > 0:
            self.cov_mlp = nn.Sequential(
                nn.Linear(n_covariates, cov_mlp_hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(cov_mlp_hidden, cov_mlp_hidden // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            classifier_input = n_bands + (cov_mlp_hidden // 2)
        else:
            self.cov_mlp = None
            classifier_input = n_bands

      
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden, n_classes),
        )

    def forward(self, x, mask=None, covariates=None):
       
        for i, stage in enumerate(self.stages):
            m = mask if i == 0 else None
            x = stage(x, m)
            
      
        spectral_features = self.global_max(x.permute(0, 2, 1)).squeeze(-1)

        
        if self.use_covariates and self.cov_mlp is not None:
            if covariates is None:
                raise ValueError("Covariates enabled but not provided!")
            cov_features = self.cov_mlp(covariates)
            features = torch.cat([spectral_features, cov_features], dim=-1)
        else:
            features = spectral_features

        return self.classifier(features)
    
    def get_n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)