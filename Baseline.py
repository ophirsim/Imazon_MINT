import torch
import torch.nn as nn
from vit import VisionTransformer

class Baseline(nn.Module):
    def __init__(self,
                img_size,
                emb_size,
                in_channels = 1,
                out_channels = 1,
                history = 1,
                patch_size=16,
                drop_path=0.1,
                drop_rate=0.1,
                learn_pos_emb=False,
                embed_dim=1024,
                depth=24,
                decoder_depth=8,
                num_heads=16,
                mlp_ratio=4.0,
                dtype=torch.float,
                device='cpu'):
        super().__init__()
        self.img_size = img_size
        self.emb_size = emb_size

        self.pos_emb = nn.Linear(1, emb_size, dtype=dtype, device=device)
        self.vit = VisionTransformer(img_size=img_size, in_channels=in_channels, out_channels=out_channels, history=history, patch_size=patch_size, drop_path=drop_path,
                                     drop_rate=drop_rate, learn_pos_emb=learn_pos_emb, embed_dim=embed_dim, depth=depth, decoder_depth=decoder_depth, num_heads=num_heads, mlp_ratio=mlp_ratio).to(device)

    def forward(self, X, gap):
        pos_emb = self.pos_emb(gap)
        shaped_pos_emb = torch.reshape(pos_emb, X.shape)
        X_emb = X + shaped_pos_emb
        out = self.vit(X_emb)
        return out
