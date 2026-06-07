import torch
import torch.nn as nn

from config import GPT_CONFIG_124M
from gelu import GELU


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, GPT_CONFIG_124M["emb_dim"])
    out = ffn(x)
    print(out.shape)
