import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class AttentionNet(nn.Module):
    def __init__(self, d_model=64, nhead=4, dim_ff=128, num_layers=2):
        super().__init__()
        self.patch_size = 7
        self.num_patches = 16
        self.proj = nn.Linear(7 * 7, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        B = x.shape[0]
        # (B,1,28,28) -> (B,16,49): 4x4 grid of 7x7 patches
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)       # (B,1,4,4,7,7)
        x = x.reshape(B, self.num_patches, 49)       # (B,16,49)
        x = self.proj(x)                             # (B,16,64)
        x = self.encoder(x)                          # (B,16,64)
        x = x.mean(dim=1)                            # mean pool
        return self.head(x)


def get_models():
    return {"MLP": MLP(), "CNN": CNN(), "Attention": AttentionNet()}


if __name__ == "__main__":
    x = torch.randn(2, 1, 28, 28)
    for name, model in get_models().items():
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:10s} output: {tuple(out.shape)}  params: {n_params:,}")