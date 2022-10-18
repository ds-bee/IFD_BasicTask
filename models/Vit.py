import torch
from vit_pytorch import ViT

Vit = ViT


if __name__ == "__main__":
    v = Vit(
        image_size=256,
        patch_size=32,
        num_classes=5,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(16, 3, 256, 256)
    preds = v(img)
    print(preds.shape)
