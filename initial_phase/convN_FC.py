import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import csv

import sys
import os
sys.path.append("store_outputs")
from log_results import log_result

# ------------------- Model ------------------- #

class MultiKernelCNN(nn.Module):
    def __init__(self, num_kernels=8, kernel_size=7, use_pool=True, in_channels=1, num_classes=10, input_hw=(28, 28)):
        super().__init__()
        pad = kernel_size // 2  

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_kernels,   
            kernel_size=kernel_size,   
            stride=1,
            padding=pad,
            bias=True
        )

        self.use_pool = use_pool

        H, W = input_hw
        fc_dim = num_kernels * (H // 2) * (W // 2)
        self.fc = nn.Linear(fc_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))          # [B,K,H,W]
        if self.use_pool:
            x = F.max_pool2d(x, 2)        # [B,K,H/2,W/2]
        x = x.flatten(1)                  # [B,K]
        return self.fc(x)                 # [B,num_classes]

# ------------------- Plot kernels ------------------- #
def plot_kernels_grid(model, title="Learned conv kernels", file_name=None):
    W = model.conv.weight.detach().cpu()  
    K = W.shape[0]

    cols = min(4, K)
    rows = (K + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.8 * rows))
    fig.suptitle(title)

    # Make axes always 2D indexable
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.axis("off")

        if i < K:
            im = ax.imshow(W[i, 0], cmap="gray")
            ax.set_title(f"kernel {i}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if file_name is not None:
        png_name = os.path.splitext(file_name)[0] + ".png"
        plt.savefig(png_name, dpi=300, bbox_inches="tight")

    plt.show()

# ------------------- main() ------------------- #
def main(
    dataset="MNIST", # "MNIST", "Fashion", or "CIFAR_G"
    num_kernels=8,
    kernel_size=7,
    use_pool=True,
    batch_size=64,
    lr=7e-4,
    num_epochs=20,
    input_hw=(28, 28),
    device=None,
    save_name_kernel='default_MNIST_kernels_7x7.pt',
    save_name_fc='default_MNIST_fc_7x7.pt'
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------- Choose dataset ------------------- #
    transform = transforms.ToTensor()

    if dataset == "MNIST":
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    elif dataset == "Fashion":
        train_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )

    elif dataset == "CIFAR_G":
        # CIFAR-10 converted to grayscale
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # ------------------- loaders ------------------- #
    n_val = int(0.1 * len(train_dataset))
    n_train = len(train_dataset) - n_val

    gen = torch.Generator().manual_seed(11)
    train_ds, val_ds = random_split(train_dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Loaded {dataset}:")
    print(f"Loaded {dataset}: train={len(train_ds)} val={len(val_ds)} test={len(test_dataset)}")

    # ------------------- Train setup ------------------- #
    model = MultiKernelCNN(num_kernels=num_kernels, kernel_size=kernel_size, use_pool=use_pool, input_hw=input_hw).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------- Train ------------------- #
    for epoch in range(num_epochs):
        # -------- train --------
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------- validation --------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()

        val_loss /= len(val_loader)

        # -------- validation accuracy --------
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.numel()

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"train loss = {train_loss:.4f} | "
            f"val loss = {val_loss:.4f} | "
            f"val acc = {val_acc*100:.2f}%"
        )

    # ------------------- Evaluate ------------------- #
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    print(f"Test accuracy: {acc:.2f}%")

    method = "conv + pool + FC"
    log_result(dataset, method, num_kernels, kernel_size, acc)


    # ------------------- Save kernels + plot ------------------- #
    kernels = model.conv.weight.detach().cpu()  
    torch.save(kernels, save_name_kernel)
    print(f"Saved kernels {save_name_kernel}")

    fc_weights = model.fc.weight.detach().cpu()  
    fc_bias    = model.fc.bias.detach().cpu()   

    torch.save(
        {
            "weight": fc_weights,
            "bias": fc_bias,
        },
        save_name_fc
    )

    plot_kernels_grid(model, title="Learned conv kernels", file_name=save_name_kernel)


if __name__ == "__main__":
    main(
        dataset="MNIST",
        num_kernels=8,
        kernel_size=7,
        use_pool=True,
        batch_size=64,
        lr=7e-4,
        num_epochs=20,
        save_name_kernel="MNIST_kernels_7x7.pt",
        save_name_fc="MNIST_fc_7x7.pt",
    )
