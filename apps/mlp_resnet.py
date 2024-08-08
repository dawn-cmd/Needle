from tqdm import tqdm  # Import tqdm for progress bars
import needle.nn as nn
import numpy as np
import time
import os
from needle.data import MNISTDataset, DataLoader
import needle as ndl
import sys
sys.path.append("../python")

np.random.seed(0)
MY_DEVICE = ndl.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    main_path = nn.Sequential(nn.Linear(dim, hidden_dim, device=MY_DEVICE), norm(hidden_dim, device=MY_DEVICE), nn.ReLU(
    ), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim, device=MY_DEVICE), norm(dim, device=MY_DEVICE))
    res = nn.Residual(main_path)
    return nn.Sequential(res, nn.ReLU())


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    resnet = nn.Sequential(nn.Linear(dim, hidden_dim, device=MY_DEVICE), nn.ReLU(),
                           *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2,
                                           norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
                           nn.Linear(hidden_dim, num_classes, device=MY_DEVICE))
    return resnet


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    tot_loss, tot_error = [], 0.0
    loss_fn = nn.SoftmaxLoss()
    if opt is None:
        model.eval()
        for X, y in tqdm(dataloader, desc="Evaluating", leave=False):
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
    else:
        model.train()
        for X, y in tqdm(dataloader, desc="Training", leave=False):
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
            opt.reset_grad()
            loss.backward()
            opt.step()
    sample_nums = len(dataloader.dataset)
    return tot_error/sample_nums, np.mean(tot_loss)


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    print(data_dir)
    np.random.seed(4)
    resnet = MLPResNet(28*28, hidden_dim=hidden_dim,
                       num_classes=10, drop_prob=0.2)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz",
                             f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, device=MY_DEVICE)
    test_loader = DataLoader(test_set, batch_size=batch_size, device=MY_DEVICE)

    for epoch_num in range(epochs):
        print(f"Epoch {epoch_num + 1}/{epochs}")
        train_err, train_loss = epoch(train_loader, resnet, opt)
        print(f"Train Error: {train_err *
              100:.4f}%, Train Loss: {train_loss * 100:.4f}%")

    test_err, test_loss = epoch(test_loader, resnet, None)
    print(f"Test Error: {test_err*100:.4f}%, Test Loss: {test_loss*100:.4f}%")

    return train_err, train_loss, test_err, test_loss


if __name__ == "__main__":
    train_err, train_loss, test_err, test_loss = train_mnist(
        data_dir="/home/hyjing/Code/DeepLearningSystem/HW4/data/")
