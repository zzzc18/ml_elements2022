import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import AdultDataset, AvilaDataset
from tqdm import tqdm
from torchmetrics import Accuracy


class AvilaMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(10, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 12)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


def train(model, optimizer, dataloader):
    loss_func = nn.CrossEntropyLoss()
    accuracy = Accuracy(num_classes=12)
    for data, label in tqdm(dataloader):
        label = label.long()
        output = model(data)
        loss = loss_func(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy.update(output, label)

    return accuracy.compute().item()


@torch.inference_mode()
def validate(model, optimizer, dataloader):
    loss_func = nn.CrossEntropyLoss()
    accuracy = Accuracy(num_classes=12)
    for data, label in tqdm(dataloader):
        label = label.long()
        output = model(data)
        accuracy.update(output, label)

    return accuracy.compute().item()


if __name__ == "__main__":
    '''
    NN for AvilaDataset
    '''
    model = AvilaMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-2)

    train_dataset = AvilaDataset(mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = AvilaDataset(mode="val")
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    best_acc_train = 0
    best_acc_val = 0
    for epoch in range(200):
        print(f"Running at epoch {epoch}")

        print("Training...")
        train_acc = train(model, optimizer, train_dataloader)
        print(f"Train Acc: {train_acc}")

        print("Validating...")
        val_acc = validate(model, optimizer, train_dataloader)
        print(f"Validate Acc: {val_acc}")
        if val_acc > best_acc_val:
            best_acc_val = val_acc
            best_acc_train = train_acc
    # 0.9941514730453491 0.9972195625305176
    print(best_acc_train, best_acc_val)
