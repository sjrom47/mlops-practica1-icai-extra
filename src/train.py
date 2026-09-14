from pathlib import Path

import torch
import torch.nn as nn
from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader
from torchvision import models, transforms
import mlflow
import dagshub
dagshub.init(repo_owner='sjrom47', repo_name='mlops-practica-icai', mlflow=True)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
batch_size = 64
lr = 1e-3
with mlflow.start_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_ds = PneumoniaMNIST(
        split="train",
        transform=transform,
        download=True,
        as_rgb=True,
        root=str(DATA_DIR),
    )
    test_ds = PneumoniaMNIST(
        split="test",
        transform=transform,
        download=True,
        as_rgb=True,
        root=str(DATA_DIR),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    n_classes = len(train_ds.info["label"])
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"epoch {epoch + 1}/{epochs} loss={running_loss / len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"test accuracy={correct / total:.4f}")
    mlflow.log_metric("test_accuracy", correct / total)
    mlflow.log_metric("test_loss", running_loss / len(test_loader))
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("lr", lr)
    mlflow.log_param("epochs", epochs)
