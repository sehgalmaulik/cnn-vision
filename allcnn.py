import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import os, pathlib, random
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from torchinfo import summary
from tqdm.auto import tqdm
from timeit import default_timer as timer
from pathlib import Path


NUM_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HIDDEN_LAYER_SIZE = 101

# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Setup path to a data folder
data_path = Path("data/")
image_path = data_path / "food-101-split"


train_dir = image_path / "train"
test_dir = image_path / "test"

NUM_CLASSES = len([d for d in os.scandir(train_dir) if d.is_dir()])


random.seed(42)
with open(data_path / "food-101/meta/classes.txt", "r") as f:
    class_names = [name.strip() for name in f.readlines()]

class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}


torch.manual_seed(42)

simple_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_conv = datasets.ImageFolder(train_dir, transform=simple_transform)
test_conv = datasets.ImageFolder(test_dir, transform=simple_transform)


train_loader_conv = DataLoader(train_conv,
                                batch_size=BATCH_SIZE,
                                num_workers=0,
                                shuffle=True)

test_loader_conv = DataLoader(test_conv,   
                                batch_size=BATCH_SIZE,
                                num_workers=0,
                                shuffle=False)


class AllCNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                        out_channels=8,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8,
                        out_channels=16,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                        stride=2),


            nn.Conv2d(in_channels=16,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                        stride=2),


            nn.Conv2d(in_channels=64,
                        out_channels=NUM_CLASSES,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=NUM_CLASSES,
                        out_channels=NUM_CLASSES,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                        stride=2),


            nn.Conv2d(in_channels=NUM_CLASSES,
                        out_channels=NUM_CLASSES,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=NUM_CLASSES,
                        out_channels=NUM_CLASSES,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1616,
                        out_features=NUM_CLASSES),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x= self.classifier(x)
        return x


torch.manual_seed(42)

def train_step(model : nn.Module,
               dataloader : DataLoader,
               loss_fn : nn.Module,
               optimizer : torch.optim.Optimizer,
               device  = device):
    
    model.train() # set model to train mode

    train_loss, train_acc = 0, 0

    for batch, (X,y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class  = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    # print(f"Train Loss : {train_loss:.4f}, Train Acc : {train_acc:.4f}")

    return train_loss, train_acc

def test_step(model : nn.Module,
              dataloader : DataLoader,
              loss_fn : nn.Module,
              device = device):
    
    model.eval() # set model to eval mode

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X,y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            X = X.to(device)
            y = y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    # print(f"Test Loss : {test_loss:.4f}, Test Acc : {test_acc:.4f}")

    return test_loss, test_acc

def train(model : nn.Module,
          train_dataloader : DataLoader,
          test_dataloader : DataLoader,
          loss_fn : nn.Module,
          optimizer : torch.optim.Optimizer,
          epochs : int,
          device = device):
    
    history = dict(train_loss = [],
                   train_acc = [],
                   test_loss = [],
                   test_acc = [])

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch : {epoch+1}/{epochs} : Train Loss : {train_loss:.4f} : Train Acc : {train_acc:.4f} : Test Loss : {test_loss:.4f} : Test Acc : {test_acc:.4f}")

    return history

torch.manual_seed(42)

#print the global variables
print(f"Device : {device}")
print(f"Batch Size : {BATCH_SIZE}")
print(f"Learning Rate : {LEARNING_RATE}")
print(f"Number of Epochs : {NUM_EPOCHS}")
print(f"Hidden Layer Size : {HIDDEN_LAYER_SIZE}")
print(f"Number of Classes : {len(class_names)}")

model_0 = AllCNN(input_shape=3, hidden_units=HIDDEN_LAYER_SIZE, output_shape=len(class_names)).to(device)
print(model_0.parameters)

# model_0 = model_0.to(device)
L2_REG = 0.001

print(model_0.parameters)

optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
loss_fn_0 = nn.CrossEntropyLoss()

start_time = timer()

history_0 = train(model_0, train_loader_conv, test_loader_conv, loss_fn_0, optimizer_0, NUM_EPOCHS, device)

end_time = timer()

print(f"Training time : {end_time - start_time:.2f}s")

def plot_history(history : dict):
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    plt.subplots_adjust(wspace=0.3)


    ax[0].plot(history["train_loss"], label="train_loss")
    ax[0].plot(history["test_loss"], label="test_loss")
    ax[0].set_title("Loss Curve")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(history["train_acc"], label="train_acc")
    ax[1].plot(history["test_acc"], label="test_acc")
    #set x and y labels
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Accuracy Curve")
    ax[1].legend()

    plt.show()

plot_history(history_0)