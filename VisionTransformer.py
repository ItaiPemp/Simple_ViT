import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
#from utils import *
import argparse
import utils
from modules import *
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 100 # how many independent sequences will we process in parallel?
epochs = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
d_model = 256 #todo: change this
n_head = 8#todo: change this
n_layer = 8#todo: change this
dropout = 0.25
expansion_factor = 4
n_classes = 10
image_size = 32
patch_size = 8
num_patches = (image_size // patch_size) ** 2
# ------------


assert(image_size % patch_size == 0)

model = VisionTransformer(patch_size, num_patches, n_classes, d_model, n_layer, expansion_factor, n_head, dropout).to(device)
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable parameters: ", pytorch_total_params)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def calculate_accuracy(predictions, targets):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy

def train():
    train_loader, val_loader = utils.get_data(batch_size=batch_size)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        train_loss = 0
        train_accuracy = 0

        model.train()
        for batch_idx, (image, label) in enumerate(tqdm(train_loader)):
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += calculate_accuracy(output, label)

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        with torch.no_grad():
            model.eval()
            val_loss = 0
            val_accuracy = 0

            for batch_idx, (image, label) in enumerate(tqdm(val_loader)):
                image = image.to(device)
                label = label.to(device)

                output = model(image)
                loss = criterion(output, label)

                val_loss += loss.item()
                val_accuracy += calculate_accuracy(output, label)

            val_loss /= len(val_loader)
            val_accuracy /= len(val_loader)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print('Epoch: {}, train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(
                epoch, train_loss, train_accuracy, val_loss, val_accuracy))

    # Save the trained model
    torch.save(model.state_dict(), 'vision_transformer.pth')
    plot_and_save(train_losses, val_losses, 'Loss', 'loss.png')
    plot_and_save(train_accuracies, val_accuracies, 'Accuracy', 'accuracy.png')


def plot_and_save(train_values, val_values, metric_name, save_name):
    plt.figure()
    plt.plot(train_values, label='Training')
    plt.plot(val_values, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(save_name)
    plt.show()


if __name__ == '__main__':
    train()
