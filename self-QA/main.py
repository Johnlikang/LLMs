from transformers import AdamW, get_scheduler
from dataloader import model, train_dataloader, valid_dataloader
from train import train_loop
from test import test_loop
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

learning_rate = 2e-5
epoch_num = 3

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)


def plot_loss(train_loss, save_path):
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, '-o', label='Training Loss', linewidth=2, markersize=6)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    total_loss = []
    loss = 0
    best_bleu = 0
    for t in range(epoch_num):
        print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
        loss_fn = nn.CrossEntropyLoss()
        loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, loss)
        epoch_loss = loss / ((t + 1) * len(train_dataloader))
        total_loss.append(epoch_loss)
        valid_bleus = test_loop(valid_dataloader, model)
        print(f"loss:{loss} BLEU: {valid_bleus}\n")
        if valid_bleus['bleu4'] > best_bleu:
            best_bleu = valid_bleus['bleu4']
            print('saving new weights...\n')
            torch.save(model.state_dict(), f'epoch_{t+1}_valid_bleu_{valid_bleus['bleu4']:0.2f}_model_weights.bin')
    plot_loss(
        train_loss=total_loss,
        save_path="convergence_curve.png"
    )
    print("Done!")