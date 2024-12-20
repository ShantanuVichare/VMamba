import os
import sys
import json
import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, KFold
from torch import autocast, GradScaler

from torchinfo import summary

from utils.dataset import TumorMRIDataset, split_dataset_by_class
# from model.VisionMamba3D import VisionMamba3D

from model.VisionMamba3D_2 import VisionMamba3D
# Training function with mixed precision
def train_model(train_loader, model, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # Use autocast for mixed precision
        # with autocast(device.type):
        #     outputs = model(images)
        #     loss = criterion(outputs, labels)
        outputs = model(images)
        loss = criterion(outputs, labels)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            # Scale the loss before backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        running_loss += loss.item()
    # Now step the scheduler after optimizer step
    scheduler.step()
    return running_loss / len(train_loader)


# Test and Evaluation function
def test_model(test_loader, model, criterion, device, evaluate=False):
    model.eval()
    true_labels, pred_labels = [], []
    running_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    results = [pred_labels, running_loss / len(test_loader)]
    if evaluate:
        acc = accuracy_score(true_labels, pred_labels)
        cm = confusion_matrix(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        auc = roc_auc_score(true_labels, pred_labels)
        auc_pr = average_precision_score(true_labels, pred_labels)
        results.append((acc, cm.tolist(), f1, auc, auc_pr))
    return results


# Cross-validation function
def cross_validate(train_loader, model_class, criterion, optimizer_class, scheduler_class, device, num_epochs=20, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_loader.dataset)):
        print(f"Fold {fold+1}/{k_folds}")

        train_subset = torch.utils.data.Subset(train_loader.dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_loader.dataset, val_idx)

        train_loader_fold = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader_fold = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Instantiate model and optimizer
        model = model_class().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        scheduler = scheduler_class(optimizer)

        # Initialize GradScaler for mixed precision training
        # scaler = GradScaler()
        scaler = None

        # Train and evaluate on each fold
        for epoch in range(num_epochs):
            train_loss = train_model(train_loader_fold, model, criterion, optimizer, scheduler, device, scaler)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')

        _1, _2, (val_acc, _, val_f1, val_auc, val_auc_pr) = test_model(val_loader_fold, model, criterion, device, evaluate=True)
        fold_results.append((val_acc, val_f1, val_auc, val_auc_pr))

    # Return average metrics across folds
    avg_acc = np.mean([r[0] for r in fold_results])
    avg_f1 = np.mean([r[1] for r in fold_results])
    avg_auc = np.mean([r[2] for r in fold_results])
    avg_auc_pr = np.mean([r[3] for r in fold_results])

    return avg_acc, avg_f1, avg_auc, avg_auc_pr


# Results saving function
def save_results_to_file(file_path, epoch_count, test_acc, test_cm, test_f1, test_auc, test_auc_pr, extra_info:str=None):
    with open(file_path+'.log', 'w') as f:
        f.write(f"Epochs: {epoch_count}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Confusion Matrix:\n{test_cm}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"Test AUC-PR: {test_auc_pr:.4f}\n")
        if extra_info:
            f.write(extra_info+'\n')
    print('Results saved to file')

def save_loss_plots_to_file(file_path, train_losses, test_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(file_path+'.png')
    
    # Save losses to json file
    with open(file_path+'.json', 'w') as f:
        json.dump({'train_losses': train_losses, 'test_losses': test_losses}, f)
    print('Losses plots and values saved')
    
def save_model_to_file(file_path, model):
    torch.save(model.state_dict(), file_path+'.pth')
    print('Model saved to file')

def getTime():
    return datetime.datetime.now().strftime("%y%m%d-%Hh%Mm")

startTime = getTime()

# Paths and configurations
root_dir = os.getenv('DATASET_PATH', './data/MICCAI_BraTS_2019_Data_Training/')

batch_size = 8
initial_lr = 1e-5
num_epochs = 100
data_limit = None # 1000
weight_decay = 1e-3
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
print('Parameters:', batch_size, initial_lr, num_epochs, data_limit, weight_decay, device)

# Dataset and DataLoader
dataset = TumorMRIDataset(root_dir, limit=data_limit)
nChannels, *imgSize = dataset[0][0].shape
train_samples, test_samples, distribution_info = split_dataset_by_class(dataset, train_ratio=0.8)
print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}, Distributions: {distribution_info}")

# Create datasets and loaders for train and test sets
train_dataset = torch.utils.data.Subset(dataset, [dataset.samples.index(s) for s in train_samples])
test_dataset = torch.utils.data.Subset(dataset, [dataset.samples.index(s) for s in test_samples])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# sys.exit()

# Model, Loss, Optimizer, and Scheduler
model_class = lambda: VisionMamba3D(
    img_size=imgSize, # (155, 240, 240)
    patch_size=(5, 8, 8),
    in_chans=nChannels, num_classes=2,
    depths=[2, 2, 6, 2],
    dims=[96, 192, 384, 768],
    debug=False,
    )
criterion = nn.CrossEntropyLoss()
optimizer_class = lambda params: optim.AdamW(params, lr=initial_lr, weight_decay=weight_decay)
scheduler_class = lambda opt: StepLR(opt, step_size=2, gamma=0.5)

# Perform cross-validation
# cv_acc, cv_f1, cv_auc, cv_auc_pr = cross_validate(train_loader, model_class, criterion, optimizer_class, scheduler_class, device, num_epochs=num_epochs, k_folds=5)

# Train on the full training set and evaluate on the test set
model = model_class().to(device)
optimizer = optimizer_class(model.parameters())
scheduler = scheduler_class(optimizer)

print('Model Summary')
summary(model, input_size=(batch_size, nChannels, *imgSize), depth=5)
# sys.exit()

# Initialize GradScaler for full training
# scaler = GradScaler()
scaler = None

# Train model on the entire training set
# with torch.autograd.detect_anomaly(): # For debugging NaNs
train_losses, test_losses = [], []
print('Training started at:', startTime)
# Create results directory if it doesn't exist
os.makedirs(f'results', exist_ok=True)
try:
    for epoch in range(num_epochs):
        train_loss = train_model(train_loader, model, criterion, optimizer, scheduler, device, scaler)
        _, test_loss, evals = test_model(test_loader, model, criterion, device, evaluate=True)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        print('Test Evaluation:\n', 'Accuracy:', evals[0], 'CM:', evals[1], 'F1:', evals[2], 'AUC:', evals[3], 'AUC-PR:', evals[4])
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Overwrite results to file after each epoch
        # Save plots of training and test losses to disk 
        save_loss_plots_to_file(f'results/{startTime}-losses', train_losses, test_losses)
        # Save test results to file
        save_results_to_file(f'results/{startTime}-results', epoch+1, *evals)

except Exception as e:
    print('Error occured:', e)
finally:
    print('Training finished at:', getTime())

# Save model to disk
save_model_to_file(f'results/{startTime}-model', model)

