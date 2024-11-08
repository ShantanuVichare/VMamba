import json
import os
import sys
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
import optuna
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt


from utils.dataset import TumorMRIDataset, split_dataset_by_class
from model.VisionMamba3D import VisionMamba3D

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class HyperparameterOptimizer:
    def __init__(self, dataset, model_class, device, base_path='results'):
        self.dataset = dataset
        self.model_class = model_class
        self.device = device
        self.base_path = base_path
        
        train_samples, val_samples, distribution_info = split_dataset_by_class(dataset)
        print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}, Distributions: {distribution_info}")

        # Create datasets and loaders for train and test sets
        self.train_dataset = Subset(dataset, [dataset.samples.index(s) for s in train_samples])
        self.val_dataset = Subset(dataset, [dataset.samples.index(s) for s in val_samples])
        
        # Create train/val/test splits
        # total_size = len(dataset)
        # train_size = int(0.7 * total_size)
        # val_size = int(0.15 * total_size)
        # test_size = total_size - train_size - val_size
        
        # self.train_dataset, self.val_dataset, self.test_dataset = random_split(
        #     dataset, [train_size, val_size, test_size]
        # )

    def objective(self, trial):
        # Hyperparameters to optimize
        batch_size = trial.suggest_int('batch_size', 4, 16)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        scheduler_type = trial.suggest_categorical('scheduler', ['cosine', 'onecycle'])
        
        # Create data loaders
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)
        
        # Initialize model and training components
        model = self.model_class().to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        elif scheduler_type == 'onecycle':
            scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=50, 
                                 steps_per_epoch=len(train_loader))
        else:
            raise ValueError('Invalid scheduler type')
        
        # Training loop with early stopping
        early_stopping = EarlyStopping(patience=10)
        scaler = GradScaler()
        best_val_loss = float('inf')
        
        train_losses, val_losses = [], []
        for epoch in range(50):  # Max 50 epochs
            # Training
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            if scheduler_type == 'cosine':
                scheduler.step()
            
            # Early stopping check
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break
            
            # Report intermediate value
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print('Pruned trial params:', trial.params)
                save_trial_to_file(f'{self.base_path}/{trial.number}-pruned-params', trial)
                # Save loss plots for the current trial
                save_loss_plots_to_file(f'{self.base_path}/{trial.number}-pruned-losses', train_losses, val_losses)
                raise optuna.TrialPruned()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        print('Completed trial params:', trial.params)
        save_trial_to_file(f'{self.base_path}/{trial.number}-params', trial)
        # Save loss plots for the current trial
        save_loss_plots_to_file(f'{self.base_path}/{trial.number}-losses', train_losses, val_losses)
        
        return best_val_loss

    def find_best_params(self, n_trials=100):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
        return trial.params, study

def save_trial_to_file(file_path, trial):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    trial_info = {trial.number: trial.params}
    # Save trial to file
    with open(file_path+'.json', 'w') as f:
        json.dump(trial_info, f, indent=4)

def save_loss_plots_to_file(file_path, train_losses, val_losses):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(file_path + '.png')
    plt.close()
    
    # Save losses to file
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses
    }, file_path + '.pt')
    print('Losses plots and values saved to:', file_path)

# def train_with_best_params(params, dataset, model_class, device, base_path):
#     # Implementation of final training with best parameters
#     batch_size = params['batch_size']
#     lr = params['lr']
#     weight_decay = params['weight_decay']
#     scheduler_type = params['scheduler']
#     num_epochs = 50
    
#     train_samples, val_samples, distribution_info = split_dataset_by_class(dataset)
#     print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}, Distributions: {distribution_info}")

#     # Create datasets and loaders for train and test sets
#     train_dataset = Subset(dataset, [dataset.samples.index(s) for s in train_samples])
#     val_dataset = Subset(dataset, [dataset.samples.index(s) for s in val_samples])
#     # Create data loaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
#     # Initialize model and training components
#     model = model_class().to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     criterion = nn.CrossEntropyLoss()
        
#     if scheduler_type == 'cosine':
#         scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
#     elif scheduler_type == 'onecycle':
#         scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=50, 
#                                 steps_per_epoch=len(train_loader))
#     else:
#         raise ValueError('Invalid scheduler type')
    
#     # Create data loaders, model, optimizer, etc. using best params
#     # ... (similar to objective function but for final training)
    
    
#     # Save results and model
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'params': params,
#         'final_metrics': metrics
#     }, base_path + '.pth')

# Usage example:
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%y%m%d-%Hh%Mm")
    save_path = f'results/final_model_{timestamp}'
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    # Paths and configurations
    root_dir = os.getenv('DATASET_PATH', './data/MICCAI_BraTS_2019_Data_Training/')
    # Dataset and DataLoader
    dataset = TumorMRIDataset(root_dir)
    nChannels, *imgSize = dataset[0][0].shape

    # Model, Loss, Optimizer, and Scheduler
    model_class = lambda: VisionMamba3D(
        img_size=imgSize, # (155, 240, 240)
        patch_size=(5, 8, 8),
        in_chans=nChannels, num_classes=2,
        depths=[2, 2, 6, 2],
        dims=[96, 192, 384, 768],
        debug=False,
    )
    optimizer = HyperparameterOptimizer(dataset, model_class, device, base_path=save_path)
    best_params, study = optimizer.find_best_params(n_trials=20)
    print('Best hyperparameters:', best_params)
    # Save all trials with their trial number and hyperparameter values
    all_trials = study.trials
    trials_info = {trial.number: trial.params for trial in all_trials}
    trials_file_path = os.path.join(save_path, 'all_trials.json')

    os.makedirs(os.path.dirname(trials_file_path), exist_ok=True)
    with open(trials_file_path, 'w') as f:
        json.dump(trials_info, f, indent=4)

    print(f'All trials saved to: {trials_file_path}')
    # train_with_best_params(best_params, dataset, model_class, device, base_path=save_path)