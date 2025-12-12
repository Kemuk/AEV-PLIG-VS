"""
Training module for GNN models.

This module provides the Trainer class and related functions for training
binding affinity prediction models.
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from aev_plig.config import Config
from math import sqrt
from scipy import stats


# ==================== Metrics ====================

def rmse(y, f):
    """
    Compute Root Mean Squared Error.

    Args:
        y: True values
        f: Predicted values

    Returns:
        float: RMSE
    """
    rmse_val = sqrt(((y - f)**2).mean(axis=0))
    return rmse_val


def mse(y, f):
    """
    Compute Mean Squared Error.

    Args:
        y: True values
        f: Predicted values

    Returns:
        float: MSE
    """
    mse_val = ((y - f)**2).mean(axis=0)
    return mse_val


def pearson(y, f):
    """
    Compute Pearson correlation coefficient.

    Args:
        y: True values
        f: Predicted values

    Returns:
        float: Pearson correlation
    """
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    """
    Compute Spearman correlation coefficient.

    Args:
        y: True values
        f: Predicted values

    Returns:
        float: Spearman correlation
    """
    rs = stats.spearmanr(y, f)[0]
    return rs


def concordance_index(y, f):
    """
    Compute concordance index (C-index).

    Args:
        y: True values
        f: Predicted values

    Returns:
        float: Concordance index
    """
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


# ==================== Trainer Class ====================

class Trainer:
    """
    Trainer class for training GNN models.

    Handles training loop, validation, early stopping, and checkpointing.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        device: PyTorch device (CPU or CUDA)
        y_scaler: StandardScaler for denormalizing predictions
        optimizer: PyTorch optimizer (if None, creates Adam optimizer)
        loss_fn: Loss function (if None, uses MSELoss)
        learning_rate: Learning rate for optimizer (default: from Config)
        weight_decay: Weight decay for optimizer (default: from Config)
    """

    def __init__(self, model, train_loader, valid_loader, device, y_scaler,
                 optimizer=None, loss_fn=None, learning_rate=None, weight_decay=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.y_scaler = y_scaler

        # Set up optimizer
        if learning_rate is None:
            learning_rate = Config.LEARNING_RATE
        if weight_decay is None:
            weight_decay = Config.WEIGHT_DECAY

        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer

        # Set up loss function
        if loss_fn is None:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_fn

        # Training state
        self.best_pc = -1.1  # Best Pearson correlation
        self.pcs = []  # History of Pearson correlations

    def train_epoch(self, epoch, log_interval=100):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
            log_interval: How often to print progress (default: 100)

        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, data.y.view(-1, 1).to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += (loss.item() * len(data.y))

            if batch_idx % log_interval == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch,
                    batch_idx * len(data.y),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader)
                ))

        avg_loss = total_loss / len(self.train_loader.dataset)
        print("Loss for epoch {}: {:.4f}".format(epoch, avg_loss))
        return avg_loss

    def validate(self):
        """
        Validate the model on validation set.

        Returns:
            tuple: (true_values, predictions) both denormalized
        """
        self.model.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()

        print('Make prediction for {} samples...'.format(len(self.valid_loader.dataset)))

        with torch.no_grad():
            for data in self.valid_loader:
                data = data.to(self.device)
                output = self.model(data)
                total_preds = torch.cat((total_preds, output.cpu()), 0)
                total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

        # Denormalize predictions and labels
        y_true = self.y_scaler.inverse_transform(
            total_labels.numpy().flatten().reshape(-1, 1)
        ).flatten()
        y_pred = self.y_scaler.inverse_transform(
            total_preds.detach().numpy().flatten().reshape(-1, 1)
        ).flatten()

        return y_true, y_pred

    def fit(self, n_epochs, model_save_path, early_stopping_window=None):
        """
        Train the model for multiple epochs with early stopping.

        Args:
            n_epochs: Number of epochs to train
            model_save_path: Path to save best model checkpoint
            early_stopping_window: Window size for rolling average (default: from Config)

        Returns:
            dict: Training history with losses and metrics
        """
        if early_stopping_window is None:
            early_stopping_window = Config.EARLY_STOPPING_WINDOW

        print('Training for {} epochs...'.format(n_epochs))
        self.model.to(self.device)

        history = {
            'train_loss': [],
            'val_pc': [],
            'val_rmse': []
        }

        for epoch in range(n_epochs):
            # Train
            train_loss = self.train_epoch(epoch + 1)
            history['train_loss'].append(train_loss)

            # Validate
            G, P = self.validate()
            current_pc = pearson(G, P)
            current_rmse = rmse(G, P)
            self.pcs.append(current_pc)
            history['val_pc'].append(current_pc)
            history['val_rmse'].append(current_rmse)

            # Early stopping based on rolling average of Pearson correlation
            low = np.maximum(epoch - (early_stopping_window - 1), 0)
            avg_pc = np.mean(self.pcs[low:epoch + 1])

            if avg_pc > self.best_pc:
                torch.save(self.model.state_dict(), model_save_path)
                self.best_pc = avg_pc
                print('Model saved! Rolling avg PC: {:.4f}'.format(avg_pc))

            print('Current validation Pearson correlation: {:.4f}'.format(current_pc))
            print('Current validation RMSE: {:.4f}'.format(current_rmse))
            print('Best rolling avg PC so far: {:.4f}'.format(self.best_pc))
            print('-' * 50)

        return history

    def predict(self, test_loader):
        """
        Make predictions on test set.

        Args:
            test_loader: DataLoader for test data

        Returns:
            tuple: (true_values, predictions) both denormalized
        """
        self.model.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()

        print('Make prediction for {} samples...'.format(len(test_loader.dataset)))

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                total_preds = torch.cat((total_preds, output.cpu()), 0)
                total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

        # Denormalize predictions and labels
        y_true = self.y_scaler.inverse_transform(
            total_labels.numpy().flatten().reshape(-1, 1)
        ).flatten()
        y_pred = self.y_scaler.inverse_transform(
            total_preds.detach().numpy().flatten().reshape(-1, 1)
        ).flatten()

        return y_true, y_pred
