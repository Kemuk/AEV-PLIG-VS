"""
Training module for GNN models.

This module provides the Trainer class and related functions for training
binding affinity prediction models.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader
from aev_plig.config import Config
from aev_plig.models import GATv2NetMixedPrecision, GATv2NetBayesianMixedPrecision
from math import sqrt
from scipy import stats
from aev_plig.prediction import denormalize, denormalize_variance

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


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


def gaussian_nll_loss(mean, var, target):
    """
    Compute Gaussian negative log-likelihood loss for Bayesian models.

    Args:
        mean: Predicted mean values
        var: Predicted variance values (must be positive)
        target: True target values

    Returns:
        torch.Tensor: Mean NLL loss
    """
    return 0.5 * (torch.log(var) + (target - mean) ** 2 / var).mean()


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
                 optimizer=None, loss_fn=None, learning_rate=None, weight_decay=None,
                 use_amp=None, lr_body=None, lr_head=None, freeze_gnn=False):
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
            if model.is_bayesian:
                # VBLL: disable weight decay on the variational last layer —
                # it has its own KL regularisation and extra weight decay hurts.
                vbll_params = list(model.vbll_head.parameters())
                vbll_ids = {id(p) for p in vbll_params}
                other_params = [p for p in model.parameters() if id(p) not in vbll_ids]
                self.optimizer = torch.optim.Adam([
                    {"params": other_params, "weight_decay": weight_decay},
                    {"params": vbll_params,  "weight_decay": 0.0},
                ], lr=learning_rate)
            elif freeze_gnn:
                for m in list(model.GNN_layers) + list(model.BN_layers):
                    for p in m.parameters():
                        p.requires_grad_(False)
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr_head or learning_rate,
                    weight_decay=weight_decay,
                )
            elif lr_body is not None or lr_head is not None:
                gnn_ids = {id(p) for m in list(model.GNN_layers) + list(model.BN_layers)
                           for p in m.parameters()}
                self.optimizer = torch.optim.Adam([
                    {"params": [p for p in model.parameters() if id(p) in gnn_ids],
                     "lr": lr_body or learning_rate},
                    {"params": [p for p in model.parameters() if id(p) not in gnn_ids],
                     "lr": lr_head or learning_rate},
                ], weight_decay=weight_decay)
            else:
                self.optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
        else:
            self.optimizer = optimizer

        # Set up loss function
        if loss_fn is None:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_fn

        # Mixed precision training setup
        if use_amp is None:
            use_amp = isinstance(model, (GATv2NetMixedPrecision, GATv2NetBayesianMixedPrecision))
        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        if self.use_amp:
            print("Mixed precision training enabled (AMP + GradScaler)")

        # Training state
        self.best_pc = -1.1  # Best Pearson correlation
        self.pcs = []  # History of Pearson correlations

    def train_epoch(self, epoch, log_interval=100):
        """
        Train for one epoch.

        Automatically detects Bayesian models (those returning (mean, var) tuple)
        and uses Gaussian NLL loss instead of MSE loss.

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

            with autocast('cuda', enabled=self.use_amp):
                output = self.model(data)
                target = data.y.view(-1, 1).to(self.device)

                if self.model.is_bayesian:
                    # VBLL model: use ELBO loss from VBLLReturn dataclass
                    loss = output.train_loss_fn(target)
                elif isinstance(output, tuple):
                    # GATv2NetAleatoric: Gaussian NLL over (mean, var) tuple
                    mean, var = output
                    loss = gaussian_nll_loss(mean, var, target)
                else:
                    loss = self.loss_fn(output, target)

            self.scaler.scale(loss).backward()
            if self.model.is_bayesian:
                # VBLL: unscale then clip to stabilise covariance optimisation
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
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

        Automatically detects Bayesian models and extracts mean predictions.

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

                with autocast('cuda', enabled=self.use_amp):
                    output = self.model(data)

                if self.model.is_bayesian:
                    total_preds = torch.cat((total_preds, output.predictive.mean.cpu()), 0)
                elif isinstance(output, tuple):
                    mean, _ = output
                    total_preds = torch.cat((total_preds, mean.cpu()), 0)
                else:
                    total_preds = torch.cat((total_preds, output.cpu()), 0)

                total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

        # Denormalize predictions and labels
        y_true = denormalize(total_labels, self.y_scaler)
        y_pred = denormalize(total_preds, self.y_scaler)

        return y_true, y_pred

    def fit(self, n_epochs, model_save_path, early_stopping_window=None, max_training_hours=None):
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

        _t0 = time.time()
        for epoch in range(n_epochs):
            if max_training_hours and (time.time() - _t0) / 3600 >= max_training_hours:
                print(f'Wall-clock limit {max_training_hours}h reached, stopping.')
                break
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

            if _wandb is not None and _wandb.run is not None:
                _wandb.log({
                    "epoch":       epoch + 1,
                    "train_loss":  train_loss,
                    "val_pearson": current_pc,
                    "val_rmse":    current_rmse,
                })

        return history

    def predict(self, test_loader):
        """
        Make predictions on test set.

        Automatically detects Bayesian models. For Bayesian models, returns
        variance as a third element in the tuple.

        Args:
            test_loader: DataLoader for test data

        Returns:
            tuple: (true_values, predictions) both denormalized
                   For Bayesian models: (true_values, predictions, variances)
        """
        self.model.eval()
        total_preds = torch.Tensor()
        total_vars = torch.Tensor()
        total_labels = torch.Tensor()
        is_bayesian = False

        print('Make prediction for {} samples...'.format(len(test_loader.dataset)))

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)

                with autocast('cuda', enabled=self.use_amp):
                    output = self.model(data)

                if self.model.is_bayesian:
                    is_bayesian = True
                    total_preds = torch.cat((total_preds, output.predictive.mean.cpu()), 0)
                    total_vars  = torch.cat((total_vars,  output.predictive.variance.cpu()), 0)
                elif isinstance(output, tuple):
                    # GATv2NetAleatoric: still collect variance for calibration metrics
                    is_bayesian = True
                    mean, var = output
                    total_preds = torch.cat((total_preds, mean.cpu()), 0)
                    total_vars  = torch.cat((total_vars,  var.cpu()), 0)
                else:
                    total_preds = torch.cat((total_preds, output.cpu()), 0)

                total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

        # Denormalize predictions and labels
        y_true = denormalize(total_labels, self.y_scaler)
        y_pred = denormalize(total_preds, self.y_scaler)

        if is_bayesian:
            y_var = denormalize_variance(total_vars, self.y_scaler)
            return y_true, y_pred, y_var

        return y_true, y_pred


# ==================== High-level training pipeline ====================

def train_model(
    hp_config,
    *,
    dataset: str,
    device,
    num_workers: int = 0,
    epochs: int = Config.NUM_EPOCHS,
    batch_size: int = Config.BATCH_SIZE,
    model_type: str = Config.MODEL_NAME,
    run_name: str,
    wandb_run=None,
    base_model_dir=None,
    max_training_hours=None,
    archetype=None,
):
    """Train one model and log val metrics.  Returns the trained model directory.

    hp_config supports attribute access and provides: hidden_dim, num_layers,
    head, lr, weight_decay, activation_function, seed.  Missing attributes fall
    back to Config defaults, so both argparse.Namespace and wandb.config work
    without an adapter.
    """
    import json
    import pickle
    import random
    from aev_plig.datasets import init_weights, load_split
    from aev_plig.models import get_model

    seed = getattr(hp_config, 'seed', 42)
    if seed is None:
        seed = 42
    random.seed(seed)
    torch.manual_seed(int(seed))

    train_data = load_split(dataset, 'train')
    valid_data = load_split(dataset, 'valid')

    scaler_path = _PROJECT_ROOT / 'data' / 'processed' / dataset / 'scaler.pickle'
    legacy_scaler = _PROJECT_ROOT / 'data' / 'processed' / f'{dataset}_scaler.pickle'
    with open(scaler_path if scaler_path.exists() else legacy_scaler, 'rb') as f:
        y_scaler = pickle.load(f)

    num_node_features = train_data[0].x.shape[1]
    num_edge_features = train_data[0].edge_attr.shape[1]

    # Thin proxy so Bayesian models can read dataset_size from config
    # without mutating hp_config (wandb.config is read-only).
    class _ModelConfig:
        def __init__(self, base, dataset_size):
            self._base = base
            self.dataset_size = dataset_size

        def __getattr__(self, name):
            return getattr(self._base, name)

    model_config = _ModelConfig(hp_config, dataset_size=len(train_data))

    freeze_gnn         = getattr(hp_config, 'freeze_gnn',         False)
    lr_body            = getattr(hp_config, 'lr_body',            None)
    lr_head            = getattr(hp_config, 'lr_head',            None)
    if base_model_dir is None:
        base_model_dir = getattr(hp_config, 'base_model_dir', None)
    if max_training_hours is None:
        max_training_hours = getattr(hp_config, 'max_training_hours', None)
    if archetype is None:
        archetype = getattr(hp_config, 'archetype', None)

    model = get_model(
        model_type,
        node_feature_dim=num_node_features,
        edge_feature_dim=num_edge_features,
        config=model_config,
    )
    if base_model_dir:
        ckpts = sorted(Path(base_model_dir).glob('model_seed_*.model'))
        if not ckpts:
            raise FileNotFoundError(f'No model_seed_*.model files found in {base_model_dir}')
        ckpt = ckpts[int(seed) % len(ckpts)]
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f'Loaded base checkpoint: {ckpt}')
    else:
        model.apply(init_weights)
    model.to(device)

    output_dir = _PROJECT_ROOT / 'output' / 'trained_models' / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        'model':               model_type,
        'hidden_dim':          getattr(hp_config, 'hidden_dim',          Config.HIDDEN_DIM),
        'head':                getattr(hp_config, 'head',                 Config.NUM_ATTENTION_HEADS),
        'activation_function': getattr(hp_config, 'activation_function', Config.ACTIVATION_FUNCTION),
        'num_layers':          getattr(hp_config, 'num_layers',           Config.NUM_GNN_LAYERS),
        'lr':                  getattr(hp_config, 'lr',                   Config.LEARNING_RATE),
        'weight_decay':        getattr(hp_config, 'weight_decay',         Config.WEIGHT_DECAY),
        'seed':                seed,
        'dataset':             dataset,
        'epochs':              epochs,
        'dataset_size':        len(train_data),
        'node_feature_dim':    num_node_features,
        'edge_feature_dim':    num_edge_features,
        'run_name':            run_name,
        'freeze_gnn':          freeze_gnn,
        'lr_body':             lr_body,
        'lr_head':             lr_head,
        'dropout':             getattr(hp_config, 'dropout', 0.0),
        'archetype':           archetype,
        'base_model_dir':      str(base_model_dir) if base_model_dir else None,
        'max_training_hours':  max_training_hours,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        y_scaler=y_scaler,
        learning_rate=getattr(hp_config, 'lr',           Config.LEARNING_RATE),
        weight_decay=getattr(hp_config,  'weight_decay', Config.WEIGHT_DECAY),
        lr_body=lr_body,
        lr_head=lr_head,
        freeze_gnn=freeze_gnn,
    )

    model_save_path = output_dir / f'model_seed_{seed}.model'
    trainer.fit(n_epochs=epochs, model_save_path=str(model_save_path),
                max_training_hours=max_training_hours)
    print(f'Saved: {model_save_path}')

    if wandb_run is not None:
        wandb_run.summary['val_pearson_r'] = trainer.best_pc

    return output_dir
