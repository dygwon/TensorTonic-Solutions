import torch
import torch.nn as nn

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Returns the mean batch loss as a Python float.
    """
    losses = []
    
    model.train()
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()

    return sum(losses) / len(losses)