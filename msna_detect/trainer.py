from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Optional
from typing import Tuple
from typing import List
from typing import Dict
from typing import Any


def train_one_step(
    model: nn.Module, 
    batch: Tuple[torch.Tensor, torch.Tensor], 
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Dict:
    """Code for training the model one step."""
    x = batch[0].to(device)
    y = batch[1].to(device)

    optimizer.zero_grad(set_to_none = True)
    y_hat = model(x)
    loss = criterion(y_hat[:, :, 1024:-1024], y[:, :, 1024:-1024])
    loss.backward()
    optimizer.step()

    return {"loss": loss.item()}


@torch.no_grad()
def eval_one_step(
    model: nn.Module, 
    batch: Tuple[torch.Tensor, torch.Tensor], 
    criterion: torch.nn.modules.loss._Loss,
    device: str
) -> Dict:
    """Code for evaluating the model one step.
    """
    x = batch[0].to(device)
    y = batch[1].to(device)

    y_hat = model(x)
    loss = criterion(y_hat, y)

    return {"loss": loss.item()}


def train(
    model: nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    device: str = "cpu",
    min_epochs: int = 1,
    max_epochs: int = 50,
    check_val_every_n_epochs: int = 1,
    verbose: bool = True
) -> Dict[str, Any]:
    """Main training loop."""
    history: Dict[str, List[float]] = {"training": [], "validation": []}
    pbar = tqdm(range(min_epochs, max_epochs + 1), disable = not verbose)

    train_loss = 0.0
    valid_loss = 0.0

    model = model.to(device)
    for epoch in pbar:
        # Whether or not to every the validation loop. Always do validation first.
        if (epoch % check_val_every_n_epochs == 0) and (val_dataloader is not None):
            model.eval()
            valid_loss = 0.0
            for step, batch in enumerate(val_dataloader, 1):
                output = eval_one_step(model, batch, criterion, device)
                valid_loss += output["loss"]

            valid_loss = valid_loss / len(val_dataloader)
            history["validation"].append(valid_loss)

            pbar.set_postfix({
                "Epoch": epoch,
                "Train Loss": f"{train_loss:.4f}",
                "Valid Loss": f"{valid_loss:.4f}"
            })

        if epoch != max_epochs:
            model.train()
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader, 1):
                output = train_one_step(model, batch, criterion, optimizer, device)
                train_loss += output["loss"]

            train_loss = train_loss / len(train_dataloader)
            history["training"].append(train_loss)

            pbar.set_postfix({
                "Epoch": epoch,
                "Train Loss": f"{train_loss:.4f}",
                "Valid Loss": f"{valid_loss:.4f}"
            })

    return history


