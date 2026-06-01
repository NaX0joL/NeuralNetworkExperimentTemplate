import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .schema import TrainerComponents, TrainerState


class EpochProcessor():
    EPOCH_TYPE = [
        "train",
        "validation",
    ]
    
    def __init__(self, trainer_components:TrainerComponents) -> None:
        self.trainer_components = trainer_components
        return
    
    ### --- public functions ---
    
    def process_epoch(self, epoch_type:str, dataloader:DataLoader) -> None:
        self._validate_epoch_type(epoch_type)
        self._change_model_mode(epoch_type)
        
        total_loss = self._get_batches_loss(epoch_type, dataloader)
        average_loss = total_loss / len(dataloader)
        
        self._update_log(epoch_type, average_loss)
        return
    
    ### --- private helper functions ---
    
    def _validate_epoch_type(self, epoch_type:str) -> None:
        if not epoch_type in self.EPOCH_TYPE:
            raise TypeError("trainer error! invalid epoch type")
        return
    
    def _change_model_mode(self, epoch_type:str):
        if epoch_type == "train":
            self.trainer_components.model.train()
        elif epoch_type == "validation":
            self.trainer_components.model.eval()
        return
    
    def _get_batches_loss(self, epoch_type:str, dataloader:DataLoader) -> float:
        total_loss = 0.0
        
        with self._get_gradient_context(epoch_type):
            for batch in dataloader:
                batch_loss = self._get_single_batch_loss(batch, epoch_type)
                total_loss += batch_loss
        
        return total_loss
    
    def _get_gradient_context(self, epoch_type:str):
        if epoch_type == "train":
            gradient_context = torch.enable_grad()
        elif epoch_type == "validation":
            gradient_context = torch.no_grad()
        return gradient_context
    
    def _get_single_batch_loss(self, batch:dict[str, Tensor|list], epoch_type:str) -> float:
        if epoch_type == "train":
            self.trainer_components.optimizer.zero_grad()
        
        input = batch["value"].to(self.trainer_components.device)
        label = batch["ground_truth"].to(self.trainer_components.device)
        
        output = self.trainer_components.model(input)
        loss = self.trainer_components.loss_manager.forward(output, label)
        
        if epoch_type == "train":
            self._apply_gradients(loss)
            
        return loss.item()
    
    def _apply_gradients(self, loss:Tensor) -> None:
        loss.backward()
        self._clip_gradients()
        self.trainer_components.optimizer.step()
        self.trainer_components.scheduler.step()
        return
    
    def _clip_gradients(self) -> None:
        torch.nn.utils.clip_grad_norm_(
            self.trainer_components.model.parameters(),
            max_norm = self.trainer_components.config.grad_clip_max_norm,
        )
        return     
    
    def _update_log(self, epoch_type:str, loss:float) -> None:
        if epoch_type == "train":
            self.trainer_components.loss_log["train"].append(loss)
        if epoch_type == "validation":
            self.trainer_components.loss_log["validation"].append(loss)
        return