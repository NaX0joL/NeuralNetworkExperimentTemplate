import torch
from torch import nn, Tensor

from core.datasets.data_module import DataModule



class ClassificationMetrics():
    def __init__(self, model:nn.Module, data_module:DataModule):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.data_module = data_module
        
        self.metrics = {
            "accuracy": Accuracy(),
        }
        return
    
    def calculate_metrics(self) -> dict[str, float]:
        
        # might use validation data in the future
        _, validation_dataloader, test_dataloader = self.data_module.unwrap()
        used_dataloader = test_dataloader
        self.model.eval()
        with torch.no_grad():
            for data_batch in used_dataloader:
                input = data_batch.get('value').to(self.device)
                label = data_batch.get('ground_truth').to(self.device)
                
                output = self.model(input)
                
                for _, metric in self.metrics.items():
                    metric.update(output, label)
            
        metric_results = {
            name: result.get_metric() for name, result in self.metrics.items()
        }
        return metric_results



class Accuracy():
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.reset()
        return
    
    ### public functions
    
    def reset(self) -> None:
        self.true: int = 0
        self.count: int = 0
        return
        
    def update(self, preds:Tensor, targets:Tensor) -> None:
        #print(f"preds: {preds.shape}, target: {targets.shape}")
        
        preds = self._convertLogitsToClass(preds)
        targets = targets.squeeze(-1)
        
        self.true += (preds == targets).sum().item()
        self.count += len(preds)
        return
    
    def get_metric(self) -> float:
        accuracy = self.true / self.count
        return accuracy
        
    ### private helper functions
    
    def _convertLogitsToClass(self, logits):
        max_val, class_indices = torch.max(logits, dim=1)
        return class_indices