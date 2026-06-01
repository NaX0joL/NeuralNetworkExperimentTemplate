import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor

from core.datasets.data_module import DataModule



class RegressionMetrics():
    """
    Class to hold input data and ground truth data to be evaluated
    """
    def __init__(self, model:nn.Module, data_module:DataModule, tolerance:int=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.data_module = data_module
        
        # list to hold input and gt role model with the shape [bs, seq_len] each
        self.expected_last_dim = None
        
        # tolerance: sideway extension of positive region in gt
        self.tolerance = tolerance 
        
        # init all components to zero 
        self.reset_components()
        
        return
        
    def reset_components(self):
        # dict to contain raw components of each metric
        self.raw_components = {
            "max_accuracy": {
                "true": 0,
                "false": 0,
            },
            "f_score": {
                "TP": 0,
                "TN": 0,
                "FP": 0,
                "FN": 0,
            }
        }
        
        return
    
    def calculate_metrics(self) -> dict[str, float]:
        self.reset_components()
        
        # might use validation data in the future
        _, validation_dataloader, test_dataloader = self.data_module.unwrap()
        used_dataloader = test_dataloader
        
        self.model.eval()
        with torch.no_grad():
            for data_batch in used_dataloader:
                input = data_batch.get('value').to(self.device)
                label = data_batch.get('ground_truth').to(self.device)
                
                output = self.model(input)
                
                # quick output norm
                min_val = output.min(dim=-1, keepdim=True)[0]
                max_val = output.max(dim=-1, keepdim=True)[0]
                eps = 1e-8
                output = (output - min_val) / (max_val - min_val + eps)
                
                self._process_max_accuracy_components(output, label)
                self._process_f_score_components(output, label)
        
        return {
            'max_accuracy': self.get_max_accuracy(),
            'f_score': self.get_f_score(),
        }
    
    ######################################################################################################################################
    
    def _handle_input_type(self, input):
        """
        > turned input into tensor
        """
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
        elif isinstance(input, pd.DataFrame):
            input = torch.from_numpy(input.values)
        return input
    
    def _check_shape_validity(self, input:torch.Tensor, ground_truth:torch.Tensor):
        """
        > check if input and gt have dim 1 or 2
        > also check if their shape match each other
        > also check if new input shape match their last one
        """
        condition_1 = input.dim() in (1, 2) and ground_truth.dim() in (1, 2)
        condition_2 = input.shape == ground_truth.shape
        assert condition_1 and condition_2, f"tensor shape mismatch, input:{input.shape} and gt:{ground_truth.shape}, expected: [bs, seq_len]"
        if self.expected_last_dim is not None:
            condition_3 = input.shape[-1] == self.expected_last_dim
            assert condition_3, f"new tensor shape mismatch, new input:{input.shape} and expected:{self.expected_last_dim}"
            condition_4 = ground_truth.shape[-1] == self.expected_last_dim
            assert condition_4, f"new tensor shape mismatch, new gt:{ground_truth.shape} and expected:{self.expected_last_dim}"
        return
    
    def _normalize_along_last_dim(self, input:torch.Tensor, eps=1e-08):
        """
        > normalize tensor's element in last dim into value between 0 and 1 
        """
        min_val = input.min(dim=-1, keepdim=True)[0]
        max_val = input.max(dim=-1, keepdim=True)[0]
        normed = (input - min_val) / (max_val - min_val + eps)
        return normed
    
    def _enlarge_ground_truth(self, ground_truth, tolerance):
        if tolerance == 0:
            return ground_truth
        
        enlarged = ground_truth.clone()
        bs, seq_len = ground_truth.shape
        
        # Process each sequence in the batch
        for i in range(bs):
            positive_positions = torch.where(ground_truth[i] == 1)[0]
            
            # Extend each positive position by tolerance
            for pos in positive_positions:
                start = max(0, pos - tolerance)
                end = min(seq_len, pos + tolerance + 1)
                enlarged[i, start:end] = 1
        
        return enlarged
    
    ######################################################################################################################################
    
    def _process_max_accuracy_components(self, input, ground_truth, eps=1e-08):
        """
        > process input and get max accuracy raw components
        """
        # create mask that gives True if the value is equal to the highest in the sequence
        # mask for input
        input_max_val = input.max(dim=-1, keepdim=True)[0]
        input_mask = torch.isclose(input, input_max_val, rtol=0, atol=eps)
        # mask for ground truth 
        ground_truth_max_val = ground_truth.max(dim=-1, keepdim=True)[0]
        ground_truth_mask = torch.isclose(ground_truth, ground_truth_max_val, rtol=0, atol=eps)
        
        # create new mask that finds True component on both input and gt in the same time step 
        intersection_mask = input_mask & ground_truth_mask
        
        # check if the given sequence has at least one True component in the previous mask
        row_has_match = intersection_mask.any(dim=1)
        
        # calculation
        true = row_has_match.sum().item()
        false = row_has_match.shape[0] - true
        
        # add to global raw components
        self.raw_components["max_accuracy"]["true"] += true
        self.raw_components["max_accuracy"]["false"] += false
    
    def _process_f_score_components(self, input, ground_truth, threshold=0.5):
        """
        > process input and get f-score raw components
        """
        # masked out all values that are above threshold into True, else False
        input_mask = (input >= threshold).int()
        ground_truth_mask = (ground_truth >= threshold).int()
        
        # get each component by summing the True on each given condition
        TP = ((input_mask == 1) & (ground_truth_mask == 1)).sum().item()
        TN = ((input_mask == 0) & (ground_truth_mask == 0)).sum().item()
        FP = ((input_mask == 1) & (ground_truth_mask == 0)).sum().item()
        FN = ((input_mask == 0) & (ground_truth_mask == 1)).sum().item()
        
        # add to global raw components
        self.raw_components["f_score"]["TP"] += TP
        self.raw_components["f_score"]["TN"] += TN
        self.raw_components["f_score"]["FP"] += FP
        self.raw_components["f_score"]["FN"] += FN
    
    ######################################################################################################################################
    
    def get_max_accuracy(self, eps=1e-08):
        """
        get the max accuracy and the raw calculation components by comparing the given input and given ground truth
        input:
            - input: input data to compare, shape: [bs, seq_len]
            - ground_truth: base data for comparison, shape: [bs, seq_len]
            - eps: small number to prevent zero divison
        output:
            - dict of keys: {
                - max_accuracy:
                - true:
                - false:
            }
        notes:
            > max accuracy is a derivation of regular accuracy where it considers a prediction correct when
            the highest value of input resides within the positive region of the ground truth
        """
        # calculation
        true = self.raw_components['max_accuracy']['true']
        false = self.raw_components['max_accuracy']['false']
        max_accuracy = true / (true + false + eps)
        
        output = {
            "max_accuracy": max_accuracy,
            "true": true,
            "false": false,
        }
        return output
    
    def get_f_score(self, threshold=0.5, beta=1, eps=1e-08):
        """
        get the generalized f score and the raw calculation components by comparing the given input and given ground truth
        input:
            - input: input data to compare, shape: [bs, seq_len]
            - ground_truth: base data for comparison, shape: [bs, seq_len]
            - threshold: the cutoff value to be considered as true prediction
            - beta: weight constant for a generalized f-score
            - eps: small number to prevent zero divison
        output:
            - dict of keys: {
                - f_score:
                - precision:
                - recall:
                - TP:
                - TN:
                - FP:
                - FN:
            }
        notes:
            > if you are not familiar with f-score, you can just google it
            > default beta = 1, that is the f1-score
        """
        # calculation
        TP = self.raw_components['f_score']['TP']
        TN = self.raw_components['f_score']['TN']
        FP = self.raw_components['f_score']['FP']
        FN = self.raw_components['f_score']['FN']
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f_score = (1 + beta*beta)*TP / ((1 + beta*beta)*TP + (beta*beta)*FN + FP + eps)
        
        output = {
            "f_score": f_score,
            "precision": precision,
            "recall": recall,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
        }
        return output
    
    def insert_data(self, input, ground_truth):
        """
        input:
            - input: data for comparison, shape: [bs, seq_len] or [seq_len]
            - ground_truth: data to compare with, shape: [bs, seq_len] or [seq_len]
        output:
            - None
        notes:
            - when inserting input and ground_truth, the shape must match each other
        """
        # type alignment
        input = self._handle_input_type(input)
        ground_truth = self._handle_input_type(ground_truth)
        
        # sanity check for shape
        self._check_shape_validity(input, ground_truth) 
        
        # normalize value into range:[0, 1]
        input = self._normalize_along_last_dim(input)
        ground_truth = self._normalize_along_last_dim(ground_truth)
        
        # force into shape [bs, seq_len]
        if input.dim() == 1: input = input.unsqueeze(0)
        if ground_truth.dim() == 1: ground_truth = ground_truth.unsqueeze(0)
        
        # add the first input for shape role model for the next input
        if self.expected_last_dim is None: self.expected_last_dim = input.shape[-1]
        
        # enlarge gt if tolerance value is applied 
        ground_truth = self._enlarge_ground_truth(ground_truth, tolerance=self.tolerance)
        
        # pass input and gt thru each metric processes
        self._process_max_accuracy_components(input, ground_truth)
        self._process_f_score_components(input, ground_truth)
        
        return



def _handle_input_type(input):
    if isinstance(input, np.ndarray): input = pd.DataFrame(input)
    if isinstance(input, pd.DataFrame): input = torch.tensor(input)
    return input

def _check_shape_validity(input:torch.Tensor, ground_truth:torch.Tensor):
    condition_1 = input.dim() == 2
    condition_2 = ground_truth.dim() == 2
    condition_3 = input.shape == ground_truth.shape
    assert condition_1 and condition_2 and condition_3, f"tensor shape mismatch, input:{input.shape} and gt:{ground_truth.shape}"
    return

def _normalize_along_last_dim(input, eps=1e-08):
    min_val = input.min(dim=-1, keepdim=True)[0]
    max_val = input.max(dim=-1, keepdim=True)[0]
    normed = (input - min_val) / (max_val - min_val + eps)
    return normed

def input_sanitation(func):
    """
    <UNDER DEVELOPMENT>
    decorator to sanitize input and ground truth of metric function before calculation
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper

@input_sanitation
def get_f_score(input:torch.Tensor, ground_truth:torch.Tensor, threshold=0.5, beta=1, eps=1e-08) -> dict:
    """
    get the generalized f score and the raw calculation components by comparing the given input and given ground truth
    input:
        - input: input data to compare, shape: [bs, seq_len]
        - ground_truth: base data for comparison, shape: [bs, seq_len]
        - threshold: the cutoff value to be considered as true prediction
        - beta: weight constant for a generalized f-score
        - eps: small number to prevent zero divison
    output:
        - f_score:
        - precision:
        - recall:
        - TP:
        - TN:
        - FP:
        - FN:
    notes:
        > default beta = 1, that is the f1-score
        > input and ground truth must be of the same shape, that is [bs, seq_len]
        > outputs are stored in a dict
    """
    # handles dataframe input
    input = _handle_input_type(input)
    ground_truth = _handle_input_type(ground_truth)
    
    # check shape validity
    _check_shape_validity(input, ground_truth)
    
    # input normalization
    input = _normalize_along_last_dim(input)
    ground_truth = _normalize_along_last_dim(ground_truth)
    
    # masked out all values that are above threshold
    input_mask = (input >= threshold).int()
    ground_truth_mask = (ground_truth >= threshold).int()
    
    # check mask for each raw component
    TP = ((input_mask == 1) & (ground_truth_mask == 1)).sum().item()
    TN = ((input_mask == 0) & (ground_truth_mask == 0)).sum().item()
    FP = ((input_mask == 1) & (ground_truth_mask == 0)).sum().item()
    FN = ((input_mask == 0) & (ground_truth_mask == 1)).sum().item()

    # calculation
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f_score = (1 + beta*beta)*TP / ((1 + beta*beta)*TP + (beta*beta)*FN + FP + eps)
    
    output = {
        "f_score": f_score,
        "precision": precision,
        "recall": recall,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }
    return output

@input_sanitation
def get_max_accuracy(input:torch.Tensor, ground_truth:torch.Tensor, eps=1e-08) -> dict:
    """
    get the max accuracy and the raw calculation components by comparing the given input and given ground truth
    input:
        - input: input data to compare, shape: [bs, seq_len]
        - ground_truth: base data for comparison, shape: [bs, seq_len]
        - eps: small number to prevent zero divison
    output:
        - max_accuracy:
        - true:
        - false:
    notes:
        > max accuracy regards the max value in the input as the 'answer' and 
        checks if its inside the positive region in ground trtuh
        > input and ground truth must be of the shape, that is [bs, seq_len]
        > outputs are stored in a dict
    """
    # handles dataframe input
    input = _handle_input_type(input)
    ground_truth = _handle_input_type(ground_truth)
    
    # check shape validity
    _check_shape_validity(input, ground_truth)
    
    # input normalization
    input = _normalize_along_last_dim(input)
    ground_truth = _normalize_along_last_dim(ground_truth)
    
    # masked out the highest value along the first dimension
    # create mask for input 
    input_max_val = input.max(dim=-1, keepdim=True)[0]
    input_mask = torch.isclose(input, input_max_val, rtol=0, atol=eps)
    
    # create mask for ground truth 
    ground_truth_max_val = ground_truth.max(dim=0, keepdim=True)[0]
    ground_truth_mask = torch.isclose(ground_truth, ground_truth_max_val, rtol=0, atol=eps)
    
    # Find intersection and check rows
    intersection_mask = input_mask & ground_truth_mask
    row_has_match = intersection_mask.any(dim=1)
    
    # calculation
    true = row_has_match.sum().item()
    false = row_has_match.shape[0] - true
    max_accuracy = true / (true + false + eps)
    
    output = {
        "max_accuracy": max_accuracy,
        "true": true,
        "false": false,
    }
    return output