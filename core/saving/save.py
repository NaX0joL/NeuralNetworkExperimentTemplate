import os
from pathlib import Path
import pickle
from pprint import pformat
import json

import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from core.schema import ExperimentContext
from core.modules.plotter import Plotter, plot_Dataframe

from ..metrics.metrics import MetricsEvaluator



class SavingService():
    def __init__(self, experiment_context:ExperimentContext, mpkg_name:str, default_save_path:Path):
        self.experiment_context = experiment_context
        self.mpkg_name = mpkg_name
        self.default_save_path = default_save_path
        
        self.plot_num = 40
        
        return
    
    ### public functions
    
    def save(self, path:Path=None):
        path = self._resolve_path(path)
        folder_path = self._create_mpkg_folder(path)
        
        self._create_marker(folder_path)
        self._save_config(folder_path)
        self._save_model_state(folder_path)
        self._save_loss_figure(folder_path)
        self._save_plot_results(folder_path)
        self._save_metrics(folder_path)
        return
    
    def insert_marker_content(self, **kwargs):
        self.marker_content = kwargs
        return
    
    ### private helper functions
    
    def _resolve_path(self, path:Path) -> Path:
        if path is None:
            path = self.default_save_path
        else:
            path = path
        return path
    
    def _create_mpkg_folder(self, path:Path) -> Path:
        mpkg_folder_path = path / self.mpkg_name
        os.makedirs(name=mpkg_folder_path, exist_ok=True)
        return mpkg_folder_path
    
    
    
    def _create_marker(self, path:Path) -> None:
        marker_file_path = path / "__mpkg__.py"
        with open(marker_file_path, "w") as file:
            if hasattr(self, "marker_content"):
                for key, value in self.marker_content.items():
                    file.write(f"{key}={repr(value)}\n")
        return
    
    
    
    def _save_config(self, path:Path) -> None:
        self._save_config_txt(path)
        self._save_config_txt(path)
        return
    
    def _save_config_txt(self, path:Path) -> None:
        config_txt_path = path / "config.txt"
        with open(config_txt_path, 'w') as file:
            file.write(pformat(
                self.experiment_context.master_config, 
                indent = 2, 
                width = 120,
            ))
        return
    
    def _save_config_pkl(self, path:Path) -> None:
        config_pickle_path = path / "config.pkl"
        with open(config_pickle_path, 'wb') as file:
            pickle.dump(
                self.experiment_context.master_config, 
                file,
            )
        return
    
    
    
    def _save_model_state(self, path:Path) -> None:
        model_state_path = path / "model.pth"
        torch.save(
            self.experiment_context.model.state_dict(), 
            model_state_path,
        )
        return
    
    
    
    def _save_loss_figure(self, path:Path):
        self._save_train_loss(path)
        self._save_validation_loss(path)
        return
    
    def _save_train_loss(self, path:Path):
        train_log_df = pd.DataFrame(self.experiment_context.loss_log["train"])
        plot_Dataframe(train_log_df, show_plot=False, savefig=True, plot_name="loss_train.png", save_path=path)
        return
    
    def _save_validation_loss(self, path:Path):
        val_log_df = pd.DataFrame(self.experiment_context.loss_log["validation"])
        plot_Dataframe(val_log_df, show_plot=False, savefig=True, plot_name="loss_validation.png", save_path=path)
        return
    
    
    
    def _save_plot_results(self, path:Path) -> None:
        self._save_plot(path, type="train")
        self._save_plot(path, type="validation")
        return
    
    def _save_plot(self, path:Path, type:str):
        plot_data = self._resolve_plot_data(type)
        device = self._resolve_model_device()
        plotter = Plotter()
        cnt = 0
        
        for batch in plot_data["dataloader"]:
            
            if cnt >= self.plot_num:
                break
            
            input = batch["value"].float().to(device)
            label = batch["ground_truth"].float().to(device)
            output = self._get_model_output(input)
            
            for index in range(input.shape[0]):
                
                if cnt >= self.plot_num:
                    break
                
                data = [input[index], label[index], output[index]]
                plotter.insert_data(*data, preview_plot=False)
                cnt += 1
                
        plotter.port_to_pdf(file_name=f"{plot_data["filename"]}.pdf", save_path=path)
        return
    
    def _resolve_plot_data(self, type:str) -> None:
        if type == "train":
            plot_data = {
                "dataloader": self.experiment_context.data_module.train_dataloader,
                "filename": "plot_train"
            }
        elif type == "validation":
            plot_data = {
                "dataloader": self.experiment_context.data_module.test_dataloader,
                "filename": "plot_validation"
            }
        return plot_data
    
    def _resolve_model_device(self) -> torch.device:
        return next(self.experiment_context.model.parameters()).device
    
    def _get_model_output(self, input:Tensor) -> Tensor:
        self.experiment_context.model.eval()
        with torch.no_grad():
            prediction = self.experiment_context.model(input)
        return prediction
    
    
    
    def _save_metrics(self, path:Path):
        metrics_result = self._calculate_metrics()
        metrics_path = path / "metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as file:
            json.dump(metrics_result, file, indent=4)
            
    def _calculate_metrics(self) -> dict[str, dict]:
        metrics_evaluator = MetricsEvaluator(self.experiment_context)
        metrics_result = metrics_evaluator.calculate()
        return metrics_result