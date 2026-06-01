from __future__ import annotations

import os
import json
import pickle
from pprint import pformat
from datetime import datetime
from dataclasses import dataclass, asdict

import pandas as pd
import torch
from torch import nn

#from core.experiment import Experiment
from core.master_config import MasterConfig
from core.architectures.architecture_control import ArchitectureControl
from .classification_metrics import ClassificationMetrics
from .plotter import Plotter, plot_Dataframe



@dataclass
class MPKGConfig():
    mpkg_save_path: str
    
    plot_format:str = None  # either png or pdf
    train_plot_num: int = 5
    test_plot_num: int = 5
    
    @classmethod
    def default(self) -> MPKGConfig:
        mpkg_config = self(
            mpkg_save_path = "core/savefolder/mpkg/tmp",
            plot_format = "pdf",
            train_plot_num = 20,
            test_plot_num = 20,
        )
        return mpkg_config



class MPKGCreator():
    def __init__(self):
        self.mpkg_config: MPKGConfig = MPKGConfig.default()
        self.default_save_path = self.mpkg_config.mpkg_save_path
        
        self.set_mpkg_name()
        
        return
        
    ### public functions
    
    def insert_experiment(self, exp) -> None:
        self.exp = exp
        
        self.set_mpkg_name(exp.experiment_id)
        self.insert_model(exp.model)
        self.insert_config(exp.master_config)
        self.insert_loss_log(exp.train_loss_log, exp.validation_loss_log)
        self.insert_metric_result(exp.get_model_metrics())
        
        return
    
    def set_mpkg_name(self, mpkg_name:str=None):
        if mpkg_name is None:
            mpkg_name = datetime.now().strftime("mpkg-%Y-%m-%d_%H%M%S")
        else:
            mpkg_name = "mpkg-" + mpkg_name
        self.mpkg_name = mpkg_name
        return
    
    def insert_model(self, model):
        self.model: nn.Module = model
        return
    
    def insert_config(self, config):
        self.config: MasterConfig = config
        return
    
    def insert_loss_log(self, train_loss_log=None, validation_loss_log=None):
        if train_loss_log is not None:
            self.train_loss_log: list = train_loss_log
        if validation_loss_log is not None:
            self.validation_loss_log: list = validation_loss_log
        return
    
    def insert_metric_result(self, metric_result):
        self.metric_result = metric_result
        return
    
    def insert_others(self, other_data):
        pass
    
    def save_mpkg(self, save_path=None, verbose=True):
        mpkg_folder_path = self._createSaveDirectory(save_path=save_path)
        
        self._createMarkerFile(mpkg_path=mpkg_folder_path)
        self._saveModelStatePTH(mpkg_path=mpkg_folder_path)
        self._saveConfigPicklePprint(mpkg_path=mpkg_folder_path)
        self._saveLossPNG(mpkg_path=mpkg_folder_path)
        self._saveMetricJSON(mpkg_path=mpkg_folder_path)
        self._saveResultPlot(mpkg_path=mpkg_folder_path)
        # save others as txt/json(?)
        
        if verbose:
            print(f"mpkg successfully saved to {mpkg_folder_path}!")
        return
    
    def load_mpkg(self, load_path, verbose=False) -> None:
        self._loadConfigPickle(config_path=f'{load_path}/config.pkl')
        self._loadModelStatePTH(model_path=f'{load_path}/model.pth')
        
        if verbose:
            print(f"mpkg successfully loaded from {load_path}!")
        return
    
    def update_mpkg(self, mpkg_path):
        self.load_mpkg(load_path=mpkg_path)
        
        self._saveMetricJSON(mpkg_path)
        self._saveResultPlot(mpkg_path)
        
        return
    
    ### private helper functions
    
    def _createSaveDirectory(self, save_path) -> None:
        if save_path is None:
            mpkg_folder_path = f'{self.default_save_path}/{self.mpkg_name}'
        else:
            mpkg_folder_path = f'{save_path}/{self.mpkg_name}'
        os.makedirs(name=mpkg_folder_path, exist_ok=True)
        
        return mpkg_folder_path
    
    def _createMarkerFile(self, mpkg_path):
        # marker file to indicate that it is a "special" folder
        marker_file_path = f"{mpkg_path}/__mpkg__.py"
        with open(marker_file_path, "w"):
            pass
        
        return
    
    def _saveModelStatePTH(self, mpkg_path):
        if hasattr(self, "model"):
            model_path = f'{mpkg_path}/model.pth'
            torch.save(self.model.state_dict(), model_path)
        return
    
    @DeprecationWarning
    def _saveConfigJSON(self, mpkg_path):
        if hasattr(self, "config"):
            config_path = f'{mpkg_path}/config.json'
            with open(config_path, "w", encoding="utf-8") as file:
                dict = self.config.to_dict()
                json.dump(dict, file, indent= 4)
        return
    
    def _saveConfigPicklePprint(self, mpkg_path):
        if hasattr(self, "config"):
            # save in pickle format
            pickle_path = f'{mpkg_path}/config.pkl'
            with open(pickle_path, 'wb') as file:
                pickle.dump(self.config, file)
            
            # save in pprint format
            pprint_path = f'{mpkg_path}/config.txt'
            with open(pprint_path, 'w') as file:
                file.write(pformat(self.config, indent=2, width=120))
                
            return
    
    def _saveLossPNG(self, mpkg_path):
        if hasattr(self, "train_loss_log") and hasattr(self, "validation_loss_log") and False:
            pass
        else:
            if hasattr(self, "train_loss_log"):
                train_log_df = pd.DataFrame(self.train_loss_log)
                plot_Dataframe(train_log_df, show_plot=False, savefig=True, plot_name="loss_train.png", save_path=mpkg_path)
            if hasattr(self, "validation_loss_log"):
                val_log_df = pd.DataFrame(self.validation_loss_log)
                plot_Dataframe(val_log_df, show_plot=False, savefig=True, plot_name="loss_validation.png", save_path=mpkg_path)
        return
    
    def _saveMetricJSON(self, mpkg_path):
        if hasattr(self, "metric_result"):
            metric_path = f'{mpkg_path}/metric_result.json'
            with open(metric_path, 'w', encoding='utf-8') as file:
                json.dump(self.metric_result, file, indent=4)
        return
    
    def _saveResultPlot(self, mpkg_path):
        if self.mpkg_config.plot_format == 'pdf':
            self._saveResultPlotPDF(mpkg_path)
        elif self.mpkg_config.plot_format == 'png':
            self._saveResultPlotPNG(mpkg_path)
        else:
            raise TypeError(f"MPKGCreator error, invalid plot_format! current plot_format = {self.mpkg_config.plot_format}")
    
    def _saveResultPlotPDF(self, mpkg_path):
        if hasattr(self, "model"):
            train_plotter, test_plotter = Plotter(), Plotter()
            
            self.exp.model.eval()
            with torch.no_grad():
                
                # add plot for train
                train_plot_cnt = 0
                for data_batch in self.exp.data_module.train_dataloader:
                    if train_plot_cnt >= self.mpkg_config.train_plot_num:
                        break
                        
                    input = data_batch['value'].float().to(self.exp.device)
                    label = data_batch['ground_truth'].float().to(self.exp.device)
                    #output = self.exp.model(input)
                    output = self.exp.model_inference(input)
                    
                    for jndex in range(input.shape[0]):
                        if train_plot_cnt >= self.mpkg_config.train_plot_num:
                            break
                        
                        data = [input[jndex], label[jndex], output[jndex]]
                        train_plotter.insert_data(*data, preview_plot=False)
                        train_plot_cnt += 1
                
                # add plot for test
                test_plot_cnt = 0
                for data_batch in self.exp.data_module.test_dataloader:
                    if test_plot_cnt >= self.mpkg_config.test_plot_num:
                        break
                        
                    input = data_batch['value'].float().to(self.exp.device)
                    label = data_batch['ground_truth'].float().to(self.exp.device)
                    #output = self.exp.model(input)
                    output = self.exp.model_inference(input)
                    
                    for jndex in range(input.shape[0]):
                        if test_plot_cnt >= self.mpkg_config.test_plot_num:
                            break
                        
                        data = [input[jndex], label[jndex], output[jndex]]
                        test_plotter.insert_data(*data, preview_plot=False)
                        test_plot_cnt += 1
                
            train_plotter.port_to_pdf(file_name='plot_train.pdf', save_path=mpkg_path)
            test_plotter.port_to_pdf(file_name='plot_test.pdf', save_path=mpkg_path)
                
        else:
            raise TypeError(f"MPKGCreator error, empty model!")
        return
    
    def _saveResultPlotPNG():
        pass
    
    def _saveOthers(self):
        raise NotImplementedError
    
    def _loadModelStatePTH(self, model_path):
        self.model = ArchitectureControl(
            self.config.architecture_config
        ).get_model()
        model_state = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(model_state)
        return 
    
    @DeprecationWarning
    def _loadConfigJSON(self, config_path):
        with open(config_path, 'r') as file:
            dict = json.load(file)
            self.config = MasterConfig.from_dict(dict)
        return
    
    def _loadConfigPickle(self, config_path):
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)
        return