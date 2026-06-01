from dataclasses import dataclass, is_dataclass

from core.abstract import ABSTRACT_Config

# from .architectures.config import ArchitectureConfig, DummyModelAConfig, ProposedModelConfig, UNetConfig
# from .datasets.config import DatasetConfig, UCR_Anomaly_Detection_Config, UCR_Classification_Config
# from .trainer.config import TrainerConfig, LossConfig

from .architectures.proposed_model.new_model import ProposedModelConfig
from .architectures.architecture_config import ArchitectureConfig
from .architectures.registry import ARCHITECTURE_REGISTRY

from .datasets.dataset_config import DatasetConfig
from .datasets.UCR_Classification.new_raw_getter import UCR_Classification_Config
from .datasets.registry import DATASET_REGISTRY

from .trainer.trainer_config import TrainerConfig



HIERARCHY = {
    "master_config": {
        "architecture_config": {
            "model_config": None,
        },
        
        "dataset_config": {
            "raw_getter_config": None,
        },
        
        "trainer_config": {
            "loss_config": None,
        },
    }    
}



@dataclass
class MasterConfig():
    architecture_config: ArchitectureConfig
    dataset_config: DatasetConfig
    trainer_config: TrainerConfig
    
    @classmethod
    def default(self):
        architecture_config = ArchitectureConfig.default()
        dataset_config = DatasetConfig.default()
        trainer_config = TrainerConfig.default()
        
        master_config = self(
            architecture_config = ArchitectureConfig.default(),
            dataset_config = DatasetConfig.default(),
            trainer_config = TrainerConfig.default(),
        )
        return master_config
    
    @classmethod
    def hardcoded_config(self):
        master_config = self(
            
            ### --- architecture config ---
            
            # architecture_config = ArchitectureConfig(
            #     architecture_name = 'dummy_model_A',
            #     arch_specific_config = DummyModelAConfig(
            #         input_feature = 500,
            #         hidden_size = 10,
            #         num_classes = 2,
            #     ),
                
            #     architecture_name = 'proposed_model',
            #     arch_specific_config = None,
            # ),
            
            architecture_config = ArchitectureConfig(
                architecture_name = "proposed_model",
                arch_specific_config = ARCHITECTURE_REGISTRY["proposed_model"].config_class(
                    seq_len = 1000,
                    pred_len = 1000,
                    patch_len = 100,
                    stride = 1,
                    
                    e_layers_num = 2,
                    enc_in_feature = 1,
                    d_layers_num = 1,
                    dec_in_feature = 1,
                    
                    n_heads_num = 4,
                    n_normal_heads = 2,
                    n_mp_attn_heads = 2,
                    qk_weight_share = False,
                    d_model = 256,
                    d_ff = 512,
                    
                    dropout = 0.5,
                    fc_dropout = 0.3,
                    head_dropout = 0.1,
                    attn_dropout = 0.1,
                    
                    use_pre_norm = False
                )
            ),
            
            ### --- dataset config ---
            
            dataset_config = DatasetConfig(
                dataset_name = "UCR_Anomaly_Detection",
                raw_getter_config = DATASET_REGISTRY["UCR_Anomaly_Detection"].config_class(
                    variant_name = "size-1000",
                    grouping_file = "designated_grouping",
                    group = "AirTemperature",
                ),
                
                batch_size = 16,
                shuffle = False,
                drop_last = False,
            ),
            
            ### --- trainer config ---
            
            trainer_config = TrainerConfig(
                train_epochs = 1,
                
                learning_rate = 1e-4,
                max_learning_rate = 1e-3,
                percentage_start = 0.3,
                
                weight_decay = 1e-4,
                grad_clip_max_norm = 1.0,
                
                optimizer_name = 'adamw',
                criterion_name = 'MSE',
                
                use_best_model = True,
            ),
            
        )
        return master_config