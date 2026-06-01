import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())

from core.new_experiment import Experiment
from core.master_config import MasterConfig

from core.architectures.architecture_config import ArchitectureConfig
from core.architectures.registry import ARCHITECTURE_REGISTRY

from core.datasets.dataset_config import DatasetConfig
from core.datasets.registry import DATASET_REGISTRY

from core.trainer.trainer_config import TrainerConfig



def main() -> None:
    architecture_name = "proposed_model"
    dataset_name = "UCR_Anomaly_Detection"
    
    config = MasterConfig(
        architecture_config = ArchitectureConfig(
            architecture_name = architecture_name,
            arch_specific_config = ARCHITECTURE_REGISTRY[architecture_name].config_class(
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
        
        dataset_config = DatasetConfig(
            dataset_name = dataset_name,
            raw_getter_config = DATASET_REGISTRY[dataset_name].config_class(
                variant_name = "size-1000",
                grouping_file = "designated_grouping",
                group = "AirTemperature",
            ),
            
            batch_size = 16,
            shuffle = False,
            drop_last = False,
        ),
        
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
        )
    )
    exp = Experiment(config)
    exp.train_model(timer=True)
    exp.save_to_mpkg()
    return



if __name__ == "__main__":
    main()
    print("Done!")