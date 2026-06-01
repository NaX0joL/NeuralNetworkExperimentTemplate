from core.experiment import Experiment
from core.master_config import MasterConfig

from core.datasets.data_module import DataModuleGetter
from core.datasets.config import DatasetConfig

from core.datasets.UCR_Anomaly_Detection.config import UCR_Anomaly_Detection_Config
from core.datasets.TSB_AD_U.config import TSB_AD_U_Config

from core.architectures.config import ArchitectureConfig
from core.architectures.proposed_model.config import ProposedModelConfig
from core.architectures.simple_cnn.config import SimpleCNNConfig
from core.architectures.traditional_matrix_profile.config import TraditionalMatrixProfileConfig

from core.trainer.config import TrainerConfig, LossConfig



class TSB_AD_U_preset():
    dataset_name = 'TSB_AD_U'
    
    group_list = [
        'random_group_1',
        'random_group_2',
        'random_group_3',
        'random_group_4',
        'random_group_5',
    ]
    
    dataset_config = DatasetConfig(
        batch_size = 16,
        shuffle = True,
        drop_last = False,
        
        dataset_name = 'TSB_AD_U',
        task_type = 'regression',
        raw_getter_config = TSB_AD_U_Config(
            task_type = 'regression',
            variant_name = 'size-200',
            grouping_file = 'random_grouping',
            group = 'random_group_1',
        ),
    )



class UCR_Anomaly_Detection_preset():
    dataset_name = 'UCR_Anomaly_Detection'
    
    group_list = [
        'AirTemperature',
        'ECG',
        'Marker',
        'InternalBleeding',
        'PowerDemand',
    ]
    
    dataset_config = DatasetConfig(
        batch_size = 16,
        shuffle = True,
        drop_last = False,
        
        dataset_name = 'UCR_Anomaly_Detection',
        task_type = 'regression',
        raw_getter_config = UCR_Anomaly_Detection_Config(
            task_type = 'regression',
            variant_name = 'size-1000',
            grouping_file = 'designated_grouping',
            group = 'AirTemperature',
        ),
    )



class Model_preset():
    # architecture_config = ArchitectureConfig(
    #     architecture_name = 'proposed_model',
    #     arch_specific_config = ProposedModelConfig(
    #         seq_len = 1000,
    #         pred_len = 1000,
    #         patch_len = 100,
    #         stride = 1,
            
    #         e_layers_num = 2,
    #         enc_in_feature = 1,
    #         d_layers_num = 1,
    #         dec_in_feature = 1,
            
    #         n_heads_num = 4,
    #         n_normal_heads = 0,
    #         n_mp_attn_heads = 4,
    #         qk_weight_share = True,
    #         d_model = 256,
    #         d_ff = 512,
            
    #         dropout = 0.5,
    #         fc_dropout = 0.3,
    #         head_dropout = 0.1,
    #         attn_dropout = 0.1,
            
    #         use_pre_norm = False
    #     )
        
    # architecture_config = ArchitectureConfig(
    #     architecture_name = 'simple_cnn',
    #     arch_specific_config = SimpleCNNConfig(
    #         in_channels = [64, 128, 256, 512],
    #         kernel_sizes = [3, 3, 3, 3],
    #         paddings = [1, 1, 1, 1],
            
    #         linear_n_layer = 1,
    #         linear_dim = 1000,
    #     )
    # )
    
    architecture_config = ArchitectureConfig(
        architecture_name = 'traditional_matrix_profile',
        arch_specific_config = TraditionalMatrixProfileConfig(
            window_size = 50,
        )
    )



class Trainer_preset():
    trainer_config = TrainerConfig(
        train_epochs = 100,
        learning_rate = 1e-4,
        max_learning_rate = 1e-4,
        percentage_start = 0.3,
        weight_decay = 1e-4,
        grad_clip_max_norm = 1.0,
        
        optimizer_name = 'adamw',
        criterion_name = 'MAE',
        
        use_best_model = False,
        checkpoint_path = 'core/savefolder/checkpoint',
        
        loss_config = LossConfig(
            use_multi_loss = False,
            multi_loss_coeff = {
                "base_loss": 1,
                "matrix_profile_loss": 0,
            }
        )
    )


class QUICK_TRAINING():
    
    def __init__(self, dataset_preset, model_preset, trainer_preset):
        self.dataset_preset = dataset_preset
        self.model_preset = model_preset
        self.trainer_preset = trainer_preset
        
        self.mpkg_name_prefix = ""
        return
    
    def train_model(self):
        master_config = MasterConfig(
            architecture_config=self.model_preset.architecture_config,
            dataset_config=self.dataset_preset.dataset_config,
            trainer_config=self.trainer_preset.trainer_config,
        )
        
        for group in self.dataset_preset.group_list:
            master_config.trainer_config.train_epochs = 2
            master_config.dataset_config.raw_getter_config.group = group
            
            exp_id = f'{self.mpkg_name_prefix}-{self.dataset_preset.dataset_name}-{group}'
            #exp_id = f'quick_test_delete_after_plz'
            exp = Experiment(master_config, experiment_id=exp_id)
            
            exp.train_model()
            exp.save_to_mpkg()



def train_proposed_model():
    prefix = 'patchTST'
    
    model_preset = Model_preset()
    model_preset.architecture_config.arch_specific_config.seq_len = 200
    model_preset.architecture_config.arch_specific_config.pred_len = 200
    model_preset.architecture_config.arch_specific_config.patch_len = 20
    model_preset.architecture_config.arch_specific_config.stride = 10
    training = QUICK_TRAINING(
        dataset_preset=TSB_AD_U_preset(),
        model_preset=model_preset,
        trainer_preset=Trainer_preset(),
    )
    training.mpkg_name_prefix = prefix
    training.train_model()
    
    model_preset = Model_preset()
    model_preset.architecture_config.arch_specific_config.seq_len = 1000
    model_preset.architecture_config.arch_specific_config.pred_len = 1000
    model_preset.architecture_config.arch_specific_config.patch_len = 100
    model_preset.architecture_config.arch_specific_config.stride = 50
    training = QUICK_TRAINING(
        dataset_preset=UCR_Anomaly_Detection_preset(),
        model_preset=model_preset,
        trainer_preset=Trainer_preset(),
    )
    training.mpkg_name_prefix = prefix
    training.train_model()
    
    return


def train_simple_cnn():
    prefix = 'simple_cnn'
    
    model_preset = Model_preset()
    training = QUICK_TRAINING(
        dataset_preset=TSB_AD_U_preset(),
        model_preset=model_preset,
        trainer_preset=Trainer_preset(),
    )
    training.mpkg_name_prefix = prefix
    training.train_model()
    
    model_preset = Model_preset()
    training = QUICK_TRAINING(
        dataset_preset=UCR_Anomaly_Detection_preset(),
        model_preset=model_preset,
        trainer_preset=Trainer_preset(),
    )
    training.mpkg_name_prefix = prefix
    training.train_model()
    
    return


def get_matrix_profile_result():
    prefix = 'traditional_matrix_profile'
    
    model_preset = Model_preset()
    model_preset.architecture_config.arch_specific_config.window_size = 20
    training = QUICK_TRAINING(
        dataset_preset=TSB_AD_U_preset(),
        model_preset=model_preset,
        trainer_preset=Trainer_preset(),
    )
    training.mpkg_name_prefix = prefix
    training.train_model()
    
    model_preset = Model_preset()
    model_preset.architecture_config.arch_specific_config.window_size = 100
    training = QUICK_TRAINING(
        dataset_preset=UCR_Anomaly_Detection_preset(),
        model_preset=model_preset,
        trainer_preset=Trainer_preset(),
    )
    training.mpkg_name_prefix = prefix
    training.train_model()
    
    return



if __name__ == "__main__":
    #train_simple_cnn()
    get_matrix_profile_result()
    print("Done!")