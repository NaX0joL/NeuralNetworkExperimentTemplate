
def test_raw_getter():
    # tested on 2026-04-26
    
    from core.datasets.UCR_Classification import raw_getter
    
    config = raw_getter.UCR_Classification_Config.default()
    config.sub_dataset_name = "FordA"
    config.sub_dataset_name = ["FordA", "FordB"]
    
    getter = raw_getter.UCR_Classification_RawGetter(config)
    
    raw = getter.get_raw()
    print(raw)
    
    return


def test_dataset():
    # test on 2026-04-26

    from core.datasets import data_module
    
    config = data_module.DatasetConfig.default()
    
    datamodule_getter = data_module.DataModuleGetter(config)
    
    data_module = datamodule_getter.get_data_module()
    print(data_module)
    print(data_module.train_dataloader)
    print(data_module.validation_dataloader)
    print(data_module.test_dataloader)
    
    return



def run_exp_tmp():
    
    from core.experiment import Experiment
    from core.master_config import MasterConfig
    
    sub_dataset = {
        'FordA': {
            'seq_len': 500,
            'n_class': 2,
        },
        'FordB': {
            'seq_len': 500,
            'n_class': 2,
        },
        'ECG5000': {
            'seq_len': 140,
            'n_class': 5,
        },
        'ElectricDevices': {
            'seq_len': 96,
            'n_class': 7,
        },
    }
    
    master_config = MasterConfig.hardcoded_config()
    for sub_dataset_name, metadata in sub_dataset.items():
        print(f"Training model on dataset {sub_dataset_name}")
        
        master_config.trainer_config.train_epochs = 1
        
        master_config.architecture_config.arch_specific_config.use_matrix_profile_attention = False
        
        master_config.architecture_config.arch_specific_config.seq_len = metadata.get('seq_len')
        master_config.architecture_config.arch_specific_config.pred_len = metadata.get('n_class')
        master_config.architecture_config.arch_specific_config.patch_len = metadata.get('seq_len') // 10
        
        master_config.dataset_config.raw_getter_config.sub_dataset_name = sub_dataset_name
        
        exp_id = f"testExp_ProposedModel_{sub_dataset_name}"
        exp = Experiment(master_config=master_config, experiment_id=exp_id)
        
        exp.train_model()
        exp.save_to_mpkg()
    
    return



def rerun_for_metrics():
    
    from core.experiment import Experiment
    from core.master_config import MasterConfig
    
    mpkg_path_list = [
        "savefolder/proposed_model/mpkg-testExp_ProposedModel_FordA",
        "savefolder/proposed_model/mpkg-testExp_ProposedModel_FordB",
        "savefolder/proposed_model/mpkg-testExp_ProposedModel_ECG5000",
        "savefolder/proposed_model/mpkg-testExp_ProposedModel_ElectricDevices",
        "savefolder/proposed_model_mpAttention/mpkg-testExp_ProposedModel_mpAttention_FordA",
        "savefolder/proposed_model_mpAttention/mpkg-testExp_ProposedModel_mpAttention_FordB",
        "savefolder/proposed_model_mpAttention/mpkg-testExp_ProposedModel_mpAttention_ECG5000",
        "savefolder/proposed_model_mpAttention/mpkg-testExp_ProposedModel_mpAttention_ElectricDevices",
    ]
    
    exp = Experiment()
    for path in mpkg_path_list:
        print(path.split('/')[-1], end=" ")
        exp.load_from_mpkg(load_path=path)
        results = exp.get_model_metrics()
        print(results)
    
    return



if __name__ == "__main__":
    #test_raw_getter()
    #test_dataset()
    run_exp_tmp()
    #rerun_for_metrics()
    print("Done!")
    
    
    ### --- tmp notes, plz delete after ---
    
    # mpkg-testExp_ProposedModel_FordA {'accuracy': 0.8931818181818182}                                                                                                                   
    # mpkg-testExp_ProposedModel_FordB {'accuracy': 0.7716049382716049}                                                                                                                   
    # mpkg-testExp_ProposedModel_ECG5000 {'accuracy': 0.9017777777777778}
    # mpkg-testExp_ProposedModel_ElectricDevices {'accuracy': 0.40876669692646866}
    
    # mpkg-testExp_ProposedModel_mpAttention_FordA {'accuracy': 0.8348484848484848}
    # mpkg-testExp_ProposedModel_mpAttention_FordB {'accuracy': 0.645679012345679}
    # mpkg-testExp_ProposedModel_mpAttention_ECG5000 {'accuracy': 0.9031111111111111}
    # mpkg-testExp_ProposedModel_mpAttention_ElectricDevices {'accuracy': 0.3475554402801193}