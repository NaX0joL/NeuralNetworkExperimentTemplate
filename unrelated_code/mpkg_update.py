import os
import sys
import json

print(os.getcwd())
sys.path.append(os.getcwd())

from core.modules.mpkg import MPKGCreator



BASE_DIR = 'core/savefolder/mpkg/performance_check'

MODEL_TYPES = [
    'normal_TF',
    'patchTST/stride_1',
    'patchTST/stride_halfPatchLen',
    'MP',
    #'CNN',
]

PATCHTST_TYPES = [
    'normal',
    'mpAttn_noWeightShare',
    'mpAttn_yesWeightShare',
]

DATASETS = [
    'tsb_ad_u',
    'ucr_anomaly_detection',
]



def update_mpkg():
    from core.experiment import Experiment
    
    load_paths = []
    for model_type in MODEL_TYPES:
        
        if model_type in ['patchTST/stride_1', 'patchTST/stride_halfPatchLen']:
            
            for patchtst_type in PATCHTST_TYPES:
                mod_path = f'{BASE_DIR}/{model_type}/{patchtst_type}'
        
                for dataset_used in DATASETS:
                    
                    folder_path = f'{mod_path}/{dataset_used}'
                    for mpkg in os.listdir(folder_path):
                        
                        mpkg_path = f'{folder_path}/{mpkg}'
                        load_paths.append(mpkg_path)
        
        else:
            for dataset_used in DATASETS:
                folder_path = f'{BASE_DIR}/{model_type}/{dataset_used}'
                for mpkg in os.listdir(folder_path):
                    
                    mpkg_path = f'{folder_path}/{mpkg}'
                    load_paths.append(mpkg_path)
    
    exp = Experiment()
    for mpkg_path in load_paths:
        print(mpkg_path)
        exp.load_from_mpkg(load_path=mpkg_path)
        
        mpkg_creator = MPKGCreator()
        mpkg_creator.load_mpkg(mpkg_path)
        mpkg_creator.update_mpkg(mpkg_path)
        
        mpkg_path = os.path.dirname(mpkg_path)
        
    return



if __name__ == "__main__":
    update_mpkg()
    print("Done!")