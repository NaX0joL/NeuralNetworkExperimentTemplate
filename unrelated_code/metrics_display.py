import os
import sys
import json

print(os.getcwd())
sys.path.append(os.getcwd())

from core.modules.regression_metrics import RegressionMetrics
from core.experiment import Experiment



def generate_path():
    base_dir = 'core/savefolder/mpkg/performance_check'
    model_types = [
        'normal_TF',
        'patchTST/stride_1',
        'patchTST/stride_halfPatchLen',
        'MP',
        #'CNN',
    ]
    patchtst_types = [
        'normal',
        'mpAttn_noWeightShare',
        'mpAttn_yesWeightShare',
    ]
    datasets = [
        'tsb_ad_u',
        'ucr_anomaly_detection',
    ]
    
    result_paths = []
    for model_type in model_types:
        if model_type in ['patchTST/stride_1', 'patchTST/stride_halfPatchLen']:
            mod_path = f'{base_dir}/{model_type}'
            for patchtst_type in patchtst_types:
                for dataset in datasets:
                    modmod_path = f'{mod_path}/{patchtst_type}/{dataset}'
                    
                    for mpkg in os.listdir(modmod_path):
                        mpkg_path = f'{modmod_path}/{mpkg}'
                        result_paths.append(mpkg_path)
        else:
            for dataset in datasets:
                mod_path = f'{base_dir}/{model_type}/{dataset}'
                
                for mpkg in os.listdir(mod_path):
                    mpkg_path = f'{mod_path}/{mpkg}'
                    result_paths.append(mpkg_path)
    
    return result_paths
                        

def print_multi_mpkg_metrics(load_paths):
    results = {
        'max_accuracy': [],
        'f_score': [],
    }
    for load_path in load_paths:
        result = get_mpkg_metrics(load_path)
        results['max_accuracy'].append(result['max_accuracy']['max_accuracy'])
        results['f_score'].append(result['f_score']['f_score'])
    
    print("==== ==== ==== xxx barrier xxx ==== ==== ====")
    for item in load_paths:
        print(item)
    print("max_accuracy")
    for max_acc in results['max_accuracy']:
        print(max_acc)
    print("f_score")
    for f_score in results['f_score']:
        print(f_score)
    print()
    
    return


def get_mpkg_metrics(load_path):
    exp = Experiment()
    exp.load_from_mpkg(load_path)
    
    metrics = RegressionMetrics(model=exp.model, data_module=exp.data_module, tolerance=2)
    results = metrics.calculate_metrics()
    
    #print(results)
    return results



if __name__ == "__main__":
    result_paths = generate_path()
    for item in result_paths:
        print(item)
    
    for index in range(len(result_paths)):
        print_multi_mpkg_metrics(result_paths[index:index+5])
    
    print("Done!")