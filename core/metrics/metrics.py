
from torch import nn

from core.schema import ExperimentContext

from .regression_metrics import RegressionMetrics
from .classification_metrics import ClassificationMetrics
from ..datasets.registry import DATASET_REGISTRY



class MetricsEvaluator():
    def __init__(self, experiment_context:ExperimentContext) -> None:
        #task_type = DATASET_REGISTRY[experiment_context.master_config.dataset_config.dataset_name].raw_getter_class.TASK_TYPE
        task_type = experiment_context.master_config.dataset_config.raw_getter_config.task_type
        if task_type == "regression":
            self.evaluator = RegressionMetrics(
                model = experiment_context.model,
                data_module = experiment_context.data_module,
                tolerance = 2,
            )
        elif task_type == "classification":
            self.evaluator = ClassificationMetrics(
                model = experiment_context.model,
                data_module = experiment_context.data_module,
            )
        else:
            raise TypeError("metrics evaluator error! invalid task type")
        
        return
    
    def calculate(self) -> dict[str, dict]:
        return self.evaluator.calculate_metrics()