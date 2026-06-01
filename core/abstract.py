from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict



class ABSTRACT_Config(ABC):
    @abstractmethod
    def default(self):
        pass



class ABSTRACT_ArchitectureControl(ABC):
    @abstractmethod
    def get_model(self):
        pass



class ABSTRACT_DataModuleGetter(ABC):
    @abstractmethod
    def get_data_module(self, **kwargs):
        pass



class ABSTRACT_RawGetter(ABC):
    DATASET_NAME: str
    TASK_TYPE: str
    ATTRIBUTES: dict[str, type]
    
    @abstractmethod
    def get_raw(self, **kwargs):
        pass



class ABSTRACT_Trainer(ABC):
    @abstractmethod
    def get_model(self):
        pass
    
    @abstractmethod
    def fit(self):
        pass



class ABSTRACT_Loss(ABC):
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def set_criterion(self):
        pass