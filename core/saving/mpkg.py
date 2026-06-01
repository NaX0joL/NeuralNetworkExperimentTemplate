from datetime import datetime
from pathlib import Path

from core.schema import ExperimentContext

from .save import SavingService
from .load import LoadingService



DEFAULT_SAVE_PATH = Path("core/savefolder/mpkg/tmp")



class MPKG():
    def __init__(self, experiment_context:ExperimentContext, mpkg_name:str=None):
        self.mpkg_name = self._resolve_mpkg_name(mpkg_name)
        self.experiment_context = experiment_context
        
        self._init_services()
        return
    
    ### public functions
    
    def add_extra(self, **kwargs) -> None:
        self.saving_service.insert_marker_content(**kwargs)
        return
    
    def save_to_mpkg(self, path:Path) -> None:
        self.saving_service.save(path)
        return
    
    def load_from_mpkg(self, path:Path) -> None:
        pass
    
    ### private helper functions
    
    def _resolve_mpkg_name(self, mpkg_name:str) -> str:
        if mpkg_name is None:
            time_now = datetime.now()
            mpkg_name = f"tmp_exp_{time_now.strftime('%Y-%m-%d_%H-%M-%S')}"
        else:
            mpkg_name = mpkg_name
        return mpkg_name
    
    def _init_services(self) -> None:
        self.saving_service = SavingService(self.experiment_context, self.mpkg_name, DEFAULT_SAVE_PATH)
        self.loading_service = LoadingService(self.experiment_context)
        return