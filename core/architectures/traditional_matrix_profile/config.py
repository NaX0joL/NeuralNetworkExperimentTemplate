from dataclasses import dataclass

from core.abstract import ABSTRACT_Config



@dataclass
class TraditionalMatrixProfileConfig(ABSTRACT_Config):
    window_size: int
    
    @classmethod
    def default(self): 
        traditional_matrix_profile_config = self(
            window_size = 100,
        )
        return traditional_matrix_profile_config