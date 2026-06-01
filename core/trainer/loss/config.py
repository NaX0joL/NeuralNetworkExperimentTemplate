from dataclasses import dataclass

from core.abstract import ABSTRACT_Config



@dataclass
class LossConfig(ABSTRACT_Config):
    use_multi_loss: str
    multi_loss_coeff: dict[str, float]
    
    @classmethod
    def default(self):
        loss_config = self(
            use_multi_loss = False,
            multi_loss_coeff = {
                "base_loss": 1,
                "matrix_profile_loss": 0,
            }
        )
        return loss_config