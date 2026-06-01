
from .schema import TrainerState



class LossLogger():
    def __init__(self, trainer_state:TrainerState):
        self.trainer_state = trainer_state
        self.reset()
        return
    
    def reset(self):
        self.trainer_state.train_loss_log = []
        self.trainer_state.validation_loss_log = []
        return
    
    def update(self):
        self.trainer_state.train_loss_log.append(self.trainer_state.last_train_loss)
        self.trainer_state.validation_loss_log.append(self.trainer_state.last_validation_loss)
        return
    
    def get_log(self):
        return self.trainer_state.train_loss_log, self.trainer_state.validation_loss_log