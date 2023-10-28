import numpy as np
import pdb
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # # Log scalar value (here a random variable)
        # value = np.random.random()
        # self.logger.record("random_value", value)
        #
        # if self.num_timesteps % 10 == 0:
        #     # self.logger.record("reward", self.locals['rewards'][0])
        #     self.logger.dump(self.locals['rewards'][0])
        # pdb.set_trace()
        # self.logger.record("reward", self.locals['rewards'][0])

        if self.locals['infos'][0]['status'] == 'goal':
            self.logger.record("status", 1.0)
        elif self.locals['infos'][0]['status'] == 'hit':
            self.logger.record("status", -1.)
        else:
            self.logger.record("status", 0.)
        return True

    # def _on_rollout_end(self) -> None:
    #     # pdb.set_trace()
    #     if self.locals['dones'][0]:
    #         self.logger.record("status", 1.0)
    #     else:
    #         if self.locals['infos'][0]['TimeLimit.truncated']:
    #             self.logger.record("status", 0.0)
    #         else:
    #             self.logger.record("status", -1.)
