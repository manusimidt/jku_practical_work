import abc
import os
from pathlib import Path
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


class Logger(abc.ABC):
    """ Extracts and/or persists tracker information. """

    def __init__(self):
        pass

    @abc.abstractmethod
    def on_step(self, step: int, **kwargs):
        pass

    @abc.abstractmethod
    def on_epoch_end(self, epoch: int, **kwargs):
        """Actions to take on the end of an epoch."""
        pass

    @abc.abstractmethod
    def on_episode_end(self, episode: int, **kwargs):
        pass


class ConsoleLogger(Logger):
    def __init__(self, log_every: int = 1):
        """
        :param log_every: log every nth episode
        """
        super().__init__()
        self.log_every = log_every

    def on_step(self, step: int, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, **kwargs):
        pass

    def on_episode_end(self, episode: int, **kwargs):
        if not episode % self.log_every == 0: return
        print(f"Episode: {str(episode).rjust(6)}, return: {kwargs['episode_return']}")


class TensorboardLogger(Logger):
    def __init__(self, log_dir: str = './tensorboard', run_id: str = datetime.today().strftime('%Y-%m-%d-%H%M%S')):
        super().__init__()
        self.writer = SummaryWriter(log_dir=log_dir + os.sep + run_id)
        print(f"Tensor board logging active. Start tensorboard with 'tensorboard --logdir {log_dir}'")

    def on_step(self, step: int, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, **kwargs):
        pass

    def on_episode_end(self, episode: int, **kwargs):
        self.writer.add_scalar('rollout/ep_return', kwargs['episode_return'], episode)
        self.writer.add_scalar('rollout/ep_length', kwargs['episode_length'], episode)


class WeightsAndBiasesLogger:
    # todo
    pass


class Tracker:
    """ Collects loggers """

    def __init__(self, *loggers: Logger):
        self.episode = 0
        self.epoch = 0
        self.steps = 0

        self.step_pointer = 0  # points to the step where the last episode ended
        self.episode_return = 0
        self.loggers = list(loggers)

    def step(self, action: int, reward: float):
        self.steps += 1
        self.episode_return += reward
        for logger in self.loggers:
            logger.on_step(self.epoch, action=action, reward=reward)

    def end_epoch(self):
        self.epoch += 1
        for logger in self.loggers:
            logger.on_epoch_end(self.epoch)

    def end_episode(self):
        self.episode += 1
        episode_length = self.steps - self.step_pointer

        for logger in self.loggers:
            logger.on_episode_end(self.episode, episode_return=self.episode_return, episode_length=episode_length)

        # reset rolling stats
        self.step_pointer = self.steps
        self.episode_return = 0
