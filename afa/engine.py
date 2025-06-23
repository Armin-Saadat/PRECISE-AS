import os
import random
import logging
import wandb
import torch
import numpy as np

from afa.builders import data_builder, classifier_builder, agent_builder, environment_builder
from afa.utils import get_classification_metrics, Profiler
from afa.agents import DDQNAgent


class Engine(object):
    def __init__(self, config, logger: logging.Logger, save_dir: str):
        self.logger = logger
        self.save_dir = save_dir
        self.config = config
        if self.config.device == 'cuda':
            self.logger.warning('Using {} GPU/s!'.format(torch.cuda.device_count()))
        else:
            self.logger.warning(f'Using {self.config.device}!')

        # Set up Wandb if required
        if config.use_wandb:
            wandb.init(project=config.wandb_project_name,
                       name=self.save_dir.removeprefix('./logs/'),
                       config=config,
                       mode=config.wandb_mode)

        # set data
        self.data_train, self.data_val, self.data_test = data_builder.build(self.config.data)

        # set classifier
        self.config.classifier.update({'device': self.config.device})
        self.classifier = classifier_builder.build(self.config.classifier, self.logger, self.data_train, self.data_val)

        # set profilers
        self.profilers = {'Classifier Predictions': Profiler('Classifier Predictions', self.logger),
                          'Agent Taking Actions': Profiler('Agent Taking Actions', self.logger),
                          'Agent Updates': Profiler('Agent Updates', self.logger),
                          'Agent Store Transitions': Profiler('Agent Store Transitions', self.logger)}

        self.n_transitions = 0
        self.env = None
        self.agent = None

    def train(self):
        # set env
        self.config.environment.update({'device': self.config.device})
        self.env = environment_builder.build(self.config.environment, self.logger, self.classifier, self.data_train)

        # set agent
        self.config.agent.update({
            'state_shape': self.env.state_shape,
            'n_actions': self.env.action_space.n,
            'device': self.config.device})
        self.agent = agent_builder.build(self.config.agent, self.logger)

        top_reward_val = float('-inf')
        for epoch in range(self.config.n_epochs):

            ######################### train one epoch #########################
            self.env.data = self.data_train
            with Profiler() as p:
                reward, n_actions, classification_metrics = self._train_one_epoch(epoch)

            self.logger.info_important(
                ' Train: Epoch {} - Time {:.1f} - Mean Reward {:.3f} - Mean N_Actions {:.3f} - Acc {:.3f} - Balanced Acc {:.3f}'.format(
                    epoch, p.times[-1], reward, n_actions, classification_metrics['ACC'],
                    classification_metrics['bACC']))

            if self.config.log_profile:
                for profiler in self.profilers.values():
                    profiler.log()
                    profiler.reset()

            if self.config.use_wandb:
                wandb.log({'epoch': epoch,
                           'train_n_actions_per_epoch': n_actions,
                           'train_reward_per_epoch': reward,
                           'train_accuracy_per_epoch': classification_metrics['ACC'],
                           'train_balanced_accuracy_per_epoch': classification_metrics['bACC'],
                           'learning_rate': self.agent.scheduler.get_last_lr()[
                               0] if self.config.agent.agent_name != 'TabularQ' else 0.0,
                           'train_epoch_time(s)': p.times[-1]})
                wandb.save(os.path.join(self.save_dir, 'output.log'))

            if self.config.agent.agent_name != 'TabularQ':
                self.agent.scheduler.step()
            ###################################################################

            ######################## validate one epoch #######################
            self.env.data = self.data_val
            with Profiler() as p:
                reward, n_actions, classification_metrics = self._validate_one_epoch(epoch)

            self.logger.info_important(
                ' Val: Epoch {} - Time {:.1f} - Mean Reward {:.3f} - Mean N_Actions {:.3f} - Acc {:.3f} - Balanced Acc {:.3f}'.format(
                    epoch, p.times[-1], reward, n_actions, classification_metrics['ACC'],
                    classification_metrics['bACC']))

            if self.config.log_profile:
                for profiler in self.profilers.values():
                    profiler.log()
                    profiler.reset()

            if self.config.use_wandb:
                wandb.log({'val_n_actions_per_epoch': n_actions,
                           'val_reward_per_epoch': reward,
                           'val_accuracy_per_epoch': classification_metrics['ACC'],
                           'val_balanced_accuracy_per_epoch': classification_metrics['bACC'],
                           'val_epoch_time(s)': p.times[-1]})
                wandb.save(os.path.join(self.save_dir, 'output.log'))

            if reward >= top_reward_val:
                self.agent.save(self.save_dir, epoch, mode='best')
                top_reward_val = reward

            if (epoch + 1) % 10 == 0:
                self.agent.save(self.save_dir, epoch)
            ###################################################################

    def evaluate(self):
        # set env
        self.config.environment.update({'device': self.config.device})
        self.env = environment_builder.build(self.config.environment, self.logger, self.classifier, self.data_test)

        # set agent
        self.config.agent.update({
            'state_shape': self.env.state_shape,
            'n_actions': self.env.action_space.n,
            'device': self.config.device})
        self.agent = agent_builder.build(self.config.agent, self.logger)
        self.agent.eps_start, self.agent.eps_final = 0, 0

        reward_per_episode, n_actions_per_episode, y_preds, y_trues = [], [], [], []

        for episode_idx in range(self.env.n_episodes()):
            episode_reward, _, y_pred, y_true, info = self._run_one_episode(episode_idx, phase='test')
            reward_per_episode.append(episode_reward)
            n_actions_per_episode.append(len(self.env.action_history))
            y_preds.append(y_pred)
            y_trues.append(y_true)

            self.logger.info(
                ' - Episode {} - Reward {:.3f} - Actions {} - y_pred {} - y_true {} {}'.format(
                    episode_idx, episode_reward, self.env.action_history, y_pred, y_true, info))

            if self.config.use_wandb:
                wandb.log({'test_episode': episode_idx,
                           'test_n_actions_per_episode': n_actions_per_episode[-1],
                           'test_reward_per_episode': reward_per_episode[-1]})

        classification_metrics = get_classification_metrics(y_trues, y_preds)

        self.logger.info_important(
            'Mean N_Actions {:.2f} - bACC {:.3f} - F1 {:.3f} - MAE {:.3f}'.format(
                np.mean(n_actions_per_episode),
                classification_metrics['bACC'],
                classification_metrics['F1'],
                classification_metrics['MAE']))

        if self.config.log_profile:
            for profiler in self.profilers.values():
                profiler.log()
                profiler.reset()

    def _train_one_epoch(self, epoch: int) -> (list, list):
        reward_per_episode, n_actions_per_episode, y_preds, y_trues = [], [], [], []

        # shuffle order of episodes in epoch
        episodes_idx = list(range(self.env.n_episodes()))
        random.shuffle(episodes_idx)

        for episode_idx in episodes_idx:
            episode_reward, batch_loss, y_pred, y_true, info = self._run_one_episode(episode_idx, phase='train')
            reward_per_episode.append(episode_reward)
            n_actions_per_episode.append(len(self.env.action_history))
            y_preds.append(y_pred)
            y_trues.append(y_true)

            self.logger.info(
                ' Epoch {} - Episode {} - Reward {:.3f} - Actions {} - y_pred {} - y_true {} - Epsilon {:.2f} {}'.format(
                    epoch, episode_idx, episode_reward, self.env.action_history, y_pred, y_true,
                    self.agent.eps if issubclass(type(self.agent), DDQNAgent) else 0, info))

            if self.config.use_wandb:
                wandb.log({'batch_loss': batch_loss,
                           'episode': episode_idx,
                           'train_n_actions_per_episode': n_actions_per_episode[-1],
                           'train_reward_per_episode': reward_per_episode[-1]})

        classification_metrics = get_classification_metrics(y_trues, y_preds)

        return np.mean(reward_per_episode), np.mean(n_actions_per_episode), classification_metrics

    def _validate_one_epoch(self, epoch: int) -> (list, list):
        reward_per_episode, n_actions_per_episode, y_preds, y_trues = [], [], [], []

        for episode_idx in range(self.env.n_episodes()):
            episode_reward, batch_loss, y_pred, y_true, info = self._run_one_episode(episode_idx, phase='val')
            reward_per_episode.append(episode_reward)
            n_actions_per_episode.append(len(self.env.action_history))
            y_preds.append(y_pred)
            y_trues.append(y_true)

            self.logger.warning(
                ' Epoch {} - Episode {} - Reward {:.3f} - Actions {} - y_pred {} - y_true {} - Epsilon {:.2f} {}'.format(
                    epoch, episode_idx, episode_reward, self.env.action_history, y_pred, y_true,
                    self.agent.eps if issubclass(type(self.agent), DDQNAgent) else 0, info))

        classification_metrics = get_classification_metrics(y_trues, y_preds)

        return np.mean(reward_per_episode), np.mean(n_actions_per_episode), classification_metrics

    def _run_one_episode(self, episode_idx: int, phase: str) -> (int, float):
        state, *_ = self.env.reset(episode_idx)
        done = False
        episode_reward = 0
        agent_loss = []

        while not done:
            # take action
            with self.profilers['Agent Taking Actions']:
                if isinstance(self.agent, DDQNAgent):
                    action = self.agent.take_one_action(state, self.n_transitions)

            # take one step
            with self.profilers['Classifier Predictions'] as profiler:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                if not done:
                    profiler.skip()

            if phase == 'train':
                # store transition
                with self.profilers['Agent Store Transitions']:
                    self.agent.store_transition(state, action, reward, next_state, done)
                self.n_transitions += 1

                # update agent
                if isinstance(self.agent, DDQNAgent):
                    if self.n_transitions >= self.agent.memory_bs:
                        with self.profilers['Agent Updates']:
                            agent_loss.append(self.agent.update_policy_net())
                    if self.n_transitions % self.config.agent.target_update_freq == 0:
                        self.agent.update_target_net()

            episode_reward += reward
            state = next_state

        mean_loss = np.mean(agent_loss) if agent_loss else 0
        y_pred, y_true = self.env.y_pred, self.env.y_true

        return episode_reward, mean_loss, y_pred, y_true, info
