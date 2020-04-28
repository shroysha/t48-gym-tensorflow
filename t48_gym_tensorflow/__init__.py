import gym
import logging
import os
import tensorflow as tf
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from t48_gym_tensorflow.t48_gym import T48GymEnv


tf.compat.v1.enable_resource_variables()
tf.compat.v1.enable_eager_execution()


class T48GymTensorflowContext:

    max_episode_steps = 20000  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}
    batch_size = 1024  # @param {type:"integer"}
    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    train_dir = "/Volumes/SECONDARY/t48_gym_tensorflow"
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    def __init__(self):
        self._train_py_env = suite_gym.load(T48GymEnv.GYM_ENV_NAME, max_episode_steps=T48GymTensorflowContext.max_episode_steps)
        self._eval_py_env = suite_gym.load(T48GymEnv.GYM_ENV_NAME, max_episode_steps=T48GymTensorflowContext.max_episode_steps)
        self._train_env = tf_py_environment.TFPyEnvironment(self._train_py_env)
        self._eval_env = tf_py_environment.TFPyEnvironment(self._eval_py_env)

        self._q_net = q_network.QNetwork(
          self._train_env.observation_spec(),
          self._train_env.action_spec(),
          fc_layer_params=(100,))
        self._agent = dqn_agent.DqnAgent(
            self._train_env.time_step_spec(),
            self._train_env.action_spec(),
            q_network=self._q_net,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=T48GymTensorflowContext.learning_rate),
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=tf.Variable(0))
        self._agent.initialize()
        self._agent.train = common.function(self._agent.train)

        self._replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self._agent.collect_data_spec,
            batch_size=self._train_env.batch_size,
            max_length=T48GymTensorflowContext.replay_buffer_max_length)
        self._dataset = self._replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self._train_env.batch_size,
            num_steps=2).prefetch(3)
        self._iterator = iter(self._dataset)

        self._RANDOM_POLICY = random_tf_policy.RandomTFPolicy(self._train_env.time_step_spec(),
                                                              self._train_env.action_spec())
        self._collect_policy = self._agent.collect_policy
        self._eval_policy = self._agent.policy

        self._collect_driver = dynamic_step_driver.DynamicStepDriver(
            self._train_env,
            self._collect_policy,
            observers=[self._replay_buffer.add_batch] + T48GymTensorflowContext.train_metrics,
            num_steps=2)

        self._global_step = tf.compat.v1.train.get_or_create_global_step()
        self._train_checkpointer = common.Checkpointer(
            ckpt_dir=T48GymTensorflowContext.train_dir,
            agent=self._agent,
            global_step=self._global_step,
            metrics=metric_utils.MetricsGroup(T48GymTensorflowContext.train_metrics, 'train_metrics'))
        self._policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(T48GymTensorflowContext.train_dir, 'policy'),
            policy=self._eval_policy,
            global_step=self._global_step)
        self._rb_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(T48GymTensorflowContext.train_dir, 'replay_buffer'),
            max_to_keep=1,
            replay_buffer=self._replay_buffer)

        self._train_checkpointer.initialize_or_restore()
        self._policy_checkpointer.initialize_or_restore()
        self._rb_checkpointer.initialize_or_restore()

    def run_episodes(self, episode_count):
        for episode_num in range(episode_count):

            logging.info("StartingEpisode:", episode_num)
            time_step = self._train_env.reset()
            policy_state = self._collect_policy.get_initial_state(self._train_env.batch_size)
            logging.debug("ResetEnvironment")

            while not time_step.is_last():
                time_step, policy_state = self._collect_driver.run(
                    time_step=time_step,
                    policy_state=policy_state,
                )
                logging.info("TimeStep:", time_step)
                logging.info("PolicyState:", policy_state)

                experience, unused_info = next(self._iterator)
                logging.info("Experience:", experience)
                logging.debug("UnusedInfo:", unused_info)

                train_loss = self._agent.train(experience)
                logging.info("TrainLoss:",  train_loss)

            logging.info("EndOfEpisode:", episode_num)
            self._train_checkpointer.save(global_step=self._global_step.numpy())
            self._rb_checkpointer.save(global_step=self._global_step.numpy())
            self._policy_checkpointer.save(global_step=self._global_step.numpy())

            #
            # episode_reward = 0
            # print(time_step.is_last())
            # while not time_step.is_last():
            #     action_step = self._collect_policy.action(time_step)
            #     print(action_step.action)
            #     next_time_step = self._train_env.step(action_step.action)
            #     # print(next_time_step)
            #     episode_reward += time_step.reward
            #     traj = trajectory.from_transition(time_step, action_step, next_time_step)
            #     self._replay_buffer.add_batch(traj)
            #     time_step = next_time_step
            #     print("Action:", action_step.action, "Reward: ", time_step.reward.numpy())
            #     print(self._train_py_env.gym.t48_game.t48_board.board_data)


class T48GymTensorflowContextManager:

    def __init__(self, t48context):
        self._t48context = t48context

