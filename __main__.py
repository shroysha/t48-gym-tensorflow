import gym

import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.utils import common

from tf_gym_2048 import T48GymEnv

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}


train_py_env = suite_gym.load(T48GymEnv.GYM_ENV_NAME)
eval_py_env = suite_gym.load(T48GymEnv.GYM_ENV_NAME)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

#
# q_net = q_network.QNetwork(
#   train_env.observation_spec(),
#   train_env.action_spec(),
#   fc_layer_params=(100,))
#
# learning_rate = 1e-3  # @param {type:"number"}
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
#
# train_step_counter = tf.Variable(0)
#
# agent = dqn_agent.DqnAgent(
#     train_env.time_step_spec(),
#     train_env.action_spec(),
#     q_network=q_net,
#     optimizer=optimizer,
#     td_errors_loss_fn=common.element_wise_squared_loss,
#     train_step_counter=train_step_counter)
#
# agent.initialize()


# env = gym.make("T48GymEnv-v0")
# env.reset()
# TIMESTEPS = 500
# for t in range(TIMESTEPS):
#     env.render()
#     observation, reward, done, info = env.step(env.action_space.sample())
#     if done:
#         break


#
# def play_with_user():
#     t48game = T48Game()
#
#     while not t48game.is_game_over:
#         print()
#         print("2048 :: Score :: ", str(t48game.score))
#         print(t48game.t48_board.board_data)
#         command = input("Enter command (wsad): ")
#
#         try:
#             if command == "exit":
#                 exit()
#             elif command == "preview":
#                 print(t48game.t48_board.preview_swipe_up)
#                 print(t48game.t48_board.preview_swipe_left)
#                 print(t48game.t48_board.preview_swipe_down)
#                 print(t48game.t48_board.preview_swipe_right)
#             else:
#                 t48game.do_swipe_choice(command)
#         except AssertionError as ex:
#             print("*** Cannot do ***")
#

# play_with_user()
