import logging
from t48_gym_tensorflow import T48GymTensorflowContext

logging.basicConfig(level=logging.DEBUG)


t48sess = T48GymTensorflowContext()
t48sess.run_episodes(200)

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
