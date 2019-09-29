# Define Hyper-parameters
# Neural network
batch_size = 64
learning_rate = 0.002
lr_decay_active = True
lr_decay = 700

# Q-Learning
memory_size = 1000000
gamma = 0.995
num_episodes = 100000

# target network update/copy weights average
target_update_average_active = False
target_update_average_tau = 0.01

# Epsilon-greedy
eps_start = 1.0
eps_end = 0.02
eps_decay = 5000 # It wil reach the end at around 4 times this value. Around 7 hands per game, around 5 steps per hand. Averages 35 steps per episode.

# Binary reward
binary_reward_active = False

# Frequencies
# Policy network copied to target network
target_network_freq = 25
# Saving model
saving_model_freq = 9999999
# Neural network copy
copy_weights_freq = 500000

# Type of rl
rl = 'DQN' # DQN or DDQN
outputs = 30