# Define Hyper-parameters
batch_size = 1
# Actor
actor_learning_rate = 0.0005
actor_lr_decay_active = True
actor_lr_decay = 5000

# Critic
critic_learning_rate = 0.001
critic_lr_decay_active = True
critic_lr_decay = 5000

gamma = 0.9995
num_episodes = 100000

env_name = 'CartPole-v1'