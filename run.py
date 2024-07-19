import torch
from src.env.env import CarEnv

if __name__ == "__main__":
    # dqn action values
    action_values = [
        -0.75, 
        -0.5, 
        -0.25, 
        -0.15, 
        -0.1, 
        -0.05, 
        0,
        0.05, 
        0.1,
        0.5, 
        0.75
    ]
    action_map = {i:x for i, x in enumerate(action_values)}

    try:
        buffer_size = 1e4
        batch_size = 32
        episodes = 5000
        state_dim = (128, 128)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_action = len(action_values)
        in_channels = 1

        replay_memory = ReplayMemory(state_dim, batch_size, buffer_size, device)
        model = DQNAgent(num_action, state_dim, in_channels, device)

        # Set to True if you want to run with pygame
        env = CarEnv(show_pygame_display=True)

        for ep in range(episodes):
            env.step(model, replay_memory, ep, action_map)
            env.reset()
    finally:
        env.reset()
        env.close()
        env.show_reward_graph()
