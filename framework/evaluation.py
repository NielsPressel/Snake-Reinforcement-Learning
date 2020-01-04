

class Evaluation:

    def __init__(self, create_env_func, agent, weights_file_path):
        self.create_env_function = create_env_func
        self.agent = agent
        self.agent.load(weights_file_path)

    def evaluate(self, max_rounds=1, max_steps=10_000, visualize=False, plot_func=None):
        self.agent.training = False
        env = self.create_env_function()

        fail_counter = 0
        
        for i in range(max_rounds):
            state = env.reset()
            episode_reward = 0
            episode_rewards = []
            episode_steps = []
    
            for step in range(max_steps):
                if visualize:
                    env.render()
                action = self.agent.act(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    fail_counter += 1
                    episode_rewards.append(episode_reward)
                    episode_steps.append(step)
                    episode_reward = 0
                    if plot_func:
                        plot_func(episode_rewards, episode_steps)
                    state = env.reset()
                else:
                    state = next_state
    
            if plot_func:
                plot_func(episode_rewards, episode_steps, False)

        return fail_counter