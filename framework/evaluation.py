import time

"""---Evaluation class---"""


class Evaluation:
    """Simple class for evaluating a trained model.

    Attributes:
        create_env_function (lambda): Function that creates the environment
        agent (Agent): Agent to evaluate
        rewards (dictionary): Collection of rewards to instantiate the environment
    """

    def __init__(self, create_env_func, agent, weights_file_path, rewards=None):
        self.create_env_function = create_env_func
        self.rewards = rewards
        self.agent = agent
        self.agent.load(weights_file_path)

    def evaluate(self, max_rounds=1, max_steps=10_000, visualize=False, plot_func=None, step_delay=None):
        """Evaluates a trained model

        Args:
            max_rounds (int): Number of rounds to evaluate on
            max_steps (int): Maximum amount of steps per round
            visualize (bool): Decides wether to render the game while evaluating or not
            plot_func (lambda): Function to plot the rewards to
            step_delay (float, None): Time to pause the game after every step
        """
        self.agent.training = False
        env = self.create_env_function(rewards=self.rewards)

        fail_counter = 0

        snake_lens = []

        for i in range(max_rounds):
            state = env.reset()
            episode_reward = 0
            episode_rewards = []
            episode_steps = []

            for step in range(max_steps):
                if step_delay:
                    time.sleep(step_delay)
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
                    snake_lens.append(len(env.snake))
                    state = env.reset()
                    if plot_func:
                        plot_func(episode_rewards, episode_steps)
                else:
                    state = next_state

            snake_lens.append(len(env.snake))
            if plot_func:
                plot_func(episode_rewards, episode_steps, False)

        print(float(sum(snake_lens)) / float(len(snake_lens)))
        return fail_counter
