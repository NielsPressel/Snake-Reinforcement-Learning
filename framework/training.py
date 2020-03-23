from framework.core import Transition
from framework.agents.dqn import MinibatchDQN
from framework.memory import DualExperienceReplay

import sys
import os
import re
import multiprocessing as mp
import cloudpickle

from collections import namedtuple


RewardState = namedtuple('RewardState', ['reward', 'state'])


"""---Training class---"""


class Training:
    """Training class for managing the training process.

    Attributes:
        create_env_func (lambda): The function to create the environment from
        agent (Agent): The agent to train
    """

    def __init__(self, create_env_func, agent):
        """Training constructor

        Args:
            create_env_func (lambda): The function to create the environment from
            agent (Agent): The agent to train
        """
        self.create_env_func = create_env_func
        self.agent = agent

    def train(self, max_steps=100_000, instances=1, visualize=False, plot_func=None, max_subprocesses=0,
              checkpnt_func=None, path="", rewards=None, resume=False):
        """Trains the agent to better perform in the specified environment.

        Args:
            max_steps (int): Maximum amount of training steps
            instances (int): Maximum amount of environment instances to collect data concurrently
            visualize (bool): Decides wether to render the game while training or not
            plot_func (lambda): Function to plot the reward
            max_subprocesses (int): Number of subprocesses that can be created by the training process
            checkpnt_func (lamda): Function to save the state of the agent
            path (string): Folder to save the checkpoint data to
            rewards (dictionary): Collection of rewards needed to instantiate the environment
            resume (bool): Decides wether a new training process is started or an old one gets resumed
        """
        if max_subprocesses == 0:
            self._sp_train(max_steps, instances, visualize, plot_func, checkpnt_func, path, rewards, resume)
        else:
            self._mp_train(max_steps, instances, max_subprocesses, visualize, plot_func, checkpnt_func, path, rewards,
                           resume)

    def train_epochal(self, max_steps=100_00, max_subprocesses=0, checkpnt_func=None, path="", rewards=None):
        """Trains the agent like train() except data collection and training alternate.

        Args:
            max_steps (int): Maximum amount of training steps
            max_subprocesses (int): Number of subprocesses that can be created by the training process
            checkpnt_func (lambda): Function to save the state of the agent
            path (string): Folder to save the checkpoint data to
            rewards (dictionary): Collection of rewards needed to instantiate the environment
        """
        if max_subprocesses == 0:
            self._sp_train_epochal(max_steps, checkpnt_func, path, rewards)
        elif max_subprocesses >= 1:
            self._mp_train_epochal(max_steps, max_subprocesses, checkpnt_func, path, rewards)

    def evaluate(self, max_steps=10_000, visualize=False, plot_func=None, rewards=None):
        """Evaluates the model after training by playing a few games.

        Args:
            max_steps (int): Maximum amount of steps to evaluate the model in
            visualize (bool): Decides wether to render the game or not
            plot_func (lambda): The function that plots the reward data. Function must be of type func(rewards: float[],
                                steps: float[], done: bool). If plot_func is None plotting is disabled.
            rewards (dictionary): Collection of rewards to instantiate the environment
        """
        self.agent.training = False
        env = self.create_env_func(rewards=rewards)
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
                episode_rewards.append(episode_reward)
                episode_steps.append(step)
                episode_reward = 0
                if plot_func:
                    plot_func(episode_rewards, episode_steps)
                state = env.reset()
            else:
                state = next_state

        if plot_func:
            plot_func(episode_rewards, episode_steps, True)

    def _sp_train(self, max_steps, instances, visualize, plot, checkpnt_func, path, rewards, resume):
        """Trains using a single process.

        (Private function, call train() with max_subprocesses=0 instead)

        Args:
            max_steps (int): Maximum amount of training steps
            instances (int): Amount of instances to collect data on
            visualize (bool): Decides wether to render the game or not
            plot (lambda): The function that plots the reward data. Function must be of type func(rewards: float[],
                           steps: float[], done: bool). If plot_func is None plotting is disabled.
            checkpnt_func (lambda): The function used to create checkpoints every 1000 steps
            path (string): The folder path to save the checkpoints to
            rewards (dictionary): Collection of rewards needed to instantiate the environment
            resume (bool): Decides wether to create a new training session or resume an old one
        """

        # Keep track of rewards per episode per instance
        episode_reward_sequences = [[] for i in range(instances)]
        episode_step_sequences = [[] for i in range(instances)]
        episode_rewards = [0] * instances

        game_info = []
        last_game_step = 0

        # Create and initialize environment instances
        envs = [self.create_env_func(rewards=rewards) for _ in range(instances)]
        states = [env.reset() for env in envs]
        curr_step = 0

        # If resuming load the weights file and retrieve the last saved step from the filename
        if resume:
            full_path = os.path.join(path, 'checkpoints')
            file_list = [str(f) for f in os.listdir(full_path)]
            for f in file_list:
                if f.find('.dat.data') != -1:
                    file = os.path.join(full_path, f.split('.data')[0])
                    self.agent.load(file)  # Load weights
                    curr_step = int(re.findall(r"\d+", f.split(".data")[0])[0])  # Parse step from filename

        # Training loop
        for step in range(curr_step, max_steps):
            for i in range(instances):
                if visualize:
                    envs[i].render()
                action = self.agent.act(states[i])  # Get next action from agent

                # Take one step using the agent's action
                if isinstance(self.agent.memory, DualExperienceReplay):
                    next_state, reward, done, save = envs[i].step(action, self.agent.memory.adjust_rewards)
                else:
                    next_state, reward, done, save = envs[i].step(action)

                # Store transition in memory
                if save:
                    self.agent.push_observation(Transition(states[i], action, reward, None if done else next_state))
                episode_rewards[i] += reward

                if isinstance(self.agent, MinibatchDQN):
                    self.agent.train_short_memory(states[i], action, reward, None if done else next_state)

                if done:  # On terminal state reset the environment and save the cumulative reward
                    game_info.append((episode_rewards[i], step - last_game_step, len(envs[i].snake)))
                    last_game_step = step

                    episode_reward_sequences[i].append(episode_rewards[i])
                    episode_step_sequences[i].append(step)
                    episode_rewards[i] = 0
                    if plot:  # Plot every step if plot function is given
                        plot(episode_reward_sequences, episode_step_sequences)
                    states[i] = envs[i].reset()

                    if isinstance(self.agent, MinibatchDQN):
                        self.agent.train(step)

                else:  # Roll state
                    states[i] = next_state

            # Perform one step of the optimization

            if not isinstance(self.agent, MinibatchDQN):
                self.agent.train(step)
            sys.stdout.write("\rDone with %.1f percent of the training" % ((float(step) / float(max_steps)) * 100.0))
            sys.stdout.flush()



            # If checkpoint function is given save a checkpoint every 1000 steps
            if checkpnt_func:
                if step % 1000 == 0:
                    checkpnt_func(path, self.agent, step)

            if step % 50_000 == 0:
                with open(os.path.join(path, "training_metrics.txt"), 'a') as f:
                    for item in game_info:
                        f.write(str(item[0]) + ", " + str(item[1]) + ", " + str(item[2]) + "\n")
                    f.flush()
                game_info = []

        # If plot function is given plot the reward at the end
        if plot:
            plot(episode_reward_sequences, episode_step_sequences, done=True)

    def _mp_train(self, max_steps, instances, max_subprocesses, visualize, plot, checkpnt_func, path, rewards, resume):
        """Trains using multiple processes.

        (Private function, call train() with max_subprocesses > 0 instead)

        Args:
            max_steps (int): Maximum amount of training steps
            instances (int): Amount of instances to collect data on
            max_subprocesses (int): Maximum amount of subprocesses the training process can create
            visualize (bool): Decides wether to render the game or not
            plot (lambda): The function that plots the reward data. Function must be of type func(rewards: float[],
                           steps: float[], done: bool). If plot_func is None plotting is disabled.
            checkpnt_func (lambda): The function used to create checkpoints every 1000 steps
            path (string): The folder path to save the checkpoints to
            rewards (dictionary): Collection of rewards needed to instantiate the environment
            resume (bool): Decides wether to create a new training session or to resume an old one
        """

        if max_subprocesses is None:
            max_subprocesses = mp.cpu_count()

        if resume:  # Not implemented yet
            raise NotImplementedError()

        n_processes = min(instances, max_subprocesses)

        # Distribute instances equally
        instances_per_process = [instances // n_processes] * n_processes
        leftover = instances % n_processes
        if leftover > 0:
            for i in range(leftover):
                instances_per_process[i] += 1

        instance_ids = [list(range(i, instances, n_processes))[:ipp] for i, ipp in enumerate(instances_per_process)]

        # Create multiprocessing infrastructure (parent and child pipes)
        pipes = []
        processes = []
        for i in range(n_processes):
            child_pipes = []
            for j in range(instances_per_process[i]):
                parent, child = mp.Pipe()
                pipes.append(parent)
                child_pipes.append(child)

            pargs = (cloudpickle.dumps(self.create_env_func), instance_ids[i], max_steps, child_pipes, visualize, rewards)
            processes.append(mp.Process(target=_train, args=pargs))

        # Start processes
        print(f"Starting {n_processes} process(es) for {instances} environment instance(s)... {instance_ids}")
        for p in processes:
            p.start()

        episode_reward_sequences = [[] for i in range(instances)]
        episode_step_sequences = [[] for i in range(instances)]
        episode_rewards = [0] * instances

        # Temporarily save reward states and last actions
        rss = [None] * instances
        last_actions = [None] * instances

        for step in range(max_steps):
            step_done = [False] * instances

            while sum(step_done) < instances:  # Synchronize on step
                awaiting_pipes = [p for iid, p in enumerate(pipes) if step_done[iid] == 0]
                ready_pipes = mp.connection.wait(awaiting_pipes, timeout=None)
                pipe_indices = [pipes.index(rp) for rp in ready_pipes]

                pipe_indices.sort()
                for iid in pipe_indices:
                    rs = pipes[iid].recv()

                    if rss[iid] is not None:  # First state exists; build Transition
                        experience = Transition(rss[iid].state, last_actions[iid], rs.reward, rs.state)
                        self.agent.push_observation(experience)
                        step_done[iid] = True
                    rss[iid] = rs  # Roll state

                    if rs.state is None:  # Terminal state
                        rss[iid] = None
                        episode_reward_sequences[iid].append(episode_rewards[iid])
                        episode_step_sequences[iid].append(step)
                        episode_rewards[iid] = 0
                        if plot:
                            plot(episode_reward_sequences, episode_step_sequences)
                    else:
                        action = self.agent.act(rs.state)  # Get next action
                        last_actions[iid] = action
                        try:
                            pipes[iid].send(action)  # Send action to child process
                        except BrokenPipeError as bpe:
                            if step < (max_steps - 1):
                                raise bpe
                        if rs.reward:
                            episode_rewards[iid] += rs.reward

            self.agent.train(step)  # Train after every step
            sys.stdout.write("\rDone with %.1f percent of the training" % ((float(step) / float(max_steps)) * 100.0))
            sys.stdout.flush()

            # If a checkpoint function is given save a checkpoint every 1000 steps
            if checkpnt_func:
                if step % 1000 == 0:
                    checkpnt_func(path, self.agent, step)

        # Plot the rewards if a function is specified
        if plot:
            plot(episode_reward_sequences, episode_step_sequences, done=True)

    def _sp_train_epochal(self, max_steps, checkpnt_func, path, rewards):
        """Trains on single process in epochal fashion.

        (Private function, use train_epochal() with max_subprocesses=0 instead)

        Args:
            max_steps (int): Maximum amount of training steps
            checkpnt_func (lambda): The function used to create checkpoints every 1000 steps
            path (string): The folder path to save the checkpoints to
            rewards (dictionary): Collection of rewards needed to instantiate the environment
        """

        # Keep track of rewards
        episode_reward_sequences = []
        episode_step_sequences = []
        episode_reward = 0

        # Create environment and reset it
        env = self.create_env_func(rewards=rewards)
        state = env.reset()

        # Training loop
        for step in range(0, max_steps):
            action = self.agent.act(state)  # Get action
            next_state, reward, done, _ = env.step(action)  # Take on step, get next state

            # Save Transition to memory
            self.agent.push_observation(Transition(state, action, reward, None if done else next_state))
            episode_reward += reward

            # On terminal state save cumulative reward and reset the environment
            if done:
                episode_reward_sequences.append(episode_reward)
                episode_step_sequences.append(step)
                episode_reward = 0

                state = env.reset()
            else:
                state = next_state  # Roll state

            # Every 1000 steps train the agent and save a checkpoint if a function is specified
            if step % 1000 == 0:
                self.agent.train(step)
                if checkpnt_func:
                    checkpnt_func(path, self.agent, step)

            sys.stdout.write("\rDone with %.1f percent of the training" % ((float(step) / float(max_steps)) * 100.0))
            sys.stdout.flush()

    def _mp_train_epochal(self, max_steps, max_subprocesses, checkpnt_func, path, rewards):
        raise NotImplementedError()


def _train(create_env, instance_ids, max_steps, pipes, visualize, rewards):
    pipes = {iid: p for iid, p in zip(instance_ids, pipes)}
    actions = {iid: None for iid in instance_ids}

    create_env = cloudpickle.loads(create_env)
    envs = {iid: create_env(rewards=rewards) for iid in instance_ids}

    for iid in instance_ids:
        state = envs[iid].reset()
        pipes[iid].send(RewardState(None, state))

    for step in range(max_steps):
        for iid in instance_ids:
            actions[iid] = pipes[iid].recv()
            if visualize:
                envs[iid].render()

            next_state, reward, done, _ = envs[iid].step(actions[iid])
            pipes[iid].send(RewardState(reward, None if done else next_state))

            if done:
                state = envs[iid].reset()
                pipes[iid].send(RewardState(None, state))