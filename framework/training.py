from framework.core import Transition

import sys
import os
import re
import multiprocessing as mp
import cloudpickle

from collections import namedtuple


RewardState = namedtuple('RewardState', ['reward', 'state'])


class Training:

    def __init__(self, create_env_func, agent):
        self.create_env_func = create_env_func
        self.agent = agent

    def train(self, max_steps=100_000, instances=1, visualize=False, plot_func=None, max_subprocesses=0,
              checkpnt_func=None, path="", rewards=None, resume=False):
        if max_subprocesses == 0:
            self._sp_train(max_steps, instances, visualize, plot_func, checkpnt_func, path, rewards, resume)
        else:
            self._mp_train(max_steps, instances, max_subprocesses, visualize, plot_func, checkpnt_func, path, rewards, resume)

    def evaluate(self, max_steps=10_000, visualize=False, plot_func=None):
        self.agent.training = False
        env = self.create_env_func()
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
        """Trains using a single process."""
        # Keep track of rewards per episode per instance
        episode_reward_sequences = [[] for i in range(instances)]
        episode_step_sequences = [[] for i in range(instances)]
        episode_rewards = [0] * instances

        # Create and initialize environment instances
        envs = [self.create_env_func(rewards=rewards) for i in range(instances)]
        states = [env.reset() for env in envs]
        curr_step = 0

        if resume:
            full_path = os.path.join(path, 'checkpoints')
            file_list = [f for f in os.listdir(full_path)]
            for f in file_list:
                if f.find('.dat.data') != -1:
                    file = os.path.join(full_path, f.split('.data')[0])
                    self.agent.load(file)
                    curr_step = int(re.findall(r'\d+', f.split('.data')[0])[0])

        for step in range(curr_step, max_steps):
            for i in range(instances):
                if visualize:
                    envs[i].render()
                action = self.agent.act(states[i])
                next_state, reward, done, _ = envs[i].step(action)
                self.agent.push_observation(Transition(states[i], action, reward, None if done else next_state))
                episode_rewards[i] += reward
                if done:
                    episode_reward_sequences[i].append(episode_rewards[i])
                    episode_step_sequences[i].append(step)
                    episode_rewards[i] = 0
                    if plot: plot(episode_reward_sequences, episode_step_sequences)
                    states[i] = envs[i].reset()
                else:
                    states[i] = next_state
            # Perform one step of the optimization
            self.agent.train(step)
            sys.stdout.write("\rDone with %.1f percent of the training" % ((float(step) / float(max_steps)) * 100.0))
            sys.stdout.flush()

            if checkpnt_func:
                if step % 1000 == 0:
                    checkpnt_func(path, self.agent, step)

        if plot:
            plot(episode_reward_sequences, episode_step_sequences, done=True)

    def _mp_train(self, max_steps, instances, max_subprocesses, visualize, plot, checkpnt_func, path, rewards, resume):
        if max_subprocesses is None:
            max_subprocesses = mp.cpu_count()

        n_processes = min(instances, max_subprocesses)

        instances_per_process = [instances // n_processes] * n_processes
        leftover = instances % n_processes
        if leftover > 0:
            for i in range(leftover):
                instances_per_process[i] += 1

        instance_ids = [list(range(i, instances, n_processes))[:ipp] for i, ipp in enumerate(instances_per_process)]

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

        print(f"Starting {n_processes} process(es) for {instances} environment instance(s)... {instance_ids}")
        for p in processes:
            p.start()

        episode_reward_sequences = [[] for i in range(instances)]
        episode_step_sequences = [[] for i in range(instances)]
        episode_rewards = [0] * instances

        rss = [None] * instances
        last_actions = [None] * instances

        for step in range(max_steps):
            step_done = [False] * instances

            while sum(step_done) < instances:
                awaiting_pipes = [p for iid, p in enumerate(pipes) if step_done[iid] == 0]
                ready_pipes = mp.connection.wait(awaiting_pipes, timeout=None)
                pipe_indices = [pipes.index(rp) for rp in ready_pipes]

                pipe_indices.sort()
                for iid in pipe_indices:
                    rs = pipes[iid].recv()

                    if rss[iid] is not None:
                        experience = Transition(rss[iid].state, last_actions[iid], rs.reward, rs.state)
                        self.agent.push_observation(experience)
                        step_done[iid] = True
                    rss[iid] = rs

                    if rs.state is None:
                        rss[iid] = None
                        episode_reward_sequences[iid].append(episode_rewards[iid])
                        episode_step_sequences[iid].append(step)
                        episode_rewards[iid] = 0
                        if plot:
                            plot(episode_reward_sequences, episode_step_sequences)
                    else:
                        action = self.agent.act(rs.state)
                        last_actions[iid] = action
                        try:
                            pipes[iid].send(action)
                        except BrokenPipeError as bpe:
                            if step < (max_steps - 1):
                                raise bpe
                        if rs.reward:
                            episode_rewards[iid] += rs.reward

            self.agent.train(step)
            sys.stdout.write("\rDone with %.1f percent of the training" % ((float(step) / float(max_steps)) * 100.0))
            sys.stdout.flush()

            if checkpnt_func:
                if step % 1000 == 0:
                    checkpnt_func(path, self.agent, step)

        if plot:
            plot(episode_reward_sequences, episode_step_sequences, done=True)


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