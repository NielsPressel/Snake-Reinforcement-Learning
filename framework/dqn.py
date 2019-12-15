"""This module provides an implementation of a Double Deep Q-Learning Agent.

Typical usage example:

    agent = DQN(model, env.action_space.n)

    state = env.reset()
    for step in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.push_observation(Transition(state, action, reward, next_state if not done else None))
        agent.train(step)

        if done:
            state = env.reset()
        else:
            state = next_state
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from framework.core import Agent
from framework.policy import EpsilonGreedy
from framework.memory import PrioritizedExperienceReplay


class DQN(Agent):
    """Simple implementation of a deep Q-Learning Agent.

        This implementation is based on "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013).
        Also some code is taken from Huskarl (https://github.com/danaugrs/huskarl).

        Attributes:
            actions (int): Number of possible actions.
            optimizer (tf.keras.optimzers.Optimizer): The optimizer which will adjust the network's weights via
                                                      backpropagation.
            policy (framework.core.Policy): The policy for training.
            mem_size (int): Maximum size of the replay memory.
            memory (framework.core.Memory): The memory in which all transitions are stored.
            target_network_update (float): Determines how the target network's weights are adjusted / updated with the
                                           ones of the model.
            gamma (float): Discount rate for calculating the discounted return while training.
            batch_size (int): Batch size that will get fetched from memory for training.
            nsteps (int): Number of steps that fit into one trace in memory.
            training (bool): Determines whether to follow the training or the evaluation policy.
            model (tf.keras.models.Model): The base model that gets trained every iteration and that predicts the
                                           QValues while acting of the agent.
    """

    def __init__(self, model, actions, optimizer=None, policy=None, mem_size=100_000, target_update=10, gamma=0.99,
                 batch_size=64, nsteps=1):

        self.actions = actions
        self.optimizer = Adam(lr=3e-3) if optimizer is None else optimizer

        self.policy = EpsilonGreedy(0.1) if policy is None else policy

        self.mem_size = mem_size
        self.memory = PrioritizedExperienceReplay(mem_size, nsteps)

        self.target_network_update = target_update
        self.gamma = gamma
        self.batch_size = batch_size
        self.nsteps = nsteps
        self.training = True

        raw_output = model.layers[-1].output
        output_layer = Dense(self.actions, activation='linear')(raw_output)
        self.model = Model(inputs=model.input, outputs=output_layer)

        # Define loss function that computes the MSE between target Q-values and cumulative discounted rewards
        # If using PrioritizedExperienceReplay, the loss function also computes the TD error and updates the trace
        # priorities
        def masked_q_loss(data, y_pred):
            """Computes the MSE between the Q-values of the actions that were taken and	the cumulative discounted
            rewards obtained after taking those actions. Updates trace priorities if using PrioritizedExperienceReplay.
            """
            action_batch, target_qvals = data[:, 0], data[:, 1]
            seq = tf.cast(tf.range(0, tf.shape(action_batch)[0]), tf.int32)
            action_idxs = tf.transpose(tf.stack([seq, tf.cast(action_batch, tf.int32)]))
            qvals = tf.gather_nd(y_pred, action_idxs)
            if isinstance(self.memory, PrioritizedExperienceReplay):
                def update_priorities(_qvals, _target_qvals, _traces_idxs):
                    """Computes the TD error and updates memory priorities."""
                    td_error = np.abs((_target_qvals - _qvals).numpy())
                    _traces_idxs = (tf.cast(_traces_idxs, tf.int32)).numpy()
                    self.memory.update_priorities(_traces_idxs, td_error)
                    return _qvals

                qvals = tf.py_function(func=update_priorities, inp=[qvals, target_qvals, data[:, 2]], Tout=tf.float32)
            return tf.keras.losses.mse(qvals, target_qvals)

        self.model.compile(optimizer=self.optimizer, loss=masked_q_loss)

        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def save(self, filename, overwrite=False):
        """Saves the model parameters to the specified file."""
        self.model.save_weights(filename, overwrite=overwrite)

    def act(self, state):
        qvals = self.model.predict(np.array([state]))[0]
        return self.policy.act(qvals)

    def push_observation(self, transition):
        self.memory.put(transition)

    def train(self, step):
        """Trains the agent for one step."""
        if len(self.memory) == 0:
            return

        # Update target network
        if self.target_network_update >= 1 and step % self.target_network_update == 0:
            # Perform a hard update
            self.target_model.set_weights(self.model.get_weights())
        elif self.target_network_update < 1:
            # Perform a soft update
            mw = np.array(self.model.get_weights())
            tmw = np.array(self.target_model.get_weights())
            self.target_model.set_weights(self.target_network_update * mw + (1 - self.target_network_update) * tmw)

        # Train even when memory has fewer than the specified batch_size
        batch_size = min(len(self.memory), self.batch_size)

        # Sample batch_size traces from memory
        state_batch, action_batch, reward_batches, end_state_batch, not_done_mask = self.memory.get(batch_size)

        # Compute the value of the last next states
        target_qvals = np.zeros(batch_size)
        non_final_last_next_states = [es for es in end_state_batch if es is not None]

        if len(non_final_last_next_states) > 0:
            q_values = self.model.predict_on_batch(np.array(non_final_last_next_states))
            actions = np.argmax(q_values, axis=1)
            target_q_values = self.target_model.predict_on_batch(np.array(non_final_last_next_states))
            selected_target_q_vals = target_q_values[range(len(target_q_values)), actions]
            selected_target_q_vals = self.target_model.predict_on_batch(np.array(non_final_last_next_states)).max(1)
            non_final_mask = list(map(lambda s: s is not None, end_state_batch))
            target_qvals[non_final_mask] = selected_target_q_vals

        # Compute n-step discounted return
        # If episode ended within any sampled nstep trace - zero out remaining rewards
        for n in reversed(range(self.nsteps)):
            rewards = np.array([b[n] for b in reward_batches])
            target_qvals *= np.array([t[n] for t in not_done_mask])
            target_qvals = rewards + (self.gamma * target_qvals)

        # Compile information needed by the custom loss function
        loss_data = [action_batch, target_qvals]

        # If using PrioritizedExperienceReplay then we need to provide the trace indexes
        # to the loss function as well so we can update the priorities of the traces
        if isinstance(self.memory, PrioritizedExperienceReplay):
            loss_data.append(self.memory.last_traces_idxs())

        # Train model
        self.model.train_on_batch(np.array(state_batch), np.stack(loss_data).transpose())
