# Snake Reinforcement Learning

## Prerequisites

* Python installation (This project is only tested on python 3.6.8 and Windows)
* pygame (1.9.6)
* tensorflow-gpu (2.1.0)
    * CUDA Toolkit (10.1)
    * cuDNN for CUDA 10.1 (7.6.8)
    
## Resources

* Huskarl (https://github.com/danaugrs/huskarl)
* Google DeepMind research
* Python Machine Learning (Sebastian Raschka, Vahid Mirjalili, 2017)
* Praxiseinstieg Deep Learning (Ramon Wartala, 2018)

## Project structure

The base of the project consists of four abstract base classes (Agent, Policy, Memory, Environment). These
classes represent the common objects used in Reinforcement Learning. The agent is responsible for collecting data,
choosing the actions that define the agent's movement inside the environment and training the underlying neural 
network to find a optimal policy for the given task. By moving inside the environment the agent gets rewarded or
punished based on its chosen action. After every single step the start state, the action, the reward and the
resulting state are pushed into memory to later learn from that data. The policy class exposes the ability to 
manipulate the action chosen by the agent eg. for getting some noise into the training process.

If you want to use this project for solving your own Reinforcement Learning problem you only have to implement
your environment by deriving yours from the base environment class. For training you can either use the built-in
classes (DQN, ReplayMemory, EpsilonGreedy, Training, Evaluation) or write your own by deriving from the base classes.

For more complex problems you should definitely use at least one GPU for training.

## Snake Results

* Snake Simple: Average length of 16.6, trained on ExperimentalDQN
* Snake Abstract: Average length of 2.9, often gets stuck at running in circles

More precise description of the results coming soon...
