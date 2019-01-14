# ATARI Taxi Game Ai Bot
## Openai Gym and Reinforcement Learning
### Creating the environments
To create the environment use the following code snippet:
```
import numpy as np
import gym
```
### Actions
There are six actions: NORTH, SOUTH, EAST, WEST, PICKUP AND DROP represented as
integers.

### Environment Attributes

This class contains the following important attributes:

- `nS` :: number of states
- `nA` :: number of actions
- `P` :: transitions, rewards, terminals

The `P` attribute will be the most important for your implementation
of value iteration and policy iteration. This attribute contains the
model for the particular map instance. It is a dictionary of
dictionary of lists with the following form:

```
P[s][a] = [(prob, nextstate, reward, is_terminal), ...]
```
##
### Running a optimal policy after value iteration
Value Iteration computes the optimal state value function by iteratively improving the estimate of V(s).
The algorithm initializes V(s) to arbitrary random values then it repeatedly updates the Q(s,a) and
V(s) values until they converge. Using these value, optimal policy is generated.
The same has been done in taxi_bot.py
V is the Value of each State.
Q is Q function.
