This repository is about training an agent to play
[French Tarot game](https://en.wikipedia.org/wiki/French_tarot]).
The game rule where taken from the [Official French Tarot Rules](http://www.fftarot.fr/assets/documents/R-RO201206.pdf)
from the French Tarot Federation.

The aim is to train this agent based on DQN, although this is not
implemented yet.

# Game introduction
TODO

# Repository structure
TODO

# Results
This section presents the findings obtained throughout the various
experiments.

## All-random games
We made agents play randomly for 1000 independent games and we made sure
each agent gets to be the starting player an equal amounts of times.
The total scores were as follows:
- Player 0: 10201
- Player 1: 16641
- Player 2: -33599
- Player 3: 6757

# TODO
1. give short introduction to game's rules
2. describe repository structure
3. rerun random experiment with different seed 
4. implement DQN-based agent and train it
5. make agent play against random agent
6. make agent play against human