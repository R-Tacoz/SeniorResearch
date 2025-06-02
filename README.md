# Towards Generalized Ground-based Multi-Agent Robotic Search from Graph Neural Networks
This is Rocco Zhang (Syslab Yilmaz pd 3) and Rishabh Kumaran's (Syslab Yilmaz pd 1) Senior Research Project for the 2025-2026 school year. Below is an abstract of our project. \

## Abstract
Multi-agent robotic systems trained on reinforcement learning have the potential to improve efficiency in deterministic or single-agent situations. In the context of search and rescue (SAR), coordinating multiple autonomous agents to locate a target in a complex, partially observable environment presents both operational and communicational challenges. Our research aims to create a multi-agent reinforcement learning (MARL) framework in which a team of robots collaboratively searches for a stationary target in a continuous environment containing obstacles. Additionally, to encourage cooperation and greater spatial awareness, a graph neural network (GNN) is employed to represent inter-agent communication. Training is conducted using a centralized learning, decentralized execution paradigm, and our preliminary results suggest that our algorithm has significantly better results than a deterministic approach.

## Setup and Environment Commands
* Export environment using "conda env export --no-builds | findstr /V "prefix" > environment.yml"
* Import environment using "conda env import -f environment.yml"
* Update environment using "conda env update -f environment.yml"
