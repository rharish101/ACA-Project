---
title: "Learn to Play Atari Games using Reinforcement Learning"
date: 2017-10-10 18:40:00 +0530
author: Harish Rajagopal
tags: <!-- Add tags for your project here -->
 - sem2
 - project
 - rl
 - atari
 - pong
categories: <!-- Add categores for your project here -->
 - project
 - ml
 - rl
---
This is an agent made to learn playing Atari Pong, which runs on the principles of Reinforcement Learning. The code is written in Python using Keras and OpenAI Gym.

The agent uses the technique of Deep Q-Networks (DQNs) to learn using a Convolutional Neural Network (CNN), which uses off-policy Temporal Difference control, also known as Q-Learning, on a Model-free Markov Decision Process (MDP). A key part of DQNs is the use of Experience Replay and Fixed Targets, which is being used here to stabilize the CNN and help it to converge.

#### Github: [Reinforcement Learning](https://github.com/rharish101/ACA-Project)
