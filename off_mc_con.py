import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
	sys.path.append("../") 
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def create_random_policy(nA):
	"""
	Creates a random policy function.

	Args:
		nA: Number of actions in the environment.

	Returns:
		A function that takes an observation as input and returns a vector
		of action probabilities
	"""
	A = np.ones(nA, dtype=float) / nA
	def policy_fn(observation):
		return A
	return policy_fn

def create_greedy_policy(Q):
	"""
	Creates a greedy policy based on Q values.

	Args:
		Q: A dictionary that maps from state -> action values

	Returns:
		A function that takes an observation as input and returns a vector
		of action probabilities.
	"""

	def policy_fn(observation):
		actions = np.argwhere(Q[observation] == np.amax(Q[observation]))
		actions = actions.flatten().tolist()
		policy = np.zeros(env.action_space.n)
		for action in actions:	
			policy[action] = 1
		policy /= np.sum(policy)
		return policy
	return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
	"""
	Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
	Finds an optimal greedy policy.

	Args:
		env: OpenAI gym environment.
		num_episodes: Nubmer of episodes to sample.
		behavior_policy: The behavior to follow while generating episodes.
			A function that given an observation returns a vector of probabilities for each action.
		discount_factor: Lambda discount factor.

	Returns:
		A tuple (Q, policy).
		Q is a dictionary mapping state -> action values.
		policy is a function that takes an observation as an argument and returns
		action probabilities. This is the optimal greedy policy.
	"""

	# The final action-value function.
	# A dictionary that maps state -> action values
	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	# Our greedily policy we want to learn
	target_policy = create_greedy_policy(Q)

	# Keeps track of sum and count of returns for each state
	# to calculate an average. We could use an array to save all
	# returns (like in the book) but that's memory inefficient.
	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)

	for e in range(num_episodes):
		#Print episode no.
		print("\rEpisode: {}/{}".format(e+1,num_episodes),end="")
		sys.stdout.flush()

		#Run the episode and find rewards   
		state = env.reset()
		Q_new = defaultdict(lambda: np.zeros(env.action_space.n))
		states_act_enc = [] # states and actions encountered array
		#Adding rewards found to Q_new  
		while True:
			pol = behavior_policy(state)
			action = np.random.choice(np.arange(len(pol)), p=pol)
			states_act_enc.append((state,action))
			state_new, Q_new[state][action], done, info = env.step(action)
			if done:
				break
			state = state_new

		#Find the return and update returns_sum 
	# Implement this!
		rew_return = 0
		for i in range(len(states_act_enc)):
			state = states_act_enc[len(states_act_enc) - i - 1][0]
			action = states_act_enc[len(states_act_enc) - i - 1][1]
			Q_new[state][action] += rew_return * discount_factor * target_policy(state)[action] / behavior_policy(state)[action]
			returns_sum[(state,action)] += Q_new[state][action]
			returns_count[(state,action)] += 1
			rew_return = Q_new[state][action]
			Q[state][action] = returns_sum[(state,action)] / returns_count[(state,action)]

	print()
	return Q, target_policy

random_policy = create_random_policy(env.action_space.n)
Q, policy = mc_control_importance_sampling(env, num_episodes=500000, behavior_policy=random_policy)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, action_values in Q.items():
	action_value = np.max(action_values)
	V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")

