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

def make_epsilon_greedy_policy(Q, epsilon, nA):
	"""
	Creates an epsilon-greedy policy based on a given Q-function and epsilon.

	Args:
		Q: A dictionary that maps from state -> action-values.
			Each value is a numpy array of length nA (see below)
		epsilon: The probability to select a random action . float between 0 and 1.
		nA: Number of actions in the environment.

	Returns:
		A function that takes the observation as an argument and returns
		the probabilities for each action in the form of a numpy array of length nA.

	"""
	def policy_fn(observation):
		# Implement this!
		policy = epsilon * np.ones(nA) / nA
		i_req = 0
		for i in range (nA):
			if Q[observation][i] > Q[observation][i_req]:
				i_req = i
		policy[i_req] += 1 - epsilon
		return policy
	return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
	"""
		Monte Carlo Control using Epsilon-Greedy policies.
		Finds an optimal epsilon-greedy policy.

	Args:
		env: OpenAI gym environment.
		num_episodes: Nubmer of episodes to sample.
		discount_factor: Lambda discount factor.
		epsilon: Chance the sample a random action. Float betwen 0 and 1.

	Returns:
		A tuple (Q, policy).
		Q is a dictionary mapping state -> action values.
		policy is a function that takes an observation as an argument and returns
		action probabilities
	"""

	# Keeps track of sum and count of returns for each state
	# to calculate an average. We could use an array to save all
	# returns (like in the book) but that's memory inefficient.
	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)

	# The final action-value function.
	# A nested dictionary that maps state -> (action -> action-value).
	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	# The policy we're following
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	epsilon_init = epsilon
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
			pol = policy(state)
			action = np.random.choice(np.arange(len(pol)), p=pol)	
			states_act_enc.append((state,action))	
			state_new, Q_new[state][action], done, info = env.step(action)
			if done:
				break
			state = state_new

		#Find the return and update returns_sum	
		rew_return = 0
		for i in range(len(states_act_enc)):
			state = states_act_enc[len(states_act_enc) - i - 1][0]
			action = states_act_enc[len(states_act_enc) - i - 1][1]
			Q_new[state][action] += rew_return * discount_factor
			returns_sum[(state,action)] += Q_new[state][action]
			returns_count[(state,action)] += 1
			rew_return = Q_new[state][action]	
			Q[state][action] = returns_sum[(state,action)] / returns_count[(state,action)]
		
	print()
	return Q, policy

Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
	action_value = np.max(actions)
	V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")

