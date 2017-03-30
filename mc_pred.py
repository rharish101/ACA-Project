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
def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
	"""
	Monte Carlo prediction algorithm. Calculates the value function	for a given policy using sampling.

	Args:
		policy: A function that maps an observation to action probabilities.
		env: OpenAI gym environment.
		num_episodes: Nubmer of episodes to sample.
		discount_factor: Lambda discount factor.

	Returns:
		A dictionary that maps from state -> value.
		The state is a tuple and the value is a float.
	"""
	# Keeps track of sum and count of returns for each state
	# to calculate an average. We could use an array to save all
	# returns (like in the book) but that's memory inefficient.
	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)

	# The final value function
	V = defaultdict(float)
	
	for e in range(num_episodes):
		#Print episode no.
		print("\rEpisode: {}/{}".format(e+1,num_episodes),end="")
		sys.stdout.flush()	
		
		state = env.reset()
		returns_count[state] += 1	
		V_new = defaultdict(float)	
		states_enc = [state] # states encountered array	
		
		while True:
			action = int((policy(state))[1])
			state_new, V_new[state], done, info = env.step(action)
			if done:
				break
			state = state_new
			returns_count[state] += 1	
			states_enc.append(state)	
		
		rew_return = 0	
		# update with return for each state encountered	
		for i in range(len(states_enc)):
			state = states_enc[len(states_enc) - i - 1] 	
			V_new[state] += rew_return * discount_factor
			returns_sum[state] += V_new[state]
			rew_return = V_new[state]
			V[state] = returns_sum[state] / returns_count[state] 
	
	print()	
	return V

def sample_policy(observation):
	"""
	A policy that sticks if the player score is > 20 and hits otherwise.
	"""
	score, dealer_score, usable_ace = observation
	return np.array([1.0, 0.0]) if score >= 20 else np.array([0.0, 1.0])

V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")

