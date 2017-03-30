import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factor=1.0):
	"""
	Value Iteration Algorithm.

	Args:
		env: OpenAI environment. env.P represents the transition probabilities of the environment.
		theta: Stopping threshold. If the value of all states changes less than theta
			in one iteration we are done.
		discount_factor: lambda time discount factor.

	Returns:
		A tuple (policy, V) of the optimal policy and the optimal value function.        
	"""

	V = np.zeros(env.nS)
	policy = np.zeros([env.nS, env.nA])
	V_new = np.zeros(env.nS)
	while True:
		V_new = np.zeros(env.nS)
		for s in range(env.nS):
			max_val = -9999	
			for a in range(env.nA):
				val = env.P[s][a][0][2] + discount_factor*env.P[s][a][0][0]*V[env.P[s][a][0][1]]
				if val>max_val:
					max_val = val 
			V_new[s] = max_val
		if sum((V-V_new)*(V-V_new))<theta:
			V = V_new	
			break
		else:
			V = V_new
	# Finding optimal policy:	
	for s in range(env.nS):
		actions = np.zeros(env.nA)
		max_val = -9999
		for a in range(env.nA):
			if V[env.P[s][a][0][1]]>max_val:
				actions = np.zeros(env.nA)
				actions[a] = 1.0
				max_val = V[env.P[s][a][0][1]]
			elif V[env.P[s][a][0][1]]==max_val:
				actions[a] = 1.0
		actions = actions/sum(actions)
		for a in range(env.nA):
			policy[s][a] = actions[a]
	return policy, V

policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
