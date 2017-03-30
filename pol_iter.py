import numpy as np
import pprint
import sys
if "../" not in sys.path:
	sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv
pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()
# Taken from Policy Evaluation Exercise!

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
	"""
	Evaluate a policy given an environment and a full description of the environment's dynamics.

	Args:
		policy: [S, A] shaped matrix representing the policy.
		env: OpenAI env. env.P represents the transition probabilities of the environment.
			env.P[s][a] is a (prob, next_state, reward, done) tuple.
		theta: We stop evaluation one our value function change is less than theta for all states.
		discount_factor: lambda discount factor.

	Returns:
		Vector of length env.nS representing the value function.
	"""
	# Start with a random (all 0) value function
	V = np.zeros(env.nS)
	while True:
		delta = 0
		# For each state, perform a "full backup"
		for s in range(env.nS):
			v = 0
			# Look at the possible next actions
			for a, action_prob in enumerate(policy[s]):
				# For each action, look at the possible next states...
				for  prob, next_state, reward, done in env.P[s][a]:
					# Calculate the expected value
					v += action_prob * prob * (reward + discount_factor * V[next_state])
			# How much our value function changed (across any states)
			delta = max(delta, np.abs(v - V[s]))
			V[s] = v
		# Stop evaluating once our value function change is below a threshold
		if delta < theta:
			break
	return np.array(V)
def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
	"""
	Policy Improvement Algorithm. Iteratively evaluates and improves a policy
	until an optimal policy is found.

	Args:
		env: The OpenAI envrionment.
		policy_eval_fn: Policy Evaluation function that takes 3 arguments:
			policy, env, discount_factor.
		discount_factor: Lambda discount factor.

	Returns:
		A tuple (policy, V). 
		policy is the optimal policy, a matrix of shape [S, A] where each state s
		contains a valid probability distribution over actions.
		V is the value function for the optimal policy.

	"""
	# Start with a random policy
	policy = np.ones([env.nS, env.nA]) / env.nA
	V = np.zeros(env.nS)
	pol_new = np.ones([env.nS, env.nA]) / env.nA 
	# """
	while True:
		V = policy_eval_fn(pol_new,env)
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
				pol_new[s][a] = actions[a]
		delta = pol_new - policy	
		if sum(sum(delta*delta))==0:	
			break
		else:
			policy = pol_new
	# """
	return policy,V 
policy, v = policy_improvement(env)
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
