import numpy as np
import time
import gym

def execute_policy(policy):
	start = env.reset()
	env.render()
	total_reward = 0
	while True:
		start, reward, done, _ = env.step(int(policy[start]))
		env.render()
		time.sleep(0.1)
		total_reward += reward
		if done:
			break
	return total_reward

def cal_policy(v_star):
	policy = np.zeros(env.env.nS)
	for s in range (env.env.nS):
		qa = np.zeros(env.env.nA)
		for a in range (env.env.nA):
			for tuples in env.env.P[s][a]:
				p, v_new, r,_ = tuples
				qa[a] += (p*(r+v_star[v_new]))
		policy[s] = np.argmax(qa)
	return policy
def val_iter(env):
	values = np.zeros(env.env.nS)
	total_iterations = 10000
	break_point = 1e-2
	for times in range (total_iterations):
		prev_v = np.copy(values)
		for s in range(env.env.nS):
			qa = [sum([p*(r+prev_v[v_new]) for  p, v_new, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)]
			values[s] = max(qa)
			check_break = (np.sum(np.fabs(values - prev_v)))
			#print (check_break)
			if ( check_break <= break_point):
				print ("converged at iter %s"%(times))
				break
	return values
env = gym.make('Taxi-v2')
v_star = val_iter(env)
prime_policy = cal_policy(v_star)
#print (prime_policy)
reward = execute_policy(prime_policy)
#print (reward)
env.close()
