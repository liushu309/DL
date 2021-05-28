import numpy as np
import pandas as pd


# pi 初始状态概率向量
# a  状态转移概率矩阵
# b  状态观测概率矩阵
# obs 观测向量
def viterbi(pi, a, b, obs):
    # 3
    nStates = np.shape(b)[0]
    # 14  [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1]
    T = np.shape(obs)[0]

    # init blank path [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    path = path = np.zeros(T,dtype=int)
    # delta --> highest probability of any path that reaches state i
    #  [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    #  [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    phi = np.zeros((nStates, T))

    # init delta and phi
    # delta
    # [[0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
    # [0.04 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
    # [0.56 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]]
    # pi = [0, 0.2, 0.8]
    # b[:, obs[0]] = [0 0.2 0.7]
    # * 对应元素相乘
    delta[:, 0] = pi * b[:, obs[0]]


    print('\nStart Walk Forward\n')
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))

    # find optimal path
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
        print('path[{}] = {}'.format(t, path[t]))

    return path, delta, phi

obs_map = {'Cold':0, 'Hot':1}
obs = np.array([1,1,0,1,0,0,1,0,1,1,0,0,0,1])

inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

print("Simulated Observations:\n",pd.DataFrame(np.column_stack([obs, obs_seq]),columns=['Obs_code', 'Obs_seq']) )

pi = [0.6,0.4] # initial probabilities vector
states = ['Cold', 'Hot']
hidden_states = ['Snow', 'Rain', 'Sunshine']
pi = [0, 0.2, 0.8]
state_space = pd.Series(pi, index=hidden_states, name='states')
print(state_space)
a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.3, 0.3, 0.4]
a_df.loc[hidden_states[1]] = [0.1, 0.45, 0.45]
a_df.loc[hidden_states[2]] = [0.2, 0.3, 0.5]
print("\n HMM matrix:\n", a_df)
a = a_df.values

observable_states = states
b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [1,0]
b_df.loc[hidden_states[1]] = [0.8,0.2]
b_df.loc[hidden_states[2]] = [0.3,0.7]
print("\n Observable layer  matrix:\n",b_df)
b = b_df.values


path, delta, phi = viterbi(pi, a, b, obs)
state_map = {0:'Snow', 1:'Rain', 2:'Sunshine'}
state_path = [state_map[v] for v in path]
pd.DataFrame().assign(Observation=obs_seq).assign(Best_Path=state_path)

print(obs)
print(path)
# print(obs +1)



# 调用库
import numpy as np
from hmmlearn import hmm

start_probability = np.array([1/n_states, 1/n_states, 1/n_states])

transition_probability = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])


emission_probability = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8]
])


seen = np.array([[0, 1, 0]]).T
logprob, box = model.decode(seen, algorithm="viterbi")
print(logprob, box)
print(model.score(seen))
print("The ball picked:", ", ".join(map(lambda x: observations[x], seen)))
print("The hidden box", ", ".join(map(lambda x: states[x], box)))

box2 = model.predict(seen)
print("The ball picked:", ", ".join(map(lambda x: observations[x], seen)))
print("The hidden box", ", ".join(map(lambda x: states[x], box2)))






