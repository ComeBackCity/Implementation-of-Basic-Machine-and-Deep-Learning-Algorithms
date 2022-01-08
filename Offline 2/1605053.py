#!/usr/bin/env python
# coding: utf-8

# In[1]:


# package imports
import numpy as np
import pathlib
from itertools import islice
from scipy.linalg import eig
from scipy.stats import norm
from numpy import float64, longdouble
from typing import List


# In[2]:


# get stationary dustribution of transition matrix
# from stack overflow
def get_stationary_distibution(state_transition_matrix: np.ndarray) -> np.ndarray:
    S, U = eig(state_transition_matrix.T)
    stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
    stationary = stationary / np.sum(stationary)
    return stationary


# In[3]:


def estimate_sequence(state_transition_matrix:np.ndarray, gaussian_params: np.ndarray, initial_state_probability:np.ndarray, observations:np.ndarray, state_count: int) -> List[int]:
    observation_count = observations.shape[0]
    state_probability_matrix = np.ndarray((state_count, observation_count), dtype=float64)
    path = np.ndarray((state_count, observation_count-1), dtype=int)
    emission_matrix = norm(loc=gaussian_params[0,:], scale=gaussian_params[1,:]).pdf(observations).T
    state_probability_matrix[:,0] = np.log(initial_state_probability) + np.log(emission_matrix[:,0])
    for i in range(1, observation_count):
        prob = state_probability_matrix[:,i-1] + np.log(state_transition_matrix) + np.log(emission_matrix[:,i].reshape(-1,1))
        path[:,i-1] = np.argmax(prob, axis=1)
        state_probability_matrix[:,i] = np.max(prob, axis=1)

    out_path = [-1] * observation_count
    sink_index = np.argmax(state_probability_matrix[:,-1])
    out_path[-1] = sink_index
    for i in range(observation_count-2,-1,-1):
        sink_index = path[sink_index, i]
        out_path[i] = sink_index

    return out_path


# In[4]:


# Viterbi algorithm
def viterbi(state_transition_matrix:np.ndarray, gaussian_params: np.ndarray, initial_state_probability:np.ndarray, observations:np.ndarray, state_count: int, state_converter:dict) -> list:
    path = estimate_sequence(state_transition_matrix, gaussian_params, initial_state_probability, observations, state_count)
    return [state_converter[index] for index in path]


# In[5]:


def forward(state_transition_matrix: np.ndarray, state_count:int, observation_count:int , emission_matrix: np.ndarray, initial_transition_probability: np.ndarray) -> np.ndarray:
    alpha = np.ndarray((state_count, observation_count), dtype=float64)
    alpha[:,0] = initial_transition_probability * emission_matrix[:,0]
    alpha[:,0] /= np.sum(alpha[:,0])

    for i in range(1, observation_count):
        prob = alpha[:,i-1] * state_transition_matrix * emission_matrix[:,i].reshape(-1,1)
        alpha[:,i] = np.sum(prob, axis=1)
        alpha[:,i] /= np.sum(alpha[:,i])

    return alpha


# In[6]:


def backward(state_transition_matrix: np.ndarray, state_count:int, observation_count: int, emission_matrix: np.ndarray) -> np.ndarray:
    beta = np.ndarray((state_count, observation_count), dtype=float64)
    beta[:,-1] = 1

    for i in range(observation_count-2, -1, -1):
        for k in range(state_count):
            prob = sum(
                beta[l, i + 1]
                * state_transition_matrix[k, l]
                * emission_matrix[l, i]
                for l in range(state_count)
            )

            beta[k,i] = prob
        beta[:,i] /= np.sum(beta[:,i])

    return beta


# In[7]:


# Baum-Welch Learning
def baum_welch(state_transition_matrix: np.ndarray, state_count:int, observations: np.ndarray, gaussian_params:np.ndarray, initial_transition_probability: np.ndarray, no_of_iterations:int):
    observation_count = observations.shape[0]

    for _ in range(no_of_iterations):
        emission_matrix = norm(loc=gaussian_params[0,:], scale=gaussian_params[1,:]).pdf(observations).T
        alpha = forward(state_transition_matrix, state_count, observation_count, emission_matrix, initial_transition_probability)
        beta = backward(state_transition_matrix, state_count, observation_count, emission_matrix)
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=0)

        xi = np.ndarray((observation_count-1, state_count, state_count), dtype=longdouble)
        for t in range(observation_count-1):
            for i in range(state_count):
                for j in range(state_count):
                    xi[t,i,j] = alpha[i,t] * state_transition_matrix[i,j] * beta[j,t+1] * emission_matrix[j,t+1]

        xi /= np.sum(xi, axis=(1,2)).reshape(-1,1)[:,np.newaxis]

        state_transition_matrix = np.sum(xi, axis=0)
        state_transition_matrix /= np.sum(state_transition_matrix, axis=1).reshape(-1,1)

        gaussian_params[0,:] = np.sum(gamma * observations.T, axis=1) / np.sum(gamma, axis=1)
        gaussian_params[1,:] = np.sqrt(np.sum(gamma * (observations.T - gaussian_params[0,:].reshape(-1,1)) ** 2, axis=1) / np.sum(gamma, axis=1))

    return state_transition_matrix, gaussian_params


# In[8]:


# read data
observed_states = np.loadtxt('./Input/data.txt', dtype=float).reshape(-1,1)
# observed_states.shape


# In[9]:


# read parameters
with open('./Input/parameters.txt.txt', 'r') as f:
    no_of_states = int(f.readline())

with open('./Input/parameters.txt.txt', 'r') as lines:
    transition_matrix = np.genfromtxt(islice(lines, 1, 1+no_of_states))

with open('./Input/parameters.txt.txt', 'r') as lines:
    gaussian_parameters = np.genfromtxt(islice(lines, 1+no_of_states, 1+2*no_of_states))

# gaussian_parameters[1,:] = np.sqrt(gaussian_parameters[1,:])
# gaussian_parameters


# In[10]:


initial_distribution = get_stationary_distibution(transition_matrix)
index_state_map = {
    0: '\"El Nino\"',
    1: '\"La Nina\"'
}
# initial_distribution.shape


# In[11]:


hidden_path = viterbi(transition_matrix, gaussian_parameters, initial_distribution, observed_states, no_of_states, index_state_map)


# In[12]:


# matching output and writing to file
viterbi_output = []
with open('./Output/states_Viterbi_wo_learning.txt', 'r') as f:
    for line in f.readlines():
        viterbi_output.append(line.rstrip('\n'))

match = 0

for item1, item2 in zip(viterbi_output, hidden_path):
    if item1 == item2:
        match += 1

print(match)

pathlib.Path('./my_output').mkdir(parents=True, exist_ok=True)

with open('./my_output/states_Viterbi_wo_learning.txt', 'w') as f:
    for item in hidden_path:
        f.write(item+'\n')


# In[13]:


a, b = baum_welch(transition_matrix, no_of_states, observed_states, gaussian_parameters, initial_distribution, 5)


# In[14]:


# a


# In[15]:


# b


# In[16]:


hidden_path_after_learning = viterbi(a, b, initial_distribution, observed_states, no_of_states, index_state_map)


# In[17]:


# matching output and writing to file
viterbi_output = []
with open('./Output/states_Viterbi_after_learning.txt', 'r') as f:
    for line in f.readlines():
        viterbi_output.append(line.rstrip('\n'))

match = 0

for item1, item2 in zip(viterbi_output, hidden_path_after_learning):
    if item1 == item2:
        match += 1

print(match)

pathlib.Path('./my_output').mkdir(parents=True, exist_ok=True)

with open('./my_output/states_Viterbi_after_learning.txt', 'w') as f:
    for item in hidden_path:
        f.write(item+'\n')

with open('./my_output/learned_parameters.txt', 'w') as f:
    f.write('Transition matrix:\n')
    np.savetxt(f, a, fmt='%1.7f')
    f.write('\nGaussian parameters:\n')
    np.savetxt(f, b, fmt='%1.6f')

