#!/usr/bin/env python
# coding: utf-8

# ## Rice Kitchen Algorithm 
# #### Part 1: Hidden Markov Modeling 
# _At the start, nutrients such as β-carotene and retinol are the basic requirements to begin metabolism in the intestinal mucosa (M). In the stomach (M), β-carotene and retinol are catabolized and transformed to create retinol esters and transported to liver cells, also known as hepatocytes (H). The liver accepts the retinol esters and if necessary stores them in hepatic stellate cells, otherwise they are converted back into retinol and proceed to peripheral cells using Cellular Retinol Binding Protein (CRBP), the end of theMarkov Model. From there, the peripheral cells transport to phototransduction in the eye and complete the Retinoid (Visual) Cycle, the end of the agent based model._
# 

# #### Step 1:
# _Import Packages_

# In[2]:


#imports
import pandas as pd
import numpy as np
import csv
import sympy as sym
import matplotlib.pyplot as plt


# #### Step 2: 
# _Create Class Function for Probability Vectors_ 

# In[3]:


## Designing Markov Model 
class ProbabilityVector:
    def __init__(self, probabilities: dict):
        states = probabilities.keys()
        probs  = probabilities.values()
        
        assert len(states) == len(probs), "The probabilities must match the states."
        assert len(states) == len(set(states)), "The states must be unique."
        assert abs(sum(probs) - 1.0) < 1e-12, "Probabilities must sum up to 1."
        assert len(list(filter(lambda x: 0 <= x <= 1, probs))) == len(probs),             "Probabilities must be numbers from [0, 1] interval."
        
        self.states = sorted(probabilities)
        self.values = np.array(list(map(lambda x: 
            probabilities[x], self.states))).reshape(1, -1)
        
    @classmethod
    def initialize(cls, states: list):
        size = len(states)
        rand = np.random.rand(size) / (size**2) + 1 / size
        rand /= rand.sum(axis=0)
        return cls(dict(zip(states, rand)))
    
    @classmethod
    def from_numpy(cls, array: np.ndarray, state: list):
        return cls(dict(zip(states, list(array))))

    @property
    def dict(self):
        return {k:v for k, v in zip(self.states, list(self.values.flatten()))}

    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.states, index=['probability'])

    def __repr__(self):
        return "P({}) = {}.".format(self.states, self.values)

    def __eq__(self, other):
        if not isinstance(other, ProbabilityVector):
            raise NotImplementedError
        if (self.states == other.states) and (self.values == other.values).all():
            return True
        return False

    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError("Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index])

    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityVector):
            return self.values * other.values
        elif isinstance(other, (int, float)):
            return self.values * other
        else:
            NotImplementedError

    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)

    def __matmul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityMatrix):
            return self.values @ other.values

    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)

    def argmax(self):
        index = self.values.argmax()
        return self.states[index]


# #### Step 3: 
# _Create Class Function for Probability Matrix_ 

# In[4]:


class ProbabilityMatrix:
    def __init__(self, prob_vec_dict: dict):
        
        assert len(prob_vec_dict) > 1,             "The numebr of input probability vector must be greater than one."
        assert len(set([str(x.states) for x in prob_vec_dict.values()])) == 1,             "All internal states of all the vectors must be indentical."
        assert len(prob_vec_dict.keys()) == len(set(prob_vec_dict.keys())),             "All observables must be unique."

        self.states      = sorted(prob_vec_dict)
        self.observables = prob_vec_dict[self.states[0]].states
        self.values      = np.stack([prob_vec_dict[x].values                            for x in self.states]).squeeze() 

    @classmethod
    def initialize(cls, states: list, observables: list):
        size = len(states)
        rand = np.random.rand(size, len(observables))              / (size**2) + 1 / size
        rand /= rand.sum(axis=1).reshape(-1, 1)
        aggr = [dict(zip(observables, rand[i, :])) for i in range(len(states))]
        pvec = [ProbabilityVector(x) for x in aggr]
        return cls(dict(zip(states, pvec)))

    @classmethod
    def from_numpy(cls, array: 
                  np.ndarray, 
                  states: list, 
                  observables: list):
        p_vecs = [ProbabilityVector(dict(zip(observables, x)))                   for x in array]
        return cls(dict(zip(states, p_vecs)))

    @property
    def dict(self):
        return self.df.to_dict()

    @property
    def df(self):
        return pd.DataFrame(self.values, 
               columns=self.observables, index=self.states)

    def __repr__(self):
        return "PM {} states: {} -> obs: {}.".format(
            self.values.shape, self.states, self.observables)

    def __getitem__(self, observable: str) -> np.ndarray:
        if observable not in self.observables:
            raise ValueError("Requesting unknown probability observable from the matrix.")
        index = self.observables.index(observable)
        return self.values[:, index].reshape(-1, 1)


# #### Step 4: 
# _Create Class Function for hidden Markov Chain_ 

# In[5]:


from itertools import product
from functools import reduce


class HiddenMarkovChain:
    def __init__(self, T, E, pi):
        self.T = T  # transmission matrix A
        self.E = E  # emission matrix B
        self.pi = pi
        self.states = pi.states
        self.observables = E.observables
    
    def __repr__(self):
        return "HML states: {} -> observables: {}.".format(
            len(self.states), len(self.observables))
    
    @classmethod
    def initialize(cls, states: list, observables: list):
        T = ProbabilityMatrix.initialize(states, states)
        E = ProbabilityMatrix.initialize(states, observables)
        pi = ProbabilityVector.initialize(states)
        return cls(T, E, pi)
    
    def _create_all_chains(self, chain_length):
        return list(product(*(self.states,) * chain_length))
    
    def score(self, observations: list) -> float:
        def mul(x, y): return x * y
        
        score = 0
        all_chains = self._create_all_chains(len(observations))
        for idx, chain in enumerate(all_chains):
            expanded_chain = list(zip(chain, [self.T.states[0]] + list(chain)))
            expanded_obser = list(zip(observations, chain))
            
            p_observations = list(map(lambda x: self.E.df.loc[x[1], x[0]], expanded_obser))
            p_hidden_state = list(map(lambda x: self.T.df.loc[x[1], x[0]], expanded_chain))
            p_hidden_state[0] = self.pi[chain[0]]
            
            score += reduce(mul, p_observations) * reduce(mul, p_hidden_state)
        return score


# #### Step 5: 
# _Create Class Function for Hidden Markov Chain with Forward Pass_ 

# In[6]:


##Scoring With Forward Pass
class HiddenMarkovChain_FP(HiddenMarkovChain):
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) 
                         @ self.T.values) * self.E[observations[t]].T
        return alphas
    
    def score(self, observations: list) -> float:
        alphas = self._alphas(observations)
        return float(alphas[-1].sum())


# #### Step 6: 
# _Create Class Function for Hidden Markov Chain Simulation_ 

# In[7]:


class HiddenMarkovChain_Simulation(HiddenMarkovChain):
    def run(self, length: int) -> (list, list):
        assert length >= 0, "The chain needs to be a non-negative number."
        s_history = [0] * (length + 1)
        o_history = [0] * (length + 1)
        
        prb = self.pi.values
        obs = prb @ self.E.values
        s_history[0] = np.random.choice(self.states, p=prb.flatten())
        o_history[0] = np.random.choice(self.observables, p=obs.flatten())
        
        for t in range(1, length + 1):
            prb = prb @ self.T.values
            obs = prb @ self.E.values
            s_history[t] = np.random.choice(self.states, p=prb.flatten())
            o_history[t] = np.random.choice(self.observables, p=obs.flatten())
        
        return o_history, s_history


# ### Implementation of an Unhealthy Individual 
# _A variables represent transition state probabilities and B variables represent emmission state probabilities_

# In[40]:


#dont change - fixed activity 
a_m = ProbabilityVector({'m': 0.5, 'h': 0.5, 's': 0.0, 'p': 0.0}) 
a_h = ProbabilityVector({'m': 0.0, 'h': 0.34, 's': 0.33, 'p': 0.33}) 
a_s = ProbabilityVector({'m': 0.0, 'h': 0.3, 's': 0.6, 'p': 0.1}) 
a_p = ProbabilityVector({'m': 0.0, 'h': 0.0, 's': 0.0, 'p': 1.0}) 

###

#a --> M H S P 
#b --> enzymes 

A  = ProbabilityMatrix({'m': a_m, 'h': a_h, 's': a_s, 'p': a_p}) 

#changes score variable
#LPAC - LPL / APOC2 / PLB1 / CES5A
#BA - Bile Acids

b_m = ProbabilityVector({'LPAC': 0.6, 'BA': 0.2, 'CL': 0.1, 'AR': 0.1, 'CB': 0.0})
b_h = ProbabilityVector({'LPAC': 0.2, 'BA': 0.2, 'CL': 0.2, 'AR': 0.2, 'CB': 0.2})
b_s = ProbabilityVector({'LPAC': 0.0, 'BA': 0.2, 'CL': 0.3, 'AR': 0.3, 'CB': 0.2})
b_p = ProbabilityVector({'LPAC': 0.0, 'BA': 0.1, 'CL': 0.2, 'AR': 0.1, 'CB': 0.6})
#0.000479

B =  ProbabilityMatrix({'m': b_m, 'h': b_h, 's': b_s, 'p': b_p})


pi = ProbabilityVector({'m': 0.2, 'h': 0.2, 's': 0.4,'p': 0.2})

hmc = HiddenMarkovChain(A, B, pi)

all_possible_scores = 1.0


# In[41]:


##print
print(a_m.df)
print(a_h.df)
print(a_s.df)
print(a_p.df)

print(A)
print(B)


# In[42]:


#data collection
print("Comparison:", a_m == a_h)
print("Element-wise multiplication:", a_m * a_h)
print("Argmax:", a_m.argmax())
print("Getitem:", a_m['m'])

observations = ['LPAC', 'BA', 'CL', 'AR', 'CB'] 

print("Score for {} is {:f}.".format(observations, hmc.score(observations)))

hmc_fp = HiddenMarkovChain_FP(A, B, pi)

observations = ['LPAC','BA', 'CL', 'AR', 'CB']
print("Score for {} is {:f}.".format(observations, hmc_fp.score(observations)))


# In[43]:


hmc_s = HiddenMarkovChain_Simulation(A, B, pi)
observation_hist, states_hist = hmc_s.run(100)  # length = 100

stats = pd.DataFrame({'observations': observation_hist, 'states': states_hist})
stats_matrix = np.matrix(stats)


from collections import Counter
a = observation_hist

plt.hist(a)

plt.title("Unhealthy Markov Occurences of Enzymatic Behaviors")
plt.xlabel("Enzymes")
plt.ylabel("Occurences")
plt.grid()


# In[44]:


b = states_hist

plt.hist(b)

plt.title("Unhealthy Markov Occurences of Tissue Behaviors")
plt.xlabel("Tissues")
plt.ylabel("Occurences")
plt.grid()


# ### Implementation of a Healthy Individual 
# _A variables represent transition state probabilities and B variables represent emmission state probabilities_

# In[45]:


c_m = ProbabilityVector({'m': 0.5, 'h': 0.5, 's': 0.0, 'p': 0.0}) 
c_h = ProbabilityVector({'m': 0.0, 'h': 0.34, 's': 0.33, 'p': 0.33}) 
c_s = ProbabilityVector({'m': 0.0, 'h': 0.3, 's': 0.6, 'p': 0.1}) 
c_p = ProbabilityVector({'m': 0.0, 'h': 0.0, 's': 0.0, 'p': 1.0}) 

#c --> M H S P 
#d --> enzymes 

C  = ProbabilityMatrix({'m': a_m, 'h': a_h, 's': a_s, 'p': a_p}) 

d_m = ProbabilityVector({'LPAC': 0.6, 'BA': 0.2, 'CL': 0.1, 'AR': 0.1, 'CB': 0.0})
d_h = ProbabilityVector({'LPAC': 0.2, 'BA': 0.2, 'CL': 0.2, 'AR': 0.2, 'CB': 0.2})
d_s = ProbabilityVector({'LPAC': 0.0, 'BA': 0.2, 'CL': 0.3, 'AR': 0.3, 'CB': 0.2})
d_p = ProbabilityVector({'LPAC': 0.0, 'BA': 0.1, 'CL': 0.2, 'AR': 0.1, 'CB': 0.6})
#0.000794

D =  ProbabilityMatrix({'m': b_m, 'h': b_h, 's': b_s, 'p': b_p})


pi = ProbabilityVector({'m': 0.25, 'h': 0.25, 's': 0.25,'p': 0.25})

hmc = HiddenMarkovChain(C, D, pi)

all_possible_scores = 1.0


# In[46]:


##print
print(c_m.df)
print(c_h.df)
print(c_s.df)
print(c_p.df)

print(C)
print(D)


# In[47]:


#data collection
print("Comparison:", c_m == c_h)
print("Element-wise multiplication:", c_m * c_h)
print("Argmax:", c_m.argmax())
print("Getitem:", c_m['m'])

observations = ['LPAC', 'BA', 'CL', 'AR', 'CB'] 

print("Score for {} is {:f}.".format(observations, hmc.score(observations)))

hmc_fp = HiddenMarkovChain_FP(C, D, pi)

observations = ['LPAC','BA', 'CL', 'AR', 'CB']
print("Score for {} is {:f}.".format(observations, hmc_fp.score(observations)))


# In[48]:


hmc_s = HiddenMarkovChain_Simulation(C, D, pi)
observation_hist, states_hist = hmc_s.run(100)  # length = 100

stats = pd.DataFrame({'observations': observation_hist, 'states': states_hist})
stats_matrix = np.matrix(stats)


from collections import Counter
c = observation_hist

plt.hist(c)

plt.title("Healthy Markov Occurences of Enzymatic Behaviors")
plt.xlabel("Enzymes")
plt.ylabel("Occurences")
plt.grid()


# In[49]:


d = states_hist

plt.hist(d)

plt.title("Healthy Markov Occurences of Tissue Behaviors")
plt.xlabel("Tissues")
plt.ylabel("Occurences")
plt.grid()


# In[50]:


#Eigen Numbers
Tissues = np.matrix([[0.5,0.5,0,0],
                     [0,0.34,0.33,0.33],
                     [0.0,0.3,0.6,0.1],
                     [0.0,0.0,0.0,1.0]])

Enzymes = np.matrix([[0.6, 0.2, 0.1, 0.1, 0.0], 
                     [0.2, 0.2, 0.2, 0.2, 0.2], 
                     [0.0, 0.2,0.3, 0.3, 0.2],
                     [0.0, 0.1, 0.2, 0.1, 0.6]])


# In[51]:


eigen_vals, eigen_vectors = np.linalg.eig(Tissues)
print("eigen values:", eigen_vals)
print("eigen vectors", eigen_vectors)


# In[ ]:




