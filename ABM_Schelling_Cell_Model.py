#!/usr/bin/env python
# coding: utf-8

# ## Rice Kitchen Algorithm 
# #### Part 2: Agent Based Modeling 
# _The Schelling Model application, Streamlit, was a new and extremely useful tool in animating Python programs. The Schelling Model as a whole was adapted for Rice Kitchen to demonstrate establishment and conditions of functional nutrition. These connections to biology may prove to be important for micro- and molecular biology. In terms of hardware, Streamlit can only be executed in the terminal which also presented some learning opportunities in both MobaXterm and Jupiter platforms involving directory changes, package updates and compatibility. Overall, the agent based facet of the model highlighted the need for nutritional regimens as healthy adjacent cells impact specialized tissues._

# #### Step 1:
# _Import Packages_

# In[2]:


## Imports
import pandas as pd
import numpy as np
import csv
import sympy as sym
import matplotlib.pyplot as plt
import math as m
import random
import streamlit as st


# #### Step 2: 
# _Create Class Function for Schelling Model of Segregation using Cellular Inputs_ 

# In[19]:


class Schelling:

    def __init__(self, size, empty_ratio, similarity_threshold, n_adjacent):
        self.size = size 
        self.empty_ratio = empty_ratio
        self.similarity_threshold = similarity_threshold
        self.n_adjacent = n_adjacent

        # Ratio of races (-1, 1) and empty cells (0)
        p = [(1-empty_ratio)/2, (1-empty_ratio)/2, empty_ratio]
        tissue_size = int(np.sqrt(self.size))**2
        self.tissue = np.random.choice([-1, 1, 0], size=tissue_size, p=p)
        self.tissue = np.reshape(self.tissue, (int(np.sqrt(tissue_size)), int(np.sqrt(tissue_size))))

    def run(self):
            for (row, col), value in np.ndenumerate(self.tissue):
                differential = self.tissue[row, col]
                if differential != 0:
                    adjacent = self.tissue[row-self.n_adjacent:row+self.n_adjacent, 
                                             col-self.n_adjacent:col+self.n_adjacent]
                    adjacent_size = np.size(adjacent)
                    n_empty_cells = len(np.where(adjacent == 0)[0])
                    if adjacent_size != n_empty_cells + 1:
                        n_similar = len(np.where(adjacent == differential)[0]) - 1
                        similarity_ratio = n_similar / (adjacent_size - n_empty_cells - 1.)
                        is_unhealthy = (similarity_ratio < self.similarity_threshold)
                        if is_unhealthy:
                            empty_cells = list(zip(np.where(self.tissue == 0)[0], np.where(self.tissue == 0)[1]))
                            random_cells = random.choice(empty_cells)
                            self.tissue[random_cells] = differential
                            self.tissue[row,col] = 0
    
    def get_mean_similarity_ratio(self):
        count = 0
        similarity_ratio = 0
        for (row, col), value in np.ndenumerate(self.tissue):
            differential = self.tissue[row, col]
            if differential != 0:
                adjacent = self.tissue[row-self.n_adjacent:row+self.n_adjacent, 
                                         col-self.n_adjacent:col+self.n_adjacent]
                adjacent_size = np.size(adjacent)
                n_empty_cells = len(np.where(adjacent == 0)[0])
                if adjacent_size != n_empty_cells + 1:
                    n_similar = len(np.where(adjacent == differential)[0]) - 1
                    similarity_ratio += n_similar / (adjacent_size - n_empty_cells - 1.)
                    count += 1
            return similarity_ratio / count


# In[20]:


st.title("Schelling's Model of Cell Health")


# #### Step 3: 
# _Create Inputs for Variation_ 

# In[21]:


population_size = st.sidebar.slider("Population Size", 500, 10000, 2500)
empty_ratio = st.sidebar.slider("Empty Cells Ratio", 0., 1., .2)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0., 1., .4)
n_iterations = st.sidebar.number_input("Number of Iterations", 50)

schelling = Schelling(population_size, empty_ratio, similarity_threshold, 3)
mean_similarity_ratio = []
mean_similarity_ratio.append(schelling.get_mean_similarity_ratio())


# In[22]:


schelling = Schelling(population_size, empty_ratio, similarity_threshold, 3)
mean_similarity_ratio = []
mean_similarity_ratio.append(schelling.get_mean_similarity_ratio())


# #### Step 4: 
# _Plot Animation_ 

# In[1]:


from matplotlib import pyplot, colors
plt.style.use("ggplot")
plt.figure(figsize=(8, 4))

# Left hand side graph with Schelling simulation plot
cmap = colors.ListedColormap(['red', 'white', 'green'])
plt.subplot(121)
plt.axis('off')
plt.pcolor(schelling.tissue, cmap=cmap, edgecolors='w', linewidths=1)

# Right hand side graph with Mean Similarity Ratio graph
plt.subplot(122)
plt.xlabel("Iterations")
plt.xlim([0, n_iterations])
plt.ylim([0.4, 1])
plt.title("Mean Similarity Ratio", fontsize=15)
plt.text(1, 0.95, "Similarity Ratio: %.4f" % schelling.get_mean_similarity_ratio(), fontsize=10)

city_plot = st.pyplot(plt)


# In[24]:


progress_bar = st.progress(0)


# In[25]:


for i in range(n_iterations):
    schelling.run()
    mean_similarity_ratio.append(schelling.get_mean_similarity_ratio())
    plt.figure(figsize=(8, 4))
    
    plt.subplot(121)
    plt.axis('off')
    plt.pcolor(schelling.tissue, cmap=cmap, edgecolors='w', linewidths=1)
    
    plt.subplot(122)
    plt.xlabel("Iterations")
    plt.xlim([0, n_iterations])
    plt.ylim([0.4, 1])
    
    plt.title("Mean Similarity Ratio", fontsize=15)    
    plt.plot(range(1, len(mean_similarity_ratio)+1), mean_similarity_ratio)    
    plt.text(1, 0.95, "Similarity Ratio: %.4f" % schelling.get_mean_similarity_ratio(), fontsize=10)    
        
    city_plot.pyplot(plt)
    plt.close("all")
    progress_bar.progress((i+1.)/n_iterations)        


# #### Step 5: 
# _Executing Code_ 
# 
# **In terminal**
# 
# Streamlit run [filename]
# 
# _may need to pip install streamlit_
