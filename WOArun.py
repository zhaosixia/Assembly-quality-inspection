"""
NAT - Assignment2
Luca G.McArthur s14422321
Gabriel Hoogervorst s1505156

This script runs the Whale Optimization Algorithm for a given set of parameters.

Code for the oiriginal implementation can be found at http://www.alimirjalili.com/WOA.html 
"""

from WOAfitness import *
from WOAG import WOA
#from WOA_conceptual import WOA_conceptual
#from WOA_mathematical import WOA_mathematical
import matplotlib.pyplot as plt
#import matplotlib.ticker as mtick
import numpy as np
# inits
n_agents = 10
max_iter = 50
n_runs = 1
func_name = ['func']

# compute the scores for each function
scores = []
lower_b, upper_b, dim, bench_f = get_function_details(func_name[0])
for i in range(n_runs):
    # to run an alternative implementation simply call WOA_conceptual/WOA_mathematical
    woa = WOA(n_agents, max_iter, lower_b, upper_b, dim,  bench_f)
    best_score, best_pos, conv_curve, leader_p = woa.forward()
    print('Best solution found by WOA at run {}: {}'.format(i, best_score))
    scores.append(best_score)
 
    print('Function: {} -> ave = {}, std = {}'.format(func_name, np.mean(scores), np.std(scores)))

# plot
fig, ax = plt.subplots()
ax.plot(conv_curve)
ax.ticklabel_format(axis='y', style='sci')
ax.set(xlabel='iter', ylabel='leader_score',
       title='Objective Space')
plt.show() 
