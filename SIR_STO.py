import matplotlib.pyplot as plt
import numpy as np
import gillespie

N = 1000 # size of population

I = 5 # number of infected individuals (at t0)
S = N-I # number of susceptible individuals (at t0)
R = 0 # number of recovered individuals

beta = 0.3 # proportion of susceptible individuals exposed to infection per unit of time
gamma = 1/7 # proportion of sick individuals recovering per unit of time

# stoichiometric matrix
def sto():
    return np.array([
        [ -1,  1,  0], 
        [  0, -1,  1]
        ])

# propensity function
def pro(values, coeff):
    S, I, R = values
    beta, gamma = coeff
    propensity_infected = beta * S * (I/N)
    propensity_recovered = gamma * I
    
    return np.array([
        propensity_infected, 
        propensity_recovered
        ])

t_start = 0
t_end = 120
tspan = [t_start, t_end]

tvec, Xarr = gillespie.SSA(pro, sto, [S, I, R], tspan, [beta, gamma])

plt.plot(tvec, Xarr[:, 0])
plt.plot(tvec, Xarr[:, 1])
plt.plot(tvec, Xarr[:, 2])

num_simulations = 15

for i in range(num_simulations):
    tvec, Xarr = gillespie.SSA(pro, sto, [S, I, R], tspan, [beta, gamma])

    plt.plot(tvec, Xarr[:, 0], label='S: Susceptible', color='tab:blue')
    plt.plot(tvec, Xarr[:, 1], label='I: Infected', color='tab:orange')
    plt.plot(tvec, Xarr[:, 2], label='R: Recovered', color='tab:green')
    i = 1 + i

# Only display 3 labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.xlabel('Time')
plt.ylabel('Number of Individuals')

plt.title('SIR-model, stochastic')

plt.show()