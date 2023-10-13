import matplotlib.pyplot as plt
import numpy as np
import gillespie

N = 1000 # size of population

E = 2     # number of exposed individuals (at t0)
I = 5     # number of infected individuals (at t0)
R = 0     # number of recovered individuals
S = N-E-I # number of susceptible individuals (at t0)

alpha = 1/5.5 # incubation time 
beta = 0.3 # proportion of susceptible individuals exposed to infection per unit of time
gamma = 1/7 # proportion of sick individuals recovering per unit of time

# stoichiometric matrix
def sto():
    return np.array([
        [-1,  1,  0,  0], 
        [ 0, -1,  1,  0], 
        [ 0,  0, -1,  1]
        ])

# propensity function
def pro(values, coeff):
    S, E, I, R = values
    beta, alpha, gamma = coeff
    propensity_exposed = beta * S * (I/N)
    propensity_infected = alpha * E
    propensity_recovered = gamma * I
    
    return np.array([
        propensity_exposed, 
        propensity_infected, 
        propensity_recovered
        ])

t0 = 0
t120 = 120
tspan = [t0, t120]

num_simulations = 15

for i in range(num_simulations):
    tvec, Xarr = gillespie.SSA(pro, sto, [S, E, I, R], tspan, [beta, alpha, gamma])

    plt.plot(tvec, Xarr[:, 0], label='S: Susceptible', color='tab:blue')
    plt.plot(tvec, Xarr[:, 1], label='E: Exposed', color='tab:purple')
    plt.plot(tvec, Xarr[:, 2], label='I: Infected', color='tab:orange')
    plt.plot(tvec, Xarr[:, 3], label='R: Recovered', color='tab:green')
    i = 1 + i

# Only display 4 labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.suptitle('SEIR-model', fontsize=16)
plt.title('Incubation time = {} days'.format(1/alpha))
plt.xlabel('Time')
plt.ylabel('Number of Individuals')

plt.show()
