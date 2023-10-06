import matplotlib.pyplot as plt
import numpy as np
import gillespie

N = 1000 # size of population

E = 2 # number of exposed individuals (at t0)
I = 5 # number of infected individuals (at t0)
R = 0 # number of recovered individuals (at t0)
D = 0 # number of dead individuals (at t0)
S = N-E-I # number of susceptible individuals (at t0)

alpha = 1/5.5 # incubation time 
beta = 0.3 # proportion of susceptible individuals exposed to infection per unit of time
gamma = 1/7 # proportion of sick individuals recovering per unit of time
mu = alpha * gamma # proportion of sick individuals dying per unit of time (mortality rate)

# stoichiometric matrix
def sto():
    return np.array([
        [-1,1,0,0,0], 
        [0,-1,1,0,0], 
        [0,0,-1,1,0],
        [0,0,-1,0,1]
        ])

# propensity function
def pro(values, coeff):
    S, E, I, R, D= values
    beta, alpha, gamma, mu = coeff
    propensity_exposed = beta * S * (I/N)
    propensity_infected = alpha * E
    propensity_recovered = gamma * I
    propensity_dead = mu * I
    
    return np.array([
        propensity_exposed, 
        propensity_infected, 
        propensity_recovered, 
        propensity_dead
        ])

t0 = 0
t120 = 120
tspan = [t0, t120]

tvec, Xarr = gillespie.SSA(pro, sto, [S, E, I, R, D], tspan, [beta, alpha, gamma, mu])

plt.plot(tvec, Xarr[:, 0], label='Susceptible (S)')
plt.plot(tvec, Xarr[:, 1], label='Exposed (E)')
plt.plot(tvec, Xarr[:, 2], label='Infected (I)')
plt.plot(tvec, Xarr[:, 3], label='Recovered (R)')
plt.plot(tvec, Xarr[:, 4], label='Dead (D)')

plt.legend()
plt.title('SIERD-model')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')

plt.show()