import matplotlib.pyplot as plt
import numpy as np
import gillespie
from scipy.integrate import solve_ivp

N = 1000 # size of population

I = 5 # number of infected individuals (at t0)
S = N-I # number of susceptible individuals (at t0)
R = 0 # number of recovered individuals

alpha = 0 
beta = 0.3 # proportion of susceptible individuals exposed to infection per unit of time
gamma = 1/7 # proportion of sick individuals recovering per unit of time

def ode(t, y): # only used when wanting to plot deterministic
    S,I,R = y
    dSdt = (-1 * beta) * (I/N) * S
    dIdt = beta * (I/N) * S - (gamma * I)
    dRdt = gamma * I 
    return np.array([dSdt, dIdt, dRdt])

# stoichiometric matrix
def sto():
    return np.array([
        [-1, 1, 0], 
        [0, -1, 1]
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

h = 501 # only used when wanting to plot deterministic
tt = np.linspace(t_start, t_end, h) # only used when wanting to plot deterministic


tvec, Xarr = gillespie.SSA(pro, sto, [S, I, R], tspan, [beta, gamma])

plt.plot(tvec, Xarr[:, 0])
plt.plot(tvec, Xarr[:, 1])
plt.plot(tvec, Xarr[:, 2])

'''
# 15 Stochastic soloution 
i = 0
while (i < 15):
    tvec, Xarr = gillespie.SSA(pro, sto, [S, I, R], tspan, [beta, gamma])

    plt.plot(tvec, Xarr[:, 0])
    plt.plot(tvec, Xarr[:, 1])
    plt.plot(tvec, Xarr[:, 2])
    i = 1 + i
'''


'''
# DETERMINISTIC SOLOUTION FOR REFRENCE
y0 = np.array([S,I,R])
solve = solve_ivp(fun = ode, t_span = tspan, y0 = y0, t_eval=tt)
plt.plot(solve.t, solve.y[0])
plt.plot(solve.t, solve.y[1])
plt.plot(solve.t, solve.y[2])
'''
plt.plot(tvec, Xarr[:, 0], label='Susceptible (S)')
plt.plot(tvec, Xarr[:, 1], label='Infected (I)')
plt.plot(tvec, Xarr[:, 2], label='Recovered (R)')

plt.legend()

plt.xlabel('Time')
plt.ylabel('Number of Individuals')

plt.title('SIR-model, stochastic')

plt.show()