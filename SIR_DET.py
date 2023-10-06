import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

N = 1000 # size of population

I = 5 # number of infected individuals (at t0)
S = N-I # number of susceptible individuals (at t0)
R = 0 # number of recovered individuals

alpha = 0 
beta = 0.3 # proportion of susceptible individuals exposed to infection per unit of time
gamma = 1/7 # proportion of sick individuals recovering per unit of time

def ode(t, y):
    S,I,R = y
    dSdt = (-1 * beta) * (I/N) * S
    dIdt = beta * (I/N) * S - (gamma * I)
    dRdt = gamma * I 
    return np.array([dSdt, dIdt, dRdt])
    

h = 501
t_start = 0
t_end = 120
tspan = [t_start, t_end]
tt = np.linspace(t_start, t_end, h)

y0 = np.array([S,I,R])

solve = solve_ivp(fun = ode, t_span = tspan, y0 = y0, t_eval=tt)

plt.plot(solve.t, solve.y[0], label='Susceptible (S)')
plt.plot(solve.t, solve.y[1], label='Infected (I)')
plt.plot(solve.t, solve.y[2], label='Recovered (R)')
plt.legend()
plt.title('SIR-model, deterministic')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')

plt.show()
