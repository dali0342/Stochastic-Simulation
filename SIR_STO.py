import matplotlib.pyplot as plt
import numpy as np
import gillespie
from scipy.integrate import solve_ivp


def ode(t, y):
    S,I,R = y
    dSdt = (-1 * beta) * (I/N) * S
    dIdt = beta * (I/N) * S - (gamma * I)
    dRdt = gamma * I 
    return np.array([dSdt, dIdt, dRdt])
    
def sto():
    return np.array([[-1, 1, 0], [0, -1, 1]])

def pro(values, coeff):
    S, I, R = values
    beta, gamma = coeff
    propensity_infected = beta * S * (I/N)
    propensity_recovered = gamma * I
    
    return [propensity_infected, propensity_recovered]

N = 1000
S0 = 995
I0 = 5
R0 = 0
beta = 0.3
gamma = 1/7
h = 501
t0 = 0
t120 = 120
tspan = [t0, t120]
tt = np.linspace(t0, t120, h)

# DETERMINISTIC SOLOUTION FOR REFRENCE
y0 = np.array([S0,I0,R0])
solve = solve_ivp(fun = ode, t_span = tspan, y0 = y0, t_eval=tt)
plt.plot(solve.t, solve.y[0])
plt.plot(solve.t, solve.y[1])
plt.plot(solve.t, solve.y[2])

# 15 Stochastic soloution 
i = 0
while (i < 15):
    tvec, Xarr = gillespie.SSA(pro, sto, [S0, I0, R0], tspan, [beta, gamma])

    plt.plot(tvec, Xarr[:, 0])
    plt.plot(tvec, Xarr[:, 1])
    plt.plot(tvec, Xarr[:, 2])
    i = 1 + i

plt.show()