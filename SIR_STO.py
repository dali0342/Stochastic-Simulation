import matplotlib.pyplot as plt
import numpy as np
import gillespie

def sto():
    return np.array([[-1, 1, 0], [0, -1, 1]])

def pro(values, coeff):
    S, I, R = values
    beta, gamma = coeff
    propensity_infected = beta * S 
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


tvec, Xarr = gillespie.SSA(pro, sto, [S0, I0, R0], tspan, [beta, gamma])

plt.plot(tvec, Xarr[:, 0])
plt.plot(tvec, Xarr[:, 1])
plt.plot(tvec, Xarr[:, 2])

plt.show()