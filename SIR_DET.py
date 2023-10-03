'''
S′(t) = −β I(t)
N S(t)
I′(t) = β I(t)
N S(t) − γI(t)
R′(t) = γI(t)

N = Population
β andelen mottagliga som blir exponerade för smitta per tidsenhet
γ andelen sjuka som tillfrisknar per tidsenhet

S(t) + I(t) + R(t) = N 

1000 = N
I = 5
beta = 0,3
gamma = 1/7
t0-t120 = tspan

'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def ode(t, y):
    S,I,R = y
    dSdt = (-1 * beta) * (I/N) * S
    dIdt = beta * (I/N) * S - (gamma * I)
    dRdt = gamma * I 
    return np.array([dSdt, dIdt, dRdt])
    

N = 1000
S = 995
I = 5
R = 0
alpha = 0 
beta = 0.3
gamma = 1/7
h = 501
t0 = 0
t120 = 120
tspan = [t0, t120]
tt = np.linspace(t0, t120, h)

y0 = np.array([S,I,R])

solve = solve_ivp(fun = ode, t_span = tspan, y0 = y0, t_eval=tt)

plt.plot(solve.t, solve.y[0])
plt.plot(solve.t, solve.y[1])
plt.plot(solve.t, solve.y[2])
plt.title('SIR - Model')

plt.show()
