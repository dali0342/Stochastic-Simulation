import matplotlib.pyplot as plt
import numpy as np
import gillespie




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
