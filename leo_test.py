import numpy as np

# Parameters
S0 = 10000
T = 8
K_1 = 10000*np.exp(0.02*T)
K_2 = 10000*np.exp(0.07*T)
r = 0.045  
sigma = 0.16
t = np.linspace(0, T, 100)  # Time range

def simulate_S_t(T):
    z = np.random.normal(0, 1)
    return S0*np.exp( (r - (sigma**2)/2 )*T + (sigma**2)*np.random.normal(0, np.sqrt(T))  )

time = np.array([i for i in range(1,21)])

result = simulate_S_t(time)

print(simulate_S_t(0))