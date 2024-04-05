import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
S0 = 10000
#T = 8
#K_1 = 10000*np.exp(0.02*T)
#K_2 = 10000*np.exp(0.07*T)
r = 0.045  
sigma = 0.16
Time = np.array([i for i in range(1,21)])  # Time range

def simulation_S_t(T):
    z = np.random.normal(0, 1)
    return S0*np.exp( (r - (sigma**2)/2 )*T + (sigma**2)*np.random.normal(0, np.sqrt(T))  )

St = np.array([simulation_S_t(T) for T in Time])

# Survival probability
def tpx(t): return np.exp(-0.002*t - 0.00025*t**2)

def K_1(T) : return 10000*np.exp(0.02*T)
def K_2(T) : return 10000*np.exp(0.07*T)

def d1(t,T) :
    return (np.log(K_1(T)/St[t]) - (r + (sigma**2)/2)*(T-t))*( 1/(sigma*np.sqrt(T-t)) )

def d2(t,T) :
    return (np.log(K_1(T)/St[t]) - (r - (sigma**2)/2)*(T-t))*( 1/(sigma*np.sqrt(T-t)) )

def l1(t,T) :
    return (np.log(K_2(T)/St[t]) - (r + (sigma**2)/2)*(T-t))*( 1/(sigma*np.sqrt(T-t)) )

def l2(t,T) :
    return (np.log(K_2(T)/St[t]) - (r - (sigma**2)/2)*(T-t))*( 1/(sigma*np.sqrt(T-t)) )

def Phi(x):
    return norm.cdf(x)

# GMAB using call options
def GMAB_call(t,T):
    d_1 = d1(t,T)
    d_2 = d2(t,T)
    l_1 = l1(t,T)
    l_2 = l2(t,T)
    return ( tpx(T)/tpx(t))*( K_1(T)*Phi(d_2)*np.exp(-r*(T-t)) + St[t]*Phi(-d_1) - St[t]*Phi(-l_1) + K_2(T)*Phi(-l_2)*np.exp(-r*(T-t)) )


# GMAB using put options
def GMAB_put(t,T):
    d_1 = d1(t,T)
    d_2 = d2(t,T)
    l_1 = l1(t,T)
    l_2 = l2(t,T)
    return ( tpx(T)/tpx(t))*( K_2(T)*Phi(-l_2)*np.exp(-r*(T-t)) + St[t]*Phi(l_1) + K_1(T)*Phi(d_2)*np.exp(-r*(T-t)) - St[t]*Phi(d_1) )

GMABcall = np.array( [GMAB_call(0,T) for T in Time] )
GMABput = np.array( [GMAB_put(0,T) for T in Time] )

for i in range(len(GMABcall)):
    print(GMABcall[i]-GMABput[i])
      
"""
plt.plot(Time, GMABcall, label='GMAB using call options')
plt.plot(Time, GMABput, label='GMAB using put options')
plt.xlabel('Time (years)')
plt.ylabel('GMAB Value')
plt.title('GMAB at time t=0 with maturities from 1 to 20 years')
plt.legend()
plt.grid(True)
plt.show()"""