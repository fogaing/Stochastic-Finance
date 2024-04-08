import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
S0 = 10000
r = 0.045  
sigma = 0.16
maturities = np.array([i for i in range(1, 21)])  # Time range

# Survival probability
def tpx(t): return np.exp(-0.002*t - 0.00025*t**2)

def K_1(T) : return 10000*np.exp(0.02*T)
def K_2(T) : return 10000*np.exp(0.07*T)

# Black & Scholes call option
def BS_call(t, S_t, T, K, r, sigma):

    d_1 = (np.log(K/S_t) - (r + (sigma**2)/2)*(T-t))/(sigma*np.sqrt(T-t)) 
    d_2 = (np.log(K/S_t) - (r - (sigma**2)/2)*(T-t))/(sigma*np.sqrt(T-t)) 

    return S_t*np.e**(r*(T-t))*norm.cdf(-d_1) - K*norm.cdf(-d_2)

# Black & Scholes put option
def BS_put(t, S_t, T, K, r, sigma):

    d_1 = (np.log(K/S_t) - (r + (sigma**2)/2)*(T-t))/(sigma*np.sqrt(T-t)) 
    d_2 = (np.log(K/S_t) - (r - (sigma**2)/2)*(T-t))/(sigma*np.sqrt(T-t)) 

    return  K*norm.cdf(d_2) - S_t*np.e**(r*(T-t))*norm.cdf(d_1) 

# GMAB using call options
def GMAB_call(t, S_t, T):

    return np.e**(-r*(T-t))*(tpx(T)/tpx(t))*(K_1(T) + BS_call(t, S_t, T, K_1(T), r, sigma) - BS_call(t, S_t, T, K_2(T), r, sigma))

# GMAB using put options
def GMAB_put(t, S_t, T):

    return np.e**(-r*(T-t))*(tpx(T)/tpx(t))*(K_2(T) - BS_put(t, S_t, T, K_2(T), r, sigma) + BS_put(t, S_t, T, K_1(T), r, sigma))

def GMDB_call(t, S_t, T):

    result = GMAB_call(t, S_t, T)
    for k in range(int(t)+1, T+1):
        survival_k= (tpx(max(t, k-1)) - tpx(k))/tpx(t)
        result += survival_k*(S0*np.e**(-r*(k-t) + 0.02*k) + np.e**(-r*(k-t))*BS_call(t, S_t, k, S0*np.e**(0.02*k), r, sigma))
    
    return result

def GMDB_put(t, S_t, T):
    
    result = GMAB_put(t, S_t, T)
    for k in range(int(t)+1, T+1):
        survival_k= (tpx(max(t, k-1)) - tpx(k))/tpx(t)
        result += survival_k*(S_t+ np.e**(-r*(k-t))*BS_put(t, S_t, k, S0*np.e**(0.02*k), r, sigma))
    
    return result

GMABcall = np.array( [GMAB_call(0, S0, T) for T in maturities] )
GMABput = np.array( [GMAB_put(0, S0, T) for T in maturities] )

GMDBcall = np.array( [GMDB_call(0, S0, T) for T in maturities] )
GMDBput = np.array( [GMDB_put(0, S0, T) for T in maturities] )

# print("Maturity = 8, GMAB = ", GMABcall[7])
# print("Maturity = 8, GMDB = ", GMDBcall[7])

plt.plot(maturities, GMABcall, label='GMAB using call options')
plt.plot(maturities, GMABput, label='GMAB using put options')

plt.plot(maturities, GMDBcall, label='GMDB using call options')
plt.plot(maturities, GMDBput, label='GMDB using put options')

plt.xlabel('Maturity (years)')
plt.xticks(maturities)
plt.ylabel('GMAB Value')
plt.title('GMAB at time t=0 with maturities from 1 to 20 years')
plt.legend()
plt.grid(True)

plt.savefig("GMAB.pdf")
plt.show()