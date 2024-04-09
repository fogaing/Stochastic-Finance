import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import comb

"""----------------------------------------------------- Question 1 : GMAB -------------------------------------------------------"""
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

GMABcall = np.array( [GMAB_call(0, S0, T) for T in maturities] )
GMABput = np.array( [GMAB_put(0, S0, T) for T in maturities] )

print("Maturity = 8, GMAB = ", GMABcall[7])
plt.plot(maturities, GMABcall, label='GMAB using call options')
plt.plot(maturities, GMABput, label='GMAB using put options')
plt.xlabel('Time (years)')
plt.xticks(maturities)
plt.ylabel('GMAB Value')
plt.title('GMAB at time t=0 with maturities from 1 to 20 years')
plt.legend()
plt.grid(True)
plt.savefig("GMAB_price.png")
plt.show()

"""----------------------------------------------------- Question 2 : GMAB -------------------------------------------------------"""
# parameters
S0 = 10000
T = 8
r = 0.045  
sigma = 0.16

def main():

    steps_per_year = [10, 25, 50, 75, 100, 150, 200, 250, 300]
    GMAB_prices = []
    for steps_per_year_i in steps_per_year :

        total_steps = T*steps_per_year_i
        dt= 1/steps_per_year_i

        u = np.exp(( r - (sigma**2)/2 ) * dt + sigma*np.sqrt(dt))  # Up factor
        d = np.exp(( r - (sigma**2)/2 ) * dt - sigma*np.sqrt(dt))  # down factor
        p = (np.exp(r * dt) - d) / (u - d)  

        GMAB_prices.append(GMAB_bt(u, d, p, total_steps, dt))

    plt.plot(steps_per_year, GMAB_prices)

    real_price = 9497.51 # obtained from running Code.py
    plt.plot(np.arange(300), [real_price for i in range(300)], linestyle='dashed', label="B&S price")
    plt.savefig("B&S_price.png")
    plt.legend()
    plt.show()

def tpx(t): return np.exp(-0.002*t - 0.00025*t**2)


def GMAB_bt(u, d, p, total_steps, dt):

    # stock index terminal values
    S_T =  S0 * np.array([u **i * (d ** (total_steps -i)) for i in range(total_steps+1)])

    # payoff function
    def P(T, S_T):
        
        if S_T < S0*np.e**(0.02*T) :
            return S0*np.e**(0.02*T)

        elif S_T >  S0*np.e**(0.07*T) :
            return S0*np.e**(0.07*T)
        else :
            return S_T

    # terminal payoff values
    P_T = [P(T, S_Ti) for S_Ti in S_T]

    # applies backward procedure
    tree_output = backward_bt(P_T, total_steps, dt, p)

    # multiplying by survival probability
    return tpx(T)*tree_output

def backward_bt(P_T, total_steps, dt, p):
    
    # Prices in dt+1
    P_t_next = P_T

    # Prices in t
    P_t_current = []

    for i in range(total_steps):
        
        P_t_current = np.zeros(total_steps-i)
        for j in range(total_steps-i):
            
            # one step update
            P_t_current[j] = np.e**(-r*dt)*(p*P_t_next[j+1] + (1-p)*P_t_next[j])

        P_t_next = P_t_current
    
    return P_t_current[0]


def binomial(n, p, P_T):
    
    result = 0
    for j in range(0, n+1):
        
        result +=  comb(n, j)* p**j * (1-p)**(n-j) * P_T[j]

    return np.e**(-r*T)*result


if __name__ == "__main__":
    
    main()


"""----------------------------------------------------- Question 3 : GMDB -------------------------------------------------------"""

