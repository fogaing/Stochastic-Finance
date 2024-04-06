import numpy as np
from math import comb
from matplotlib import pyplot as plt

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

