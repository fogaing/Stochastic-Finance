import numpy as np
from math import comb
from matplotlib import pyplot as plt

# parameters
S0 = 10000
T = 8
r = 0.045  
sigma = 0.16

def main():

    steps_per_year = [10, 25, 50, 75, 100] #, 150, 200, 250]
    GMDB_prices = []
    for steps_per_year_i in steps_per_year :

        total_steps = T*steps_per_year_i
        dt= 1/steps_per_year_i

        u = np.exp(( r - (sigma**2)/2 ) * dt + sigma*np.sqrt(dt))  # Up factor
        d = np.exp(( r - (sigma**2)/2 ) * dt - sigma*np.sqrt(dt))  # down factor
        p = (np.exp(r * dt) - d) / (u - d)  

        GMDB = GMDB_bt(u, d, p, total_steps, steps_per_year_i, dt)
        print("steps = ", steps_per_year_i, " GMDB value = ", GMDB)
        GMDB_prices.append(GMDB)

    plt.plot(steps_per_year, GMDB_prices)

    real_price = 9837.5 # obtained from running Code.py
    plt.plot(np.arange(300), [real_price for i in range(300)], linestyle='dashed', label="B&S price")
    plt.legend()
    plt.show()

def tpx(t): return np.exp(-0.002*t - 0.00025*t**2)

# payoff function
def P(T, S_T):
    
    if S_T < S0*np.e**(0.02*T) :
        return S0*np.e**(0.02*T)

    elif S_T >  S0*np.e**(0.07*T) :
        return S0*np.e**(0.07*T)
    else :
        return S_T

def GMDB_bt(u, d, p, total_steps, steps_per_year, dt):

    # stock values at integer periods
    S = [S0 * np.array([u **i * (d ** (steps_per_year*k - i)) for i in range(steps_per_year*k+1)]) for k in range(0, T+1)]

    # building the death tree ->
    death_tree = generate_death_tree(S, u, d, p, total_steps, steps_per_year, dt)

    # terminal payoff values
    P_T = [P(T, S_Ti) for S_Ti in S[T]]

    # Prices in dt+1
    P_t_next = P_T

    # Prices in t
    P_t_current = []

    for i in range(total_steps):
        
        P_t_current = np.zeros(total_steps-i)

        # current time
        t = T-(i+1)*dt
        # probability of death during the interval [t, t+dt]
        p_death = (tpx(t) - tpx(t+dt))/tpx(t)
        death_layer = death_tree[total_steps - i]
        for j in range(total_steps-i):
            
            # one step update
            fu = p_death*death_layer[j+1] + (1-p_death)*P_t_next[j+1]
            fd = p_death*death_layer[j] + (1-p_death)*P_t_next[j]

            P_t_current[j] = np.e**(-r*dt)*(p*fu + (1-p)*fd)

        P_t_next = P_t_current
    
    return P_t_current[0]


def generate_death_tree(S, u, d, p, total_steps, steps_per_year, dt):

    # building the death tree
    death_tree = [np.zeros(i) for i in range(1, total_steps+2)]

    # death payoff 
    def P_death(T, S_T):
        
        if S_T < S0*np.e**(0.02*T) :
            return S0*np.e**(0.02*T)
        else :
            return S_T
        
    for k in range(1, T+1):

        death_tree[k*steps_per_year] = [P_death(k, S[k][i]) for i in range(steps_per_year*k+1)]

        # backward procedure for one period
        for i in range(k*steps_per_year - 1, (k-1)*steps_per_year - 1, -1):
            for j in range(i+1):

                death_tree[i][j] = np.e**(-r*dt)*(p*death_tree[i+1][j+1] + (1-p)*death_tree[i+1][j])

    return death_tree

def GMAB_bt(u, d, p, total_steps, dt):

    # stock index terminal values
    S_T =  S0 * np.array([u **i * (d ** (total_steps -i)) for i in range(total_steps+1)])

    # terminal payoff values
    P_T = [P(T, S_Ti) for S_Ti in S_T]

    # applies backward procedure
    GMAB = backward_bt(P_T, total_steps, dt, p)

    return GMAB

def backward_bt(P_T, total_steps, dt, p):
    
    # Prices in dt+1
    P_t_next = P_T

    # Prices in t
    P_t_current = []

    for i in range(total_steps):
        
        P_t_current = np.zeros(total_steps-i)

        # current time
        t = T-(i+1)*dt
        # probability of death during the interval
        p_death = (tpx(t) - tpx(t+dt))/tpx(t)

        for j in range(total_steps-i):
            
            # one step update
            P_t_current[j] = p_death*0 + (1-p_death)*np.e**(-r*dt)*(p*P_t_next[j+1] + (1-p)*P_t_next[j])

        P_t_next = P_t_current
    
    return P_t_current[0]


def binomial(n, p, P_T):
    
    result = 0
    for j in range(0, n+1):
        
        result +=  comb(n, j)* p**j * (1-p)**(n-j) * P_T[j]

    return np.e**(-r*T)*result


if __name__ == "__main__":
    
    main()

