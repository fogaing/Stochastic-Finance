import numpy as np
from matplotlib import pyplot as plt

# parameters
S0 = 10000
T = 8
r = 0.045  
sigma = 0.16

def main():

    max_steps = 100
    #plot_GMAB_bt(max_steps)
    plot_GMDB_bt(max_steps)


def plot_GMDB_bt(max_steps):

    """Makes plot for question 4"""

    steps_per_year = np.arange(10, max_steps, 10)
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

    plt.plot(steps_per_year, GMDB_prices, label="Price using binomial tree")

    real_price =   9837.5 # obtained from running Code.py
    plt.plot(np.arange(max_steps), [real_price for i in range(max_steps)], linestyle='dashed', label="Price using analytical formula")
    plt.xlabel("steps per year", fontsize=12)
    plt.ylabel("GMDB price", fontsize=12)
    plt.legend()
    plt.savefig("GMDB_bt.pdf")
    plt.show()

def plot_GMAB_bt(max_steps):

    """Makes plot for question 2"""

    steps_per_year = np.arange(10, max_steps, 10)
    GMAB_prices = []
    for steps_per_year_i in steps_per_year :

        total_steps = T*steps_per_year_i
        dt= 1/steps_per_year_i

        u = np.exp(( r - (sigma**2)/2 ) * dt + sigma*np.sqrt(dt))  # Up factor
        d = np.exp(( r - (sigma**2)/2 ) * dt - sigma*np.sqrt(dt))  # down factor
        p = (np.exp(r * dt) - d) / (u - d)  

        GMAB = GMAB_bt(u, d, p, total_steps, dt)
        print("steps = ", steps_per_year_i, " GMAB value = ", GMAB)
        GMAB_prices.append(GMAB)

    plt.plot(steps_per_year, GMAB_prices, label="Price using binomial tree")

    real_price =   9497.51 # obtained from running Code.py
    plt.plot(np.arange(max_steps), [real_price for i in range(max_steps)], linestyle='dashed', label="Price using analytical formula")
    plt.xlabel("steps per year", fontsize=12)
    plt.ylabel("GMAB price", fontsize=12)
    plt.legend()
    plt.savefig("GMAB_bt.pdf")
    plt.show()


def tpx(t): 

    """Survival probability up to time t"""

    return np.exp(-0.002*t - 0.00025*t**2)

def P(T, S_T):

    """Payoff function in case of survival"""

    if S_T < S0*np.e**(0.02*T) :
        return S0*np.e**(0.02*T)

    elif S_T >  S0*np.e**(0.07*T) :
        return S0*np.e**(0.07*T)
    else :
        return S_T

def GMDB_bt(u, d, p, total_steps, steps_per_year, dt):

    """Computes GMDB value"""

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
    
    """Generates tree of GMDB value in case of death"""

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

    """Computes GMAB value"""

    # stock index terminal values
    S_T =  S0 * np.array([u **i * (d ** (total_steps -i)) for i in range(total_steps+1)])

    # terminal payoff values
    P_T = [P(T, S_Ti) for S_Ti in S_T]

    # applies backward procedure
    P_t_next = P_T
    P_t_current = []

    for i in range(total_steps):
        
        P_t_current = np.zeros(total_steps-i)

        # current time
        t = T-(i+1)*dt
        # probability of death during the interval
        p_death = (tpx(t) - tpx(t+dt))/tpx(t)

        for j in range(total_steps-i):
            
            # one step update
            P_t_current[j] = (1-p_death)*np.e**(-r*dt)*(p*P_t_next[j+1] + (1-p)*P_t_next[j])

        P_t_next = P_t_current
    
    return P_t_current[0]

if __name__ == "__main__":
    
    main()

