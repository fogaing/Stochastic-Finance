import numpy as np

#------------------------------------------------- Question 1 -----------------------------------------------------------

S0 = 10000
T = 8
K_1 = 10000*np.exp(0.02*T)
K_2 = 10000*np.exp(0.07*T)
r = 0.045  
sigma = 0.16

def P_call(S_T):
    return K_1 + np.maximum(S_T - K_1, 0) - np.maximum(S_T - K_2, 0)

def P_put(S_T):
    return K_2 - np.maximum( K_2 - S_T, 0) + np.maximum(K_1 - S_T, 0)


def binomial_tree_GMAB(steps_per_year):
    
    delta= 1/steps_per_year

    u = np.exp(( r - (sigma**2)/2 ) * delta + sigma*np.sqrt(delta))  # Up factor
    d = np.exp(( r - (sigma**2)/2 ) * delta - sigma*np.sqrt(delta))  # Up factor
    p = (np.exp(r * delta) - d) / (u - d)  # Probability of up movement

    # Initialize stock price tree and GMAB value tree
    stock_prices = np.zeros((T * steps_per_year + 1, T * steps_per_year + 1))

    # Construct stock price tree
    for j in range(T * steps_per_year + 1):
        for i in range(j + 1):
            stock_prices[i, j] = S0 * (u ** (j - i)) * (d ** i)



    call_prices = np.zeros_like(stock_prices)
    put_prices = np.zeros_like(stock_prices)

    # Backward induction to calculate GMAB values at earlier times

    # --------------------- Calculate call and put prices at maturity ----------------------
    
    # 1- Call
    """Set the payoff at expiry time T"""
    for i in range(len(stock_prices)) :
        call_prices[i, -1] = P_call(stock_prices[i, -1])
        #put_prices[i, -1] = P_put(stock_prices[i, -1])
        #print((call_prices[i, -1] ,put_prices[i, -1]))

    """ start the backward tree for call option"""
    call_prices[:, -1] = np.maximum( call_prices[:, -1] - K_1 , 0 )

    # 2- Put
    """Set the payoff at expiry time T"""
    for i in range(len(stock_prices)) :
        put_prices[i, -1] = P_put(stock_prices[i, -1])


    """ start the backward tree for call option"""
    put_prices[:, -1] = np.maximum( K_2 - put_prices[:, -1] , 0 )


    for j in range(T * steps_per_year - 1, -1, -1):
        for i in range(j + 1):
            # GMAB_values calculation (assuming r and delta are defined)
            
            #stock_prices[i, j] = np.exp(-r * dt) * (p * call_prices[i, j + 1] + (1 - p) * call_prices[i + 1, j + 1])

            call_prices[i, j] = np.exp(-r * delta) * (p * call_prices[i, j + 1] + (1 - p) * call_prices[i + 1, j + 1])

            put_prices[i, j] = np.exp(-r * delta) * (p * put_prices[i, j + 1] + (1 - p) * put_prices[i + 1, j + 1])
    
    return ( call_prices[0,0], put_prices[0,0] )



steps_per_year_values = [10, 50, 100, 250]
for steps_per_year in steps_per_year_values:
    GMAB_binomial = binomial_tree_GMAB(steps_per_year)
    print("Steps per year:", steps_per_year)
    print("call GMAB price (binomial tree):", GMAB_binomial[0])
    print("put GMAB price (binomial tree):", GMAB_binomial[1])
    print("\n")