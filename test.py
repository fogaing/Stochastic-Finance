import numpy as np
import copy as cp

def posDiff(x,y):
    if  (x-y)<0:
        return 0
    return x-y

S0 = 10
T = 4
K = 10
r = 0.045  
sigma = 0.16  # Volatility
steps_per_year = 2

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

print(stock_prices)

call_prices = np.zeros_like(stock_prices)

# Backward induction to calculate GMAB values at earlier times

# Calculate call and put prices at maturity
call_prices[:, -1] = np.maximum(stock_prices[:, -1] - K, 0)
#put_prices[:, -1] = np.maximum(K - stock_prices[:, -1], 0)

for j in range(T * steps_per_year - 1, -1, -1):
    for i in range(j + 1):
        # GMAB_values calculation (assuming r and delta are defined)
        
        #stock_prices[i, j] = np.exp(-r * dt) * (p * call_prices[i, j + 1] + (1 - p) * call_prices[i + 1, j + 1])

        call_prices[i, j] = np.exp(-r * delta) * (p * call_prices[i, j + 1] + (1 - p) * call_prices[i + 1, j + 1])


print("*************************")
print("*************************")
print(call_prices)

