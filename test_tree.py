import numpy as np

# Parameters
S0 = 10000  # Initial stock price
K = 10000   # Strike price
r = 0.045   # Risk-free interest rate
sigma = 0.16  # Volatility
T = 8       # Maturity time in years

# Function to calculate GMAB price using binomial tree
def binomial_tree_GMAB(steps_per_year):
    delta = 1 / steps_per_year  # Time step
    u = np.exp(( r - (sigma**2)/2 ) * delta + sigma*np.sqrt(delta))  # Up factor
    d = np.exp(( r - (sigma**2)/2 ) * delta - sigma*np.sqrt(delta))  # Up factor
    p = (np.exp(r * delta) - d) / (u - d)  # Probability of up movement

    # Initialize stock price tree and GMAB value tree
    stock_prices = np.zeros((T * steps_per_year + 1, T * steps_per_year + 1))
    GMAB_values = np.zeros((T * steps_per_year + 1, T * steps_per_year + 1))

    # Construct stock price tree
    for j in range(T * steps_per_year + 1):
        for i in range(j + 1):
            stock_prices[i, j] = S0 * (u ** (j - i)) * (d ** i)

    # Calculate GMAB values at maturity
    for i in range(T * steps_per_year + 1):
        GMAB_values[i, T * steps_per_year] = max(stock_prices[i, T * steps_per_year] - K, 0)

    # Backward induction to calculate GMAB values at earlier times
    for j in range(T * steps_per_year - 1, -1, -1):
        for i in range(j + 1):
            GMAB_values[i, j] = np.exp(-r * delta) * (p * GMAB_values[i, j + 1] + (1 - p) * GMAB_values[i + 1, j + 1])

    return GMAB_values[0, 0]

# Function to calculate GMAB price using analytical formula
def analytical_GMAB():
    TPx = np.exp(-0.002 * T - 0.00025 * T**2)  # Survival probability
    if S0 * np.exp(0.07 * T) > 10000 * np.exp(0.07 * T):
        return TPx * (np.exp(-r * T) * K + np.exp(-r * T) * (np.exp(0.07 * T) * S0 - K) + abs(np.exp(-r * T) * (S0 - K)))
    else:
        return TPx * (S0 + np.exp(-r * T) * (K - np.exp(0.02 * T) * S0) + abs(np.exp(-r * T) * (K - np.exp(0.02 * T) * S0)))

# Calculate GMAB prices using binomial tree and analytical formula for different numbers of steps per year
steps_per_year_values = [10, 50, 100, 250]
for steps_per_year in steps_per_year_values:
    GMAB_binomial = binomial_tree_GMAB(steps_per_year)
    GMAB_analytical = analytical_GMAB()
    print("Steps per year:", steps_per_year)
    print("GMAB price (binomial tree):", GMAB_binomial)
    print("GMAB price (analytical formula):", GMAB_analytical)
    print("-------------------------------------------------")
