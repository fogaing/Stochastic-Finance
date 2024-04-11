import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import cholesky
from math import comb


""" ----------------------------------------- Question 1 -----------------------------------------------------------"""
df = pd.read_excel('Data_project.xlsx', header=1)

df['Date'] = pd.to_datetime(df['Date'])

# Define the 'Date' column as an index
df.set_index('Date', inplace=True)

"""
plt.figure(figsize=(10, 6))
plt.plot(df['Close'], label='Eurxx50', color='blue')
plt.plot(df['Close.1'], label='Amex', color='red')
plt.xlabel('Date')
plt.ylabel('Valeur de l\'indice')
plt.title('Indices boursiers')
plt.legend()
plt.grid(True)
#plt.show()"""



indices_columns = ['Close', 'Close.1']


df_returns = df[indices_columns] / df[indices_columns].shift(1)

"""Plot a graph of daily returns
df_returns.plot(figsize=(10, 6))
plt.xlabel('Date')
plt.ylabel('Daily Returns')
plt.title('Daily Returns of Indices')
plt.grid(True)
#plt.show()
"""
#df_returns[indices_columns] = df_returns[indices_columns].fillna(0)


averages = df_returns.mean()

standard_deviations = df_returns.std()

correlation = df_returns.corr()

print("averages",averages)
print("\n standar deviation",standard_deviations)
print("\n correlation",correlation)


days_per_year = 257  # For a typical trading year

annual_standard_deviations = standard_deviations * np.sqrt(days_per_year)

annual_returns = df_returns.mean() * days_per_year

covariance_matrix = df_returns.cov() * days_per_year

print("variance :\n", annual_standard_deviations)
print("\nRendements annuels :\n", annual_returns)
print("\nMatrice de covariance :\n", covariance_matrix)



""" ----------------------------------------- Question 2 -----------------------------------------------------------"""

def V_t(r,t,T,S0,sigma11,sigma21,sigma22):
    S1_t = S0
    S2_t = S0
    mu = -0.5*(sigma22**2 + (sigma21-sigma11)**2)*(T-t)
    
    v = np.sqrt((sigma22**2 + (sigma21 - sigma11)**2)*(T-t))

    d_2 = (np.log(S1_t/S2_t)-mu)/v
    
    d_1 = d_2 - v

    return S1_t + S2_t*norm.cdf(-d_1) - S1_t*norm.cdf(-d_2)

SIGMA = cholesky(covariance_matrix)
SIGMA = SIGMA.T

#Choleski matrix
print("\ncholeski matrix : ")
print(SIGMA)

sigma11 = SIGMA[0][0]
sigma21 = SIGMA[1][0]
sigma22 = SIGMA[1][1]

std_1 = standard_deviations[0]
std_2 = standard_deviations[1]
maturities = np.array([i for i in range(1, 11)])  # Time range

vt = []
S0 = 15000
r=0.0375

for T in maturities :
    vt.append(V_t(r,0,T,S0,sigma11,sigma21,sigma22))

"""plt.plot(maturities,vt)
plt.xlabel('Maturity (T)')
plt.ylabel('Option Value')
plt.xticks(maturities)
plt.title('Maximum Return Insurance Option Value vs. Maturity')
plt.grid(True)
plt.savefig("Return Insurance.png")
plt.show()"""


"--------------------------------------------------Question 3 --------------------------------------------------------"

def simulate_asset_paths(r,S0, chol, T, m, N):
    dt = T / m
    S1 = np.zeros((m+1, N))
    S2 = np.zeros((m+1, N))
    S1[0,:] = S0
    S2[0,:] = S0

    for k in range(N):  # Loop over each simulation
    
        for j in range(1, m+1):
            epsilon = np.random.normal(size=2) # array of size 2 that contains our two epsilon
            dS1 = S1[j-1,k] * ( r*dt + chol @ (np.sqrt(dt)*epsilon )  )[0]
            dS2 = S2[j-1,k] * ( r*dt + chol @ (np.sqrt(dt)*epsilon )  )[1]
            S1[j,k] = S1[j-1,k] + dS1
            S2[j,k] = S2[j-1,k] + dS2
    
    option_payoffs = np.maximum(S1[-1] - S2[-1], 0)
    
    discounted_payoffs = np.exp(-r * T) * option_payoffs
    
    option_price = np.mean(discounted_payoffs)
    
    return S0 + option_price

# Parameters
r = 0.0375  # Interest rate
m = 25  # Number of steps
N = 10000  # Number of simulations

S0 = 15000
option_prices = []

m_values = [10, 25, 50, 75, 100]

# Plotting vt as scatter plot
plt.scatter(maturities, vt, label='closed form')

# Plotting option prices for different values of m as scatter plots
for m_val in m_values:
    option_prices = [simulate_asset_paths(r, S0, SIGMA, T, m_val, N) for T in maturities]
    plt.scatter(maturities, option_prices, label=f'm={m_val}')

plt.xlabel('Maturity (T)')
plt.ylabel('Option Price')
plt.title('Option Price vs. Maturity for Different Values of m')
plt.legend()
plt.grid(True)
plt.show()