import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

#Download stock price
data = yf.download("AAPL", period="1y")

#Calculate daily returns
returns = data["Close"].pct_change().dropna()

#Calculate average and standard deviation
mean = returns.mean()
std = returns.std()

print("Average Daily Return: ", mean)
print("Standard Deviation: ", std)

#Draw histogram
plt.hist(returns, bins=50, density=True, alpha=0.6)

#Draw normal bell curve
x = np.linspace(returns.min(), returns.max(), 200)
y = stats.norm.pdf(x, mean, std)

plt.plot(x,y)
plt.title("Apple Daily Return vs Normal Distribution")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")


#Sharpiro-Wilk Normality Test
sample = returns.sample(n=min(5000, len(returns)), random_state=42)

statistsics, p_value = stats.shapiro(sample)

print("P-value: ", p_value)

if p_value < 0.05:
  print("Evidence against Normality")
else:
  print("not enough evidence against normality")
