"""
Generate dummy data for a noisy exponentially decaying sine function
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# generate features
n_data = 100
x_data = np.linspace(-20, 20, n_data)

# generate targets
fn = lambda x: np.exp(-0.1*x) * np.sin(x) 
y_data = fn(x_data) + np.random.rand(n_data)*0.5

# create df
df = pd.DataFrame({
    'Index': range(1,n_data+1),
    'X': x_data,
    'Y': y_data
})
df.to_csv("dummy_data.csv")

# plot 
plt.figure(figsize=(5,4))
plt.plot(x_data, fn(x_data), label='True Function')
plt.scatter(x_data, y_data, label='Collected Data', alpha=0.2)
plt.legend()
plt.xlabel("Feature, x")
plt.ylabel("Target, y")
plt.savefig(f"dummy_data.png")
