import pandas as pd

# Load CSV files
csv1 = pd.read_csv('random_1.csv')
csv2 = pd.read_csv('random_2.csv')
csv3 = pd.read_csv('random_3.csv')
csv4 = pd.read_csv('random_4.csv')

# Concatenate along rows
result = pd.concat([csv1, csv2, csv3, csv4])

# Save the concatenated data to a new CSV file
result.to_csv('total.csv', index=False)