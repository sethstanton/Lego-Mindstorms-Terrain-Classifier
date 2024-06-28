import pandas as pd
import ast  # To safely evaluate string literals as Python expressions

# Empty list to store rows
data = []

# Open the file and read line by line
with open('Stones.txt', 'r') as file:
    for line in file:
        # Strip newlines and other surrounding whitespace
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            # Safely evaluate the string as a Python list
            row = ast.literal_eval(line)
            data.append(row)

# Create a DataFrame from the list of rows
df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(df.head())

# Now, check for NaN values in this DataFrame
print("\nPresence of NaN in each column:")
print(df.isna().any())
print("\nNumber of NaNs in each column:")
print(df.isna().sum())

# Display rows with NaN values to find their exact locations
nan_rows = df[df.isna().any(axis=1)]
print("\nRows with NaN values:")
print(nan_rows)
