# Example matplotlib visuals #

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({'X': [1, 2, 3], 'Y': [4, 5, 6]})
plt.plot(df['X'], df['Y'])
plt.title("Test Plot")
plt.show()
