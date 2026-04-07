## Heat Map utilziation for two categorical variables ##


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

df = dataset.copy()
df['Passing (%)'] = pd.to_numeric(df['Passing (%)'], errors='coerce')

pivot_df = df.pivot_table(
    index='Rack',
    columns='Shift Name',
    values='Passing (%)',
    aggfunc='mean'
)

pivot_df = pivot_df.sort_index(axis=0).sort_index(axis=1)
custom_cmap = LinearSegmentedColormap.from_list(
    'completion_cmap',
    ['#F2F2F2', '#AAE296'] # Adjust color scheme #
)

plt.figure(figsize=(14, 9))
ax = sns.heatmap(
    pivot_df,
    cmap=custom_cmap,
    annot=True,
    fmt=".0%",
    annot_kws={"size": 11},          # Adjust font size #
    linewidths=0.5,
    cbar_kws={'label': '% Complete'},
    vmin=0.80, # Adjust minimum legend value #
    vmax=.97 # Adjust maximum legend value #
)

ax.invert_yaxis()

# Title - Currrently utilzing Visual Title for uniformity #
#ax.set_title(
#    'Z3 Ruggedization Status – % Complete by Block and Cube',
#    fontsize=18,
#   pad=16
#)

ax.set_xlabel('Shift Name', fontsize=14, labelpad=10)
ax.set_ylabel('Rack', fontsize=14, labelpad=10)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Passing (%)', fontsize=13)

plt.tight_layout()
plt.show()
