import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

## Read parquet
parquet_path = "dataset/full_train.parquet"
df = pd.read_parquet(parquet_path)

## Exclude unwanted classes
exclude_types = {"unknown", "background", "construction", "static", "riderless_motorcycle", "riderless_bicycle"}
df = df[~df["object_type"].isin(exclude_types)].copy()

## Print first rows of df
print("========= HEAD OF DF =========")
print(df.head())

## Get labels of x and y
x_lab = []
y_lab = []
for c in df.columns:
    if c.startswith("position_x_"):
        x_lab.append(c)
    elif c.startswith("position_y_"):
        y_lab.append(c)

## Plot 6 random trajectories
fig, axs = plt.subplots(2, 3)
for j in range(6):
    random_index = np.random.randint(0, len(df))
    x_pos = []
    y_pos = []
    for i in range(1, len(x_lab)):
        x = df[x_lab[i]].values[random_index]
        y = df[y_lab[i]].values[random_index]
        x_pos.append(x)
        y_pos.append(y)
        # print(x, y)
    ax = axs[j//3, j%3]
    points = np.array([x_pos, y_pos]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='plasma')
    lc.set_array(np.linspace(1, 109, len(segments)))
    ax.add_collection(lc)
    ax.autoscale()
    plt.colorbar(lc, ax=ax, label='Timestep')
    ax.scatter(x_pos, y_pos, c=np.arange(len(x_pos)), cmap='plasma',  s=5, zorder=3)
    ax.set_title(f"Trajectory of {df['object_type'].values[random_index]}")
    ax.grid()
plt.show()

