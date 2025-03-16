import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
data, labels = load_wine(return_X_y=True, as_frame=True)
dn = data.size
dl = labels.size
print(f'data = {dn}; labels = {dl}')
# print(data.columns)
# print(data.info())
# print(data[["alcohol", "malic_acid", "ash"]].describe())
data["alcohol"] = data["alcohol"].astype("float32")
data["malic_acid"] = data["malic_acid"].astype("float32")
data["ash"] = data["ash"].astype("float32")
data["alcalinity_of_ash"] = data["alcalinity_of_ash"].astype("float32")
# print(data.info())
data.sort_values(by="alcohol", ascending=False).head()
# print(data.head(3))

df = pd.DataFrame(data)
cl = df.columns.tolist()
print(cl)
data.sort_values(by=cl[:3], ascending=False).head()
# print(data.head(3))
d = {2: "low", 3: "low"}
data["ash"] = data["ash"].map(d)

print(data.ash.unique())
data["ash"] = data["ash"].map(lambda x: "low" if pd.isna(x) else x)
print(data.head(3))
# Группировка данных  выведем статистики по трём столбцам в зависимости от значения признака alcohol:
columns_to_show = ["malic_acid", "total_phenols", "proanthocyanins"]
d = data.groupby(["alcohol"])[columns_to_show].describe(percentiles=[])
print(d)
d = data.groupby(["alcohol"])[columns_to_show].agg(["mean", "std", "min", "max"])
print(d)
import matplotlib.pyplot as plt

# fig, axs = plt.subplots(figsize=(4, 3))
# labels.hist()
# plt.suptitle("Label balance")
# plt.show()


df = pd.concat([data, labels], axis=1)
# print(df)
# df.groupby("target")["alcohol"].mean().plot(legend=True)
# plt.show()
# df.groupby("target")["alcohol"].mean().plot(kind="bar", legend=True, rot=45)
# plt.show()

import seaborn as sns

data, labels = load_wine(return_X_y=True, as_frame=True)
df = pd.concat([data, labels], axis=1)
# cols = ["alcohol", "malic_acid", "ash", "target"]
# sns_plot = sns.pairplot(df[cols])
# sns_plot.savefig("pairplot.png")
# data, labels = load_wine(return_X_y=True, as_frame=True)
# df = pd.concat([data, labels], axis=1)
# cols = cl[3:9] + ["target"]
# sns_plot = sns.pairplot(df[cols])
# sns_plot.savefig("pairplot.png")

sns.histplot(df.color_intensity, kde=True)
plt.show()

# top_alcohol = (
#     df.alcohol.value_counts().sort_values(ascending=False).head(5).index.values
# )
# sns.boxplot(
#     y="alcohol", x="flavanoids", data=df[df.alcohol.isin(top_alcohol)], orient="h"
# )
# plt.show()

# top_alcohol = (
#     df.alcohol.value_counts().sort_values(ascending=False).head(20).index.values
# )
# sns.boxplot(
#     y="alcohol", x="color_intensity", data=df[df.alcohol.isin(top_alcohol)], orient="h"
# )
# plt.show()

df["alcoholGroup"] = pd.cut(df["alcohol"], bins=5)
platform_genre_sales = (
    df.pivot_table(
        index="target",
        columns="alcoholGroup",
        values="proanthocyanins",
        aggfunc="sum",
        observed=False,
    )
    .fillna(0)
    .map(float)
)
sns.heatmap(platform_genre_sales, annot=True, fmt=".1f", linewidths=0.05)
# plt.show()