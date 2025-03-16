import random
import numpy as np


def set_seed():
    random.seed(42)
    np.random.seed(42)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from matplotlib.pyplot import figure
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("/Users/vfast/Library/CloudStorage/OneDrive-Личная/МГУ/MSU AI/GigaProject/MSU_AI/L2/petrol_consumption.csv")
print(dataset.shape)
# print(dataset.head())
# print(dataset.describe())

x = dataset[
    ["Petrol_tax", "Average_income", "Paved_Highways", "Population_Driver_licence(%)"]
]
y = dataset["Petrol_Consumption"]
set_seed()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Инициализация StandardScaler
scaler = StandardScaler()

# Применение scaler к обучающим данным
x_train_scaled = scaler.fit_transform(x_train)

# Применение scaler к тестовым данным (используем transform, а не fit_transform, чтобы не "подглядывать" в тестовые данные)
x_test_scaled = scaler.transform(x_test)

# Multiple regression
# Your code here
regressor = LinearRegression()
regressor.fit(x_train_scaled, y_train)

# Print table

coeff_df = pd.DataFrame(regressor.coef_, x.columns, columns=["Coefficient"])
coeff_df

y_pred = regressor.predict(x_test_scaled)


df = pd.DataFrame({"Real": y_test, "Predicted": y_pred, "Delta": y_test - y_pred})

import seaborn as sns
from matplotlib.pyplot import figure

figure(figsize=(6, 4))
sns.residplot(x="Real", y="Delta", data=df)
plt.show()

# Metrics
# Your code here
print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score:", metrics.r2_score(y_test, y_pred))


# Вывод коэффициентов модели
print("\nКоэффициенты модели:")
for feature, coef in zip(x.columns, regressor.coef_):
    print(f"{feature}: {coef}")
print("Intercept (свободный член):", regressor.intercept_)






