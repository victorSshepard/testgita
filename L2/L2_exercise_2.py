import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

from L2_exercise_1 import set_seed
set_seed()


def twospirals(n_points, noise=0.8):
    """
    Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    return (
        np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
        np.hstack((np.zeros(n_points), np.ones(n_points))),
    )


x, y = twospirals(500)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y, s=50, ax=ax)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["class 0", "class 1"])
ax.set(xlabel="feature 1", ylabel="feature 2")
plt.show()

set_seed()
# Your code here

from sklearn.model_selection import train_test_split

# Разделение данных на тренировочную и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Проверка размеров выборок
print("Размер тренировочной выборки (x_train):", x_train.shape)
print("Размер тестовой выборки (x_test):", x_test.shape)
print("Размер тренировочных меток (y_train):", y_train.shape)
print("Размер тестовых меток (y_test):", y_test.shape)
# Инициализация StandardScaler
scaler = StandardScaler()

# Применение scaler к обучающим данным
x_train_scaled = scaler.fit_transform(x_train)

# Применение scaler к тестовым данным (используем transform, а не fit_transform, чтобы не "подглядывать" в тестовые данные)
x_test_scaled = scaler.transform(x_test)


from sklearn import svm
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay


def plot_svm(x, y, clf):
    dull_cmap = ListedColormap(["#B8E1EC", "#FEE7D0"])
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        x,
        response_method="predict",
        cmap=dull_cmap,
        alpha=0.8,
        xlabel="feature 1",
        ylabel="feature 2",
        ax=ax,
    )

    sns.scatterplot(
        x=x[:, 0],
        y=x[:, 1],
        hue=y,
        s=50,
        ax=ax,
        palette=sns.color_palette(["#2DA9E1", "#F9B041"]),
    )
    plt.show()
clf = svm.SVC(kernel="linear")
clf.fit(x_train_scaled, y_train)
plot_svm(x_train_scaled, y_train, clf)

clf = svm.SVC(kernel="poly")
clf.fit(x_train_scaled, y_train)
plot_svm(x_train_scaled, y_train, clf)

clf = svm.SVC(kernel="rbf")
clf.fit(x_train_scaled, y_train)
plot_svm(x_train_scaled, y_train, clf)

from sklearn.metrics import accuracy_score

# Линейное ядро
clf_linear = svm.SVC(kernel="linear")
clf_linear.fit(x_train_scaled, y_train)
y_pred_linear = clf_linear.predict(x_test_scaled)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f"Accuracy for linear kernel: {accuracy_linear:.4f}")

# Полиномиальное ядро
clf_poly = svm.SVC(kernel="poly")
clf_poly.fit(x_train_scaled, y_train)
y_pred_poly = clf_poly.predict(x_test_scaled)
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print(f"Accuracy for polynomial kernel: {accuracy_poly:.4f}")

# RBF ядро
clf_rbf = svm.SVC(kernel="rbf")
clf_rbf.fit(x_train_scaled, y_train)
y_pred_rbf = clf_rbf.predict(x_test_scaled)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"Accuracy for RBF kernel: {accuracy_rbf:.4f}")


# Your code here

from sklearn.model_selection import GridSearchCV

# Определение сетки параметров
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# Инициализация модели SVM с RBF ядром
svm_rbf = svm.SVC(kernel='rbf')

# Инициализация GridSearchCV
grid_search = GridSearchCV(
    estimator=svm_rbf,
    param_grid=param_grid,
    cv=4,  # 4 фолда
    scoring='accuracy',  # Метрика accuracy
    n_jobs=-1  # Использование всех доступных ядер процессора
)

# Поиск лучших параметров
grid_search.fit(x_train_scaled, y_train)

# Лучшие параметры
best_params = grid_search.best_params_
print(f"Лучшие параметры: {best_params}")

# Your code here


# Лучшая модель
best_svm_rbf = grid_search.best_estimator_

# Предсказание на тестовых данных
y_pred_best = best_svm_rbf.predict(x_test_scaled)

# Оценка accuracy
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Accuracy для лучшей модели: {accuracy_best:.4f}")

# Визуализация границы принятия решений
plot_svm(x_train_scaled, y_train, best_svm_rbf)
