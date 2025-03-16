import lightgbm
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.datasets
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from warnings import simplefilter

simplefilter("ignore", category=FutureWarning)
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
print(breast_cancer.DESCR)

x = breast_cancer.data
y = breast_cancer.target
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

def bootstrap_metric(y_true, y_pred, metric_fn, samples_cnt=1000, random_state=42):
    np.random.seed(random_state)
    b_metric = np.zeros(samples_cnt)
    for i in range(samples_cnt):
        poses = np.random.choice(y_true.shape[0], size=y_true.shape[0], replace=True)

        y_true_boot = y_true[poses]
        y_pred_boot = y_pred[poses]
        m_val = metric_fn(y_true_boot, y_pred_boot)
        b_metric[i] = m_val

    return b_metric


models = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'LGBM': lightgbm.LGBMClassifier(random_state=42),
    'SVC': SVC(random_state=42),
    'BaggingSVC': BaggingClassifier(
        estimator=SVC(random_state=42),
        n_estimators=10,
        random_state=42
    )
}

for name, model in models.items():
    model.fit(x_train, y_train)

# Предсказания на тесте
y_preds = {name: model.predict(x_test) for name, model in models.items()}

# Расчет MCC и бутстрэп
bootstrap_results = {}
for name, y_pred in y_preds.items():
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"{name} MCC: {mcc:.3f}")
    bootstrap_results[name] = bootstrap_metric(y_test, y_pred, matthews_corrcoef)

# Подготовка данных для графиков
data_for_plot = pd.DataFrame()
for name, mcc_values in bootstrap_results.items():
    temp_df = pd.DataFrame({'Model': name, 'MCC': mcc_values})
    data_for_plot = pd.concat([data_for_plot, temp_df])

# Боксплот
plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='MCC', data=data_for_plot)
plt.title('Распределение корреляции Мэтьюса')
plt.ylabel('MCC')
plt.xlabel('Модель')
plt.show()

# Доверительные интервалы
for name, mcc_values in bootstrap_results.items():
    lower = np.percentile(mcc_values, 5)
    upper = np.percentile(mcc_values, 95)
    print(f"{name} 90% ДИ: [{lower:.3f}, {upper:.3f}]")


