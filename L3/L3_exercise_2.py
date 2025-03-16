import imblearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_validate,
)
real_labels = [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
model1_res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
model2_res = [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]

print(f"Accuracy for model1: {accuracy_score(real_labels, model1_res):.4f}")
print(f"Accuracy for model2: {accuracy_score(real_labels, model2_res):.4f}")

# Balanced accuracy for model1 = (16/16+0/5)/2 = 0.5
print(
    f"Balanced accuracy for model1: {balanced_accuracy_score(real_labels, model1_res):.3f}"
)
# Balanced accuracy for model2 = (12/16+4/5)/2 = 0.775
print(
    f"Balanced accuracy for model2: {balanced_accuracy_score(real_labels, model2_res):.3f}"
)
cancer = pd.read_table(
    "Cancer_dataset_2.tsv",
    index_col="sample_id",
)
print(cancer.head())


# split the data on features (x) and dependant variable (y)
y = cancer["Response"]
x = cancer.drop("Response", axis=1)
print("\nNumber of patients responded to immunotherapy:")
print(y.value_counts())



# Your code here

# Инициализация моделей
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "Balanced RF (class_weight)": RandomForestClassifier(
        class_weight='balanced', random_state=42
    ),
    "BalancedRandomForest (imblearn)": imblearn.ensemble.BalancedRandomForestClassifier(
        random_state=42
    ),
}

# Настройки кросс-валидации
cv_methods = {
    "KFold": KFold(n_splits=5, shuffle=True, random_state=42),
    "StratifiedKFold": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
}

# Сбор результатов
results = []
for model_name, model in models.items():
    for cv_name, cv in cv_methods.items():
        scores = cross_validate(
            model,
            x,
            y,
            cv=cv,
            scoring=["accuracy", "balanced_accuracy"],
            n_jobs=-1,
        )
        results.append(
            {
                "Model": model_name,
                "CV Method": cv_name,
                "Accuracy": np.mean(scores["test_accuracy"]),
                "Balanced Accuracy": np.mean(scores["test_balanced_accuracy"]),
            }
        )

# Визуализация результатов
results_df = pd.DataFrame(results)
print("\nРезультаты кросс-валидации:")
print(
    results_df.pivot(
        index="Model", columns="CV Method", values=["Accuracy", "Balanced Accuracy"]
    )
    .round(3)
    .to_string()
)

# Графическое представление
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
for i, metric in enumerate(["Accuracy", "Balanced Accuracy"]):
    sns.barplot(
        data=results_df,
        x="Model",
        y=metric,
        hue="CV Method",
        ax=axes[i],
        errorbar=None,
    )
    axes[i].set_title(metric)
    axes[i].set_ylim(0, 1)
plt.tight_layout()
plt.show()