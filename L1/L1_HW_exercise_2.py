import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

import pandas as pd
# import wget
# url = 'https://edunet.kea.su/repo/EduNet-web_dependencies/datasets/y_pred_proba.csv'
# filename = wget.download(url)
# print(filename)

df = pd.read_csv("y_pred_proba.csv")
# print(df.head())

df["y_pred_05"] = df['y_score'].map(lambda x: 0 if x <0.5 else 1)

# print(df)
# print(df.y_true.unique())

from sklearn.metrics import classification_report
target_names = ["class 0", "class 1"]

# print(classification_report(df["y_true"], df["y_pred_05"], target_names=target_names))


# sns.histplot(df.y_score, kde=True)
import seaborn as sns
import matplotlib.pyplot as plt


# sns.histplot(df.y_score)
#
# plt.axvline(x=0.5, c="r", linestyle="--")
# plt.show()

# Создаем гистограмму предсказанных вероятностей с раскраской по истинным классам
sns.histplot(data=df, x="y_score", hue="y_true", multiple="stack")

# Добавляем заголовок и метки осей
plt.title("Гистограмма предсказанных вероятностей")
plt.xlabel("Предсказанные вероятности")
plt.ylabel("Количество")

# Отображаем гистограмму
# plt.show()

from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, auc


precision, recall, thresholds = precision_recall_curve(df["y_true"], df["y_score"])
pr_auc = auc(recall, precision)
print(f"PR AUC: {pr_auc:.2f}")



pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
plt.plot(recall, precision, label=f"PR curve (area = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")

# pr_display.plot()
# plt.show()


# from sklearn.metrics import precision_score, recall_score
#
# y_true = list(map(int, list(df['y_true'])))
# y_pred = list(map(int, list(df['y_score'])))
#
# print(f'{len(y_pred)}, {len(y_true)}')
# precision = precision_score(y_true, y_pred)
#
# df['recall'] = recall_score(df['y_true'], df['y_pred'])
#
# print(f"Precision: {precision:.3f}")
# print(f"Recall: {recall:.3f}")

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(df["y_true"], df["y_score"])

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Thresholds: {thresholds}")

# from sklearn.metrics import f1_score
#
# f1_scores = [f1_score(df["y_true"], df["y_score"] > threshold) for threshold in thresholds]
#
# max_f1_score = max(f1_scores)
# max_f1_threshold = thresholds[f1_scores.index(max_f1_score)]
#
# print(f"Max F1-score: {max_f1_score:.2f}")
# print(f"Threshold at max F1-score: {max_f1_threshold:.2f}")


# Вычисляем значение F1-score
f1_score = 2 * precision * recall / (precision + recall)

# Находим порог, при котором F1-score максимален
max_f1_score = np.max(f1_score)
max_f1_threshold = thresholds[np.argmax(f1_score)]

print(f"Max F1-score: {max_f1_score:.2f}")
print(f"Threshold at max F1-score: {max_f1_threshold:.2f}")


