from sklearn.datasets import load_wine

x, y = load_wine(return_X_y=True, as_frame=True)

print(x.head(3))
