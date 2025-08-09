from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print("Accuracy:", round(clf.score(Xte, yte), 3))
