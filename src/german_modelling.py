# Data Science Libraries
import pandas as pd
import numpy as np

# Sklearn modules
from sklearn.neighbors import KNeighborsClassifier

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns




def best_knn(x_train, y_train):
    k_range = range(1, 20)
    train_scores = []
    validate_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train, y_train)
        train_scores.append(knn.score(X_train, y_train))
        validate_scores.append(knn.score(X_validate, y_validate))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.plot(k_range, train_scores, label='Train')
    plt.plot(k_range, validate_scores, label='Validate')
    plt.legend()
    plt.xticks([0,5,10,15,20])
    plt.show()