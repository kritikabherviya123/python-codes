

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Match_Making.csv")
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 0)


from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(features_train, labels_train)


labels_pred = classifier.predict(features_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)


score = classifier.score(features_test,labels_test)





x_min, x_max = features_train[:, 0].min() - 1, features_train[:, 0].max() + 1
y_min, y_max = features_train[:, 1].min() - 1, features_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))


Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.plot(features_test[labels_test == 0, 0], features_test[labels_test == 0, 1], 'ro', label='Class 1')
plt.plot(features_test[labels_test == 1, 0], features_test[labels_test == 1, 1], 'bo', label='Class 2')

plt.contourf(xx, yy, Z, alpha=1.0)

plt.show()
