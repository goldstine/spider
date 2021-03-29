#交叉验证
# https://blog.csdn.net/qq_36523839/article/details/80707678
from sklearn import datasets
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# %matplotlib inline

iris=datasets.load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3.,random_state=7)
k_range=range(1,31)
cv_scores=[]
for n in k_range:
    knn=KNeighborsClassifier(n)
    scores=cross_val_score(knn,X_train,y_train,cv=10,scoring='accuracy')
    cv_scores.append(scores.mean())

plt.plot(k_range,cv_scores)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

#选择最优的K
best_knn=KNeighborsClassifier(n_neighbors=5)
best_knn.fit(X_train,y_train)
print(best_knn.score(X_test,y_test))
print(best_knn.predict(X_test))
