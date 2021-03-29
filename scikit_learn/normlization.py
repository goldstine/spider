"""
scikit-learn模型的选择
reference：scikit-learn.org/stable/tutorial/machine_learning_map/index.html
"""

#特征归一化，在列上进行

import numpy as np
from sklearn.model_selection import train_test_split

X=np.random.randint(0,100,(10,4))
y=np.random.randint(0,3,10)
y.sort()
print('样本:')
print(X)
print('标签：',y)

#分割训练集，测试集
#random_state确保每次随机分割得到相同的结果,random_satte相当于随机数种子，可以复现
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3.,random_state=7)

print('训练集：')
print(X_train)
print(y_train)

print('测试集：')
print(X_test)
print(y_test)

#特征归一化
from sklearn import preprocessing
x1=np.random.randint(0,1000,5).reshape(5,1)
x2=np.random.randint(0,10,5).reshape(5,1)
x3=np.random.randint(0,100000,5).reshape(5,1)

X=np.concatenate([x1,x2,x3],axis=1)
print(X)
print(preprocessing.scale(X))



#生成分类数据进行验证scale的必要性
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
# %matplotlib inline
X,y=make_classification(n_samples=300,n_features=2,n_redundant=0)

plt.scatter(X[:,0],X[:,1],c=y)
plt.show()
