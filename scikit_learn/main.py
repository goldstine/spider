"""
    什么是scikit-learn
    (1)面向python的免费机器学习库
    （2）包含分类，回归，聚类算法，比如：SVM，随机森林，k-means等
    （3）包含降维，模型筛选，预处理等算法
    （4）支持numpy和scipy等数据结构
    用户：http://scikit-learn.org/stable/testimonials/testimonials.html
    安装：pip install scikit-learn  或者升级一下：pip install -u(-upgrade) scikit-learn
    conda install scikit-learn



    (1)加载示例数据集iris/digits
    (2)在训练集上训练数据集svm model fit() 训练模型
    (3)在测试集上测试模型predict()进行预测
    (4)保存模型
    pickle.dumps()

"""
from sklearn import datasets
#加载示例数据集
iris=datasets.load_iris()
digits=datasets.load_digits()

#查看数据集
#iris
print(iris.data)
print(iris.data.shape)
print(iris.target_names)
print(iris.target)

#digits数据集
print(digits.data)
print(digits.data.shape)
print(digits.target_names)
print(digits.target)

#手动划分数据集
n_test=100  #测试样本个数
train_X=digits.data[:-n_test,:]
train_y=digits.target[:-n_test]

test_X=digits.data[-n_test:,:]
y_true=digits.target[-n_test:]

#选择svm模型
from sklearn import svm
svm_model=svm.SVC(gamma=0.001,C=100.)#设置支持向量分类的超参数
#svm_model=svm.SVC(gamma=100.,c=1.)

#训练模型
svm_model.fit(train_X,train_y)


#选择LR模型
from sklearn.linear_model import LogisticRegression
# lr_model=LogisticRegression()
lr_model=LogisticRegression(max_iter=3000)
#训练模型
lr_model.fit(train_X,train_y)

#在测试集上测试模型
y_pred_svm=svm_model.predict(test_X)
y_pred_lr=lr_model.predict(test_X)

#查看结果
from sklearn.metrics import accuracy_score

#print '预测标签：’，y_pred
#print ‘真实标签:',y_true
print('svm结果：',accuracy_score(y_true,y_pred_svm))

print('lr结果：',accuracy_score(y_true,y_pred_lr))


#保存模型
import pickle
with open('svm_model.pkl','wb') as f:
    pickle.dump(svm_model,f)

#重新加载预测模型
import numpy as np

with open('svm_model.pkl','rb') as f:
    model=pickle.load(f)

random_samples_index=np.random.randint(0,1796,5)
print(random_samples_index)
random_samples=digits.data[random_samples_index, :]
random_targets=digits.target[random_samples_index]

random_predict=model.predict(random_samples)

print(random_predict)
print(random_targets)

