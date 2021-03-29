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

