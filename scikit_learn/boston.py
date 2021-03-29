#回归模型,波士顿房价
from sklearn import datasets

#选择线性回归模型
from sklearn.linear_model import LinearRegression
#对训练集和测试集进行分割
from sklearn.model_selection import train_test_split

boston_data=datasets.load_boston()
X=boston_data.data
y=boston_data.target

print("样本：")
print(X[:5,:])
print('标签：')
print(y[:5])


lr_model=LinearRegression()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3.,random_state=7)

lr_model.fit(X_train,y_train)

#返回参数
print(lr_model.get_params())

print(lr_model.score(X_train,y_train))

print(lr_model.score(X_test,y_test))

