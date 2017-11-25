import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# インプットを乱数で生成
x = np.sort(5 * np.random.rand(40,1), axis=0)
# アウトプットはsin関数
y = np.sin(x).ravel()

# アウトプットにノイズを与える
y[::5] += 3 * (0.5 - np.random.rand(8))

# RBFカーネル、線形、多項式でフィッティング
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
svr_rbf.fit(x, y)
svr_lin.fit(x, y)
svr_poly.fit(x, y)
y_rbf = svr_rbf.predict(x)
y_lin = svr_lin.predict(x)
y_poly = svr_poly.predict(x)

# 図を作成
plt.figure(figsize=[10, 5])
plt.scatter(x, y, c='k', label='data')  # 散布図を描く
plt.hold('on')
plt.plot(x, y_rbf, c='b', label='RBF model')
plt.plot(x, y_lin, c='g', label='Linear model')
plt.plot(x, y_poly, c='r', label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()



