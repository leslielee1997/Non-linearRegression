# 导入相关库与数据集
import numpy as np
import matplotlib.pyplot as plt
import openpyxl as xl 
from scipy import optimize
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False   
Filename = 'F:\code\meachine-learning\data.xlsx'
Sheet = 'Sheet1'
data = xl.load_workbook(Filename)[Sheet]
def getColValues(data,column):
    rows = data.max_row
    columndata=[]
    for i in range(1,rows+1):
        cellvalue = data.cell(row=i,column=column).value
        columndata.append(cellvalue)
    return columndata
data_x = np.array(getColValues(data,1))
data_y = np.array(getColValues(data,2))
data_n = np.array(getColValues(data,3))
# 定义高斯过程类
class GPR:
    def __init__(self, h):
        self.is_fit = False
        self.train_x, self.train_y = None, None
        self.h = h
    def fit(self, x, y):
        self.train_x = np.asarray(x)
        self.train_y = np.asarray(y)
        self.is_fit = True
    def predict(self, x):
        if not self.is_fit:
            print("不能拟合!")
            return
        x = np.asarray(x)
        kff = self.kernel(x,x)
        kyy = self.kernel(self.train_x, self.train_x)
        kfy = self.kernel(x, self.train_x)
        kyy_inv =  np.linalg.inv(kyy + 1e-8*np.eye(len(self.train_x)))
        mu = kfy.dot(kyy_inv).dot(self.train_y)
        return mu
    def kernel(self, x1, x2):
        m,n = x1.shape[0], x2.shape[0]
        dist_matrix = np.zeros((m,n), dtype=float)
        for i in range(m):
            for j in range(n):
                dist_matrix[i][j] = np.sum((x1[i]-x2[j])**2)
        return np.exp(-0.5/self.h**2*dist_matrix)
# 显示训练集的分布
h=0.1
for i in range(5):
    gpr = GPR(h)
    gpr.fit(data_x, data_n)
    mu = gpr.predict(data_x)
    test_y = mu.ravel()
    plt.figure()
    plt.title("h=%.2f"%(h))
    plt.plot(data_x, test_y, label="拟合曲线")
    plt.scatter(data_x, data_n, label="带噪声的真实数据", c="red",s=5)
    plt.legend()
    h += 0.1