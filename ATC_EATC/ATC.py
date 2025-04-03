# Author : Guo JiaLi
# CreateTime : 2023/3/26
# FileName : ATC.py
# Description : Standard ATC for LST reconstruction
import datetime as dt
from osgeo import gdal,ogr,osr
import os
import numpy as np
from math import sqrt
from scipy.optimize import leastsq  # 导入 scipy 中的最小二乘法工具
import pandas as pd

class IMG:
    date = None
    path = None
    ds = None  #影像数据
    vt_array = None  #以array形式读取

    def __init__(self, former_path, latter_path):  #两个参数分别代表打开影像的前后两个path
        self.value = latter_path
        self.date = get_d(latter_path)

        img_path = former_path + '\\' + latter_path
        self.ds = gdal.Open(img_path)
        self.vt_array = self.ds.ReadAsArray()  #以array形式读取


# 得到拟合中距春分的天数
def get_d(a):
    # a = 'LC08_123032_20160129_warp.tif'  #  file_all 即此处的a
    # date = int(a[12:19])  #截取日期

    date = a[12:20]

    A= list(date) # 转化
    A.insert(4,'-')
    A.insert(7,'-')

    d1 = ''.join(A) # 转化回来
    d2 = '2016-03-20'

    date1 = dt.datetime.strptime(d1, "%Y-%m-%d").date()  ##datetime.date(2016, 1, 06)
    date2 = dt.datetime.strptime(d2, "%Y-%m-%d").date()  ##datetime.date(2016, 3, 20)

    Days = (date1 - date2).days# 计算两个日期date的天数差

    # Days = date - 80

    return Days


# 定义拟合函数以及残差计算函数
def fitfunc3(p, x):  # 定义拟合函数
    p0, p1, p2 = p  # 拟合函数的参数
    y = p0 + p1 * np.sin(x*np.pi*2/366+p2)  # 拟合函数的表达式
    return y

def error3(p, x, y):  # 定义残差函数
    err = fitfunc3(p,x) - y  # 残差
    return err

input_file = "E:\\yulin\\yulin_warp"
file_all = os.listdir(input_file)
img_count = len(file_all)

img_collection = {}
ds = np.array(img_count)

file_all_new = list()
for file_i in file_all:
    the_date = file_i[12:]  # 截取日期
    file_all_new.append(the_date)

file_all_new.sort()  #  对影像按照时间重新排序

print(file_all_new)

print(type(file_all_new))
for file_i_new in file_all_new:
    if file_i_new.endswith('.tif'):

        print("file_i_new ",file_i_new)

        # LE07_127033_20161218_warp

        # beijing
        file_i_1 = 'LE07_123032_'

        # weishan
        # file_i_2 = 'LC08_123035_'
        # file_i_3 = 'LE07_123035_'

        # yulin
        file_i_2 = 'LC08_127033_'
        file_i_3 = 'LE07_127033_'

        file_key = int(file_i_new[0:8])
        print("1",file_key)
        if os.path.exists(input_file+"\\"+file_i_1+file_i_new):  # 判断是LE07还是LC08，以此按顺序打开影像
            img_collection['img' + str(file_key)] = IMG(input_file,file_i_1+file_i_new)  # 批量创建影像对象
        else :
            print("fsdvggsfb")
            if os.path.exists(input_file+"\\"+file_i_2+file_i_new):
                img_collection['img' + str(file_key)] = IMG(input_file, file_i_2+file_i_new)
            else:
                img_collection['img' + str(file_key)] = IMG(input_file, file_i_3 + file_i_new)

        print(img_collection['img' + str(file_key)].date)

print('Img collection have been created!')
print(len(img_collection))




# 预设存储数据的数组
num = 500
index = np.zeros(num)
rmse_error = np.zeros((num,num))
Mast= np.zeros((num,num))
Yast= np.zeros((num,num))
Theta= np.zeros((num,num))

img_20161128 = np.zeros((num,num))

pFit = [0,0,0]
p = [225.2770, 75.6377, 170.2353]

list_x = []  # 创建两个空列表存储日期和像素值
list_y = []
core_y_in_2016 = []
print(file_all_new)
for i_col in range(0,500) :
    for j_row in range(0,500):
        for file_i_new in file_all_new:
            if file_i_new.endswith('.tif'):

                file_key = int(file_i_new[0:8])
                d = img_collection['img' + str(file_key)].date
                list_x.append(d)

                pixel_now = img_collection['img' + str(file_key)].vt_array[i_col][j_row]  # 取当前位置上的像素值

                list_y.append(pixel_now)

        print('-----------It is turn to [',i_col,',',j_row,']-----------' )

        list_y = [elem_y if not np.isnan(elem_y) else None for elem_y in list_y]  #  查询list_y里是否存在None值
        if None in list_y:  #有空值那么直接将数值赋为0

            pos = [i for i, x in enumerate(list_y) if x == None]  # 获取None值的索引
            list_x = [n for x_i, n in enumerate(list_x) if x_i not in pos]  # 清除y_Obs里的空值对应的天数
            list_y = list(filter(None, list_y))  # 清除y_Obs里的空值

            list_a = []  # 置零重新参与计算

        if len(list_x) < 3:
            print('数据过少，无法进行拟合！')
            pFit = [0,0,0]

            Mast[i_col][j_row] = pFit[0]
            Yast[i_col][j_row] = pFit[1]
            Theta[i_col][j_row] = pFit[2]
            print(pFit[0],pFit[1],pFit[2])

            list_x = []
            list_y = []
            continue;  #小于参数个数，无法进行拟合

        print(list_x)
        x = np.array(list_x)  # 日期d
        y_Obs = np.array(list_y)  # LST_Obs
        length = len(y_Obs)

        # 由给定数据点集 (x,y) 求拟合函数的参数 pFit
        p0 = [280, 15, 160]  # 设置拟合函数的参数初值
        element_exist = np.nan in y_Obs  # 判断LST数组里是否存在前面设置的零值

        if element_exist  ==True:
            # print(True)
            p0=[0,0,0]  # 存在的话，暂时将三参数赋值为零值，后面再进行插值
        else:
            pFit, info = leastsq(error3, p0, args=(x, y_Obs))  # 最小二乘法求拟合参数


        print(pFit[0])
        print(pFit[1])
        print(pFit[2])
        print('-----------  Fitting has finished!  -------------')
        print('                                  ')
        Mast[i_col][j_row] = pFit[0]
        Yast[i_col][j_row] = pFit[1]
        Theta[i_col][j_row] = pFit[2]


        # # # # # # # # # # # # # # # 计算各点残差# # # # # # # # # # # # # # #
        sum_error = 0
        for j in range(0, length):
            error = (fitfunc3(pFit, x[j]) - y_Obs[j]) * (fitfunc3(pFit, x[j]) - y_Obs[j])
            sum_error = sum_error + error

        #修改为二维行列的矩阵
        rmse_error[i_col][j_row] = sqrt(sum_error / length)  #计算单点上的误差矩阵
        p[2] = p0[2] + 0.2
        p[0] = p0[0]
        p[1] = p0[1]


        if i_col == 250 and j_row == 250:

            y_Fit_core = []
            for j in range(0, length):
                core_j = fitfunc3(pFit, x[j])
                y_Fit_core.append(core_j)
        #
        #     array00 = np.array(x)
        #     df00 = pd.DataFrame(array00)
        #     df00.to_csv('F:/SAT/weishan_x.csv')
        #
            for day in range(-79,286):
                y_core= fitfunc3(pFit, day)
                core_y_in_2016.append(y_core)

            array44 = np.array(core_y_in_2016)
            df44 = pd.DataFrame(array44)
            df44.to_csv('E:\\yulin\\yulin_2016_1.csv')

            array11 = np.array(y_Obs)
            df11 = pd.DataFrame(array11)
            df11.to_csv('E:\\yulin\\yulin_yObs_1.csv')

            array22 = np.array(y_Fit_core)
            df22 = pd.DataFrame(array22)
            df22.to_csv('E:\\yulin\\yulin_yFit_1.csv')


        list_x = []  # 置零重新参与计算
        list_y = []

print('All finished!')
#  存储到xls文件中

array1 = np.array(Mast)
df1 = pd.DataFrame(array1)
df1.to_csv('E:\\yulin\\yulin_1.csv')

array2 = np.array(Yast)
df2 = pd.DataFrame(array2)
df2.to_csv('E:\\yulin\\yulin _2.csv')

array3 = np.array(Theta)
df3 = pd.DataFrame(array3)
df3.to_csv('E:\\ yulin\\yulin_3.csv')

print('Saving File Over!')






