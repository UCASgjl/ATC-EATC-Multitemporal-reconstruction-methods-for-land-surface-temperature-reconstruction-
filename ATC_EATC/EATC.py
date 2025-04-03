# Author : GJL
# CreateTime : 2023/4/19
# FileName : EATC.py
# Description : include SAT and NDVI, EATC model
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

class SAT_img:
    # SAT_date = None
    SAT_ds = None  # 影像数据
    SAT_array = None  # 以array形式读取

    def __init__(self, former_path, latter_path):  # 两个参数分别代表打开影像的前后两个path

        img_path = former_path + '\\' + latter_path
        self.SAT_ds = gdal.Open(img_path)
        self.SAT_array= self.SAT_ds.ReadAsArray()  # 以array形式读取

class SAT_ATC_img:

    SAT_ATC_ds = None  # 影像数据
    SAT_ATC_array = None  # 以array形式读取

    def __init__(self, former_path, latter_path):  # 两个参数分别代表打开影像的前后两个path

        img_path = former_path + '\\' + latter_path
        self.SAT_ATC_ds = gdal.Open(img_path)
        self.SAT_ATC_array= self.SAT_ATC_ds.ReadAsArray()  # 以array形式读取

class NDVI_img:
    NDVI_ds = None  # 影像数据
    NDVI_array = None  # 以array形式读取

    def __init__(self, former_path, latter_path):  # 两个参数分别代表打开影像的前后两个path

        img_path = former_path + '\\' + latter_path
        self.NDVI_ds = gdal.Open(img_path)
        self.NDVI_array = self.NDVI_ds.ReadAsArray()  # 以array形式读取


# 得到拟合中距春分的天数
def get_d(a):
    # a = 'LC08_123032_20160129_warp.tif'  #  file_all 即此处的a
    date = a[12:20]  #截取日期

    A= list(date) # 转化
    A.insert(4,'-')
    A.insert(7,'-')

    d1 = ''.join(A) # 转化回来
    d2 = '2016-03-20'

    date1 = dt.datetime.strptime(d1, "%Y-%m-%d").date()  ##datetime.date(2016, 1, 06)
    date2 = dt.datetime.strptime(d2, "%Y-%m-%d").date()  ##datetime.date(2016, 3, 20)

    Days = (date1 - date2).days# 计算两个日期date的天数差
    return Days

def get_d_ndvi(a):
    # a = '20160129.tif'  #  file_all 即此处的a
    d1 = a[:8]
    d1 = list(d1)
    d1.insert(4,'-')
    d1.insert(7,'-')
    d1 = ''.join(d1)
    d2 = '2016-03-20'

    date1 = dt.datetime.strptime(d1, "%Y-%m-%d").date()  ##datetime.date(2016, 1, 06)
    date2 = dt.datetime.strptime(d2, "%Y-%m-%d").date()  ##datetime.date(2016, 3, 20)

    Days = (date1 - date2).days# 计算两个日期date的天数差
    return Days

def get_d_sat(a):
    d1 = a[:8]
    d1 = list(d1)
    d1.insert(4, '-')
    d1.insert(7, '-')
    d1 = ''.join(d1)
    d2 = '2015-12-31'

    date1 = dt.datetime.strptime(d1, "%Y-%m-%d").date()  ##datetime.date(2016, 1, 06)
    date2 = dt.datetime.strptime(d2, "%Y-%m-%d").date()  ##datetime.date(2016, 3, 20)

    Days = (date1 - date2).days# 计算两个日期date的天数差
    return Days


# 定义拟合函数以及残差计算函数

# ATCE模型
def fitfunc3(p, x, T_NDVI):  # 定义拟合函数
    p0, p1, p2,Lambda= p  # 拟合函数的参数，p0 p1 p2 p3分别代表 T0 A θ λ
    # Lambda = 0.1
    y = p0 + p1 * np.sin(x * np.pi * 2 / 366 + p2) + T_NDVI * Lambda  # 拟合函数的表达式
    return y

def error3(p, x, T_NDVI, y):  # 定义残差函数

    err = fitfunc3(p, x, T_NDVI) - y  # 残差
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

# 创建原始LST影像对象
for file_i_new in file_all_new:
    if file_i_new.endswith('.tif'):

        file_i_1 = 'LE07_127033_'
        file_i_2 = 'LC08_127033_'
        file_i_3 = 'LC08_123034_'

        if os.path.exists(input_file+"\\"+file_i_1+file_i_new):  # 判断是LE07还是LC08，以此按顺序打开影像
            img_collection['img' + str(file_i_new)] = IMG(input_file, file_i_1+file_i_new)  # 批量创建影像对象
        else :
            if os.path.exists(input_file+"\\"+file_i_2+file_i_new):
                img_collection['img' + str(file_i_new)] = IMG(input_file, file_i_2+file_i_new)
            else:
                img_collection['img' + str(file_i_new)] = IMG(input_file, file_i_3 + file_i_new)

        print(int(img_collection['img' + str(file_i_new)].date)+80)

print('Img collection have been created!')

# 创建SAT影像对象
input_SAT_file = 'E:\\MODIS_data\\2016\\img\\yulin'
file_SAT_all = os.listdir(input_SAT_file)
SAT_img_collection = {}
for file_SAT_new in file_SAT_all:
    if file_SAT_new.endswith('.tif'):

        sat_key = file_SAT_new[15:18]

        if os.path.exists(input_SAT_file + "\\" + file_SAT_new):
            SAT_img_collection['img' + str(int(sat_key))] = SAT_img(input_SAT_file, file_SAT_new)  # 批量创建影像对象

print('SAT Img collection have been created!')


# 创建SAT_ATC影像对象
input_SAT_ATC_file = "E:\\ MODIS_data_process \\typical_ATC"  # 仅仅是用来做拟合的部分SAT_ATC影像
file_SAT_ATC_all = os.listdir(input_SAT_ATC_file)
SAT_ATC_img_collection = {}
for file_SAT_ATC_new in file_SAT_ATC_all:
    if file_SAT_ATC_new.endswith('.tif'):

        SAT_ATC_key = int(file_SAT_ATC_new[0:3])

        if os.path.exists(input_SAT_ATC_file + "\\" + file_SAT_ATC_new):
            SAT_ATC_img_collection['img' + str(int(SAT_ATC_key))] = SAT_ATC_img(input_SAT_ATC_file, file_SAT_ATC_new)  # 批量创建影像对象
print('SAT_ATC Img collection have been created!')


# 创建NDVI影像对象
input_NDVI_file = "E:\\method2_data\\yulin\\ndvi"  # 仅仅是用来做拟合的部分NDVI影像
file_NDVI_all = os.listdir(input_NDVI_file)
NDVI_img_collection = {}
print(file_NDVI_all)
for file_NDVI_new in file_NDVI_all:
    if file_NDVI_new.endswith('.tif'):

        if os.path.exists(input_NDVI_file + "\\" + file_NDVI_new):
            NDVI_img_collection['img' + str(file_NDVI_new)] = NDVI_img(input_NDVI_file, file_NDVI_new)  # 批量创建影像对象

print('NDVI Img collection have been created!')
print(NDVI_img_collection)


# 预设存储数据的数组
num = 500
index = np.zeros(num)
rmse_error = np.zeros((num,num))
Mast = np.zeros((num,num))
Yast = np.zeros((num,num))
Theta = np.zeros((num,num))
Lambda = np.zeros((num,num))

# 研究区的植被物候学因子
NDVI_max = 6065   #修改/////////weishan:6383///////////yulin:6065/////////////beijing:6124/////////////////////
NDVI_min = 0


pFit = [0,0,0,0]
p = [290, 15, 170, 0.5]

list_x = []  # 创建两个空列表存储日期和像素值
list_y = []
list_SAT = []
list_SAT_ATC = []
list_delta_Tair_d = []
list_ndvi = []
core_y_in_2016 = []
for i_col in range(0,500):
    for j_row in range(0,500):
        for file_i_new in file_all_new:

            d = img_collection['img' + str(file_i_new)].date
            list_x.append(d)

            pixel_now = img_collection['img' + str(file_i_new)].vt_array[i_col][j_row]  # 取当前位置上的像素值
            # print(pixel_min)

            list_y.append(pixel_now)

            # 类气温数据的读取
            ############################## 需要修改 key ##############################
            file_sat_key = file_i_new[:8]
            file_sat_key =get_d_sat(file_sat_key)
            Tair_now = SAT_img_collection['img' + str(file_sat_key)].SAT_array[i_col][j_row]  #得到sat数组
            list_SAT.append(Tair_now)

            # # SAT_ATC的读取
            Tair_ATC_now = SAT_ATC_img_collection['img' + str(file_sat_key)].SAT_ATC_array[i_col][j_row]  # 得到sat_ATC数组
            list_SAT_ATC.append(Tair_ATC_now)

            # 求解delta-Tair
            delta_Tair_d = SAT_img_collection['img' + str(file_sat_key)].SAT_array[i_col][j_row] - SAT_ATC_img_collection['img' + str(file_sat_key)].SAT_ATC_array[i_col][j_row]
            list_delta_Tair_d.append(delta_Tair_d)


            # 植被指数的读取
            file_ndvi_key = file_i_new[:8]
            file_ndvi_key = file_ndvi_key+'.tif'
            ndvi_now = NDVI_img_collection['img' + str(file_ndvi_key)].NDVI_array[i_col][j_row]
            ndvi_now = (NDVI_max - NDVI_min) * 0.0001 / ((ndvi_now - NDVI_min) * 0.0001 + 1.0)
            list_ndvi.append(ndvi_now)

        print('-----------It is turn to [',i_col,',',j_row,']-----------' )

        list_y = [elem_y if not np.isnan(elem_y) else None for elem_y in list_y]  #  查询list_y里是否存在None值
        if None in list_y:  #有空值那么直接将数值赋为0

            pos = [i for i, x in enumerate(list_y) if x == None]  # 获取None值的索引

            list_x = [n for x_i, n in enumerate(list_x) if x_i not in pos]  # 清除y_Obs里的空值对应的天数
            list_delta_Tair_d = [n for sat_i, n in enumerate(list_delta_Tair_d) if sat_i not in pos]
            list_ndvi = [n for ndvi_i, n in enumerate(list_ndvi) if ndvi_i not in pos]

            list_y = list(filter(None, list_y))  # 清除y_Obs里的空值

            list_a = []  # 置零重新参与计算

        if len(list_x) < 4:
            print('数据过少，无法进行拟合！')
            pFit = [0,0,0,0]

            Mast[i_col][j_row] = pFit[0]
            Yast[i_col][j_row] = pFit[1]
            Theta[i_col][j_row] = pFit[2]
            Lambda[i_col][j_row] = pFit[3]
            print(pFit[0],pFit[1],pFit[2],pFit[3])

            list_x = []
            list_y = []
            list_delta_Tair_d = []
            list_ndvi = []
            list_SAT = []
            list_SAT_ATC = []
            continue;  #小于参数个数，无法进行拟合

        x = np.array(list_x)  # 日期d
        y_Obs = np.array(list_y)  # LST_Obs
        sat = np.array(list_SAT)
        sat_atc = np.array(list_SAT_ATC)
        T_air = np.array(list_delta_Tair_d)
        NDVI = np.array(list_ndvi)
        T_NDVI = np.multiply(T_air, NDVI)
        length = len(y_Obs)
        # print('x',x)
        # print(y_Obs)
        print(T_air)
        print(NDVI)

        # 由给定数据点集 (x,y) 求拟合函数的参数 pFit
        p0 = [290, 15, 170, 0.2]  # 设置拟合函数的参数初值
        element_exist = np.nan in y_Obs  # 判断LST数组里是否存在前面设置的零值

        if element_exist  ==True:
            p0=[0,0,0,0]  # 存在的话，暂时将三参数赋值为零值，后面再进行插值
        else:
            pFit, info = leastsq(error3, p0, args=(x, T_NDVI,y_Obs))  # 最小二乘法求拟合参数
            # pFit, info = leastsq(error3, p0, args=(x, y_Obs))  # 最小二乘法求拟合参数

        print(pFit[0])
        print(pFit[1])
        print(pFit[2])
        print(pFit[3])

        print('-----------  Fitting has finished!  -------------')
        print('                                  ')

        Mast[i_col][j_row] = pFit[0]
        Yast[i_col][j_row] = pFit[1]
        Theta[i_col][j_row] = pFit[2]
        Lambda[i_col][j_row] = pFit[3]


        # # # # # # # # # # # # # # # 计算各点残差# # # # # # # # # # # # # # #
        sum_error = 0
        for j in range(0, length):
            error = (fitfunc3(pFit, x[j], T_NDVI[j]) - y_Obs[j]) * (fitfunc3(pFit, x[j], T_NDVI[j]) - y_Obs[j])
            sum_error = sum_error + error

        #修改为二维行列的矩阵
        rmse_error[i_col][j_row] = sqrt(sum_error / length)  #计算单点上的误差矩阵
        p[3] = p0[3]
        p[2] = p0[2] + 0.2
        p[0] = p0[0]
        p[1] = p0[1]

        list_x = []  # 置零重新参与计算
        list_y = []
        list_delta_Tair_d = []
        list_ndvi = []
        list_SAT = []
        list_SAT_ATC = []

print('All finished!')

#  存储到xls文件中
array1 = np.array(Mast)
df1 = pd.DataFrame(array1)
df1.to_csv('E:\\EATC\\yulin_1.csv')

array2 = np.array(Yast)
df2 = pd.DataFrame(array2)
df2.to_csv(' E:\\EATC\\yulin_2.csv ')

array3 = np.array(Theta)
df3 = pd.DataFrame(array3)
df3.to_csv(' E:\\EATC\\yulin_3.csv ')

array4 = np.array(Lambda)
df4 = pd.DataFrame(array4)
df4.to_csv(' E:\\EATC\\yulin_4.csv ')


print('Saving File Over!')
