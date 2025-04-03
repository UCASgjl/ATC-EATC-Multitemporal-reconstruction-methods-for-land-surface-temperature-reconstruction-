# Author : GJL
# CreateTime : 2023/8/4
# FileName : img_class
# Description : definition for Landsat clear images,cloudy images and NDVI images
import datetime as dt
from osgeo import gdal,ogr,osr
import numpy as np
import re
from math import sqrt
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square

# 原始云污染影像
class Cloudy_IMG:
    DOY = None
    path = None
    ds = None  #影像数据
    vt_array = None  #以array形式读取

    def __init__(self, former_path, latter_path):  #两个参数分别代表打开影像的前后两个path
        self.value = latter_path
        self.DOY = get_d(latter_path)

        img_path = former_path + '\\' + latter_path
        self.ds = gdal.Open(img_path)
        self.vt_array = self.ds.ReadAsArray()  #以array形式读取


# 原始晴空影像
class Clear_IMG:
    DOY = None
    path = None
    ds = None  #影像数据
    vt_array = None  #以array形式读取

    def __init__(self, former_path, latter_path):  #两个参数分别代表打开影像的前后两个path
        self.value = latter_path
        self.DOY = get_d(latter_path)

        img_path = former_path + '\\' + latter_path
        self.ds = gdal.Open(img_path)
        self.vt_array = self.ds.ReadAsArray()  #以array形式读取


# 原始晴空影像
class ATC_IMG:
    DOY = None
    path = None
    ds = None  #影像数据
    vt_array = None  #以array形式读取

    def __init__(self, former_path, latter_path):  #两个参数分别代表打开影像的前后两个path
        self.value = latter_path
        # self.DOY = get_d(latter_path)

        img_path = former_path + '\\' + latter_path
        self.ds = gdal.Open(img_path)
        self.vt_array = self.ds.ReadAsArray()  #以array形式读取



def get_d(a):
    # a = 'LC08_123032_20160129_warp.tif'  #  file_all 即此处的a
    date = a[12:20]  # 截取日期

    A = list(date)  # 转化
    A.insert(4, '-')
    A.insert(7, '-')

    d1 = ''.join(A)  # 转化回来
    d2 = '2015-12-31'

    date1 = dt.datetime.strptime(d1, "%Y-%m-%d").date()  ##datetime.date  e.g.(2016, 1, 06)
    date2 = dt.datetime.strptime(d2, "%Y-%m-%d").date()  ##datetime.date(2015, 12, 31)

    Days = (date1 - date2).days  # 计算两个日期date的天数差
    return Days


# NDVI影像
class NDVI_IMG:

    NDVI_DOY = None
    NDVI_ds = None  # 影像数据
    ndvi_array = None  # 以array形式读取

    def __init__(self, former_path, latter_path):  # 两个参数分别代表打开影像的前后两个path

        self.NDVI_DOY = get_d_ndvi(latter_path)

        img_path = former_path + '\\' + latter_path
        self.NDVI_ds = gdal.Open(img_path)
        self.ndvi_array = self.NDVI_ds.ReadAsArray()  # 以array形式读取

def get_d_ndvi(a):
    # a = '20160129.tif'  #  file_all 即此处的a

    d1 = a[:8]
    d1 = list(d1)
    d1.insert(4,'-')
    d1.insert(7,'-')
    d1 = ''.join(d1)
    d2 = '2015-12-31'

    date1 = dt.datetime.strptime(d1, "%Y-%m-%d").date()  ##datetime.date(2016, 1, 06)
    date2 = dt.datetime.strptime(d2, "%Y-%m-%d").date()  ##datetime.date(2015, 12, 31)

    Days = (date1 - date2).days # 计算两个日期date的天数差
    return Days


# 重建LST影像
class Result_LST_IMG:

    Result_LST_date = None  # 影像数据
    Result_LST_array = None  # 以array形式读取

    def __init__(self, LST_date):  # 两个参数分别代表打开影像的前后两个path

        self.Result_LST_date = LST_date   # "20160301"，为字符串
        self.Result_LST_array = np.zeros((500, 500))  # 以array形式读取

class result_IMG:
    result_DOY = None  # 影像数据
    result_ds = None
    result_array = None  # 以array形式读取

    def __init__(self, former_path, latter_path):  # 两个参数分别代表打开影像的前后两个path

        result_key = re.sub('\D', '', latter_path)
        self.result_DOY = get_d_ndvi(result_key)

        img_path = former_path + '\\' + latter_path
        self.result_ds = gdal.Open(img_path)
        self.result_array = self.result_ds.ReadAsArray()  # 以array形式读取

# 精度验证
def accuracy_verification(cloudy,fitted):

    four_index = dict();  # 创建保存四个评价指标的字典

    list_Fit_img = np.array(fitted).flatten().tolist()
    list_Cloudy_img = cloudy.vt_array.flatten().tolist()

    print(len(list_Fit_img),len(list_Cloudy_img))

    # list_Cloudy_img = [elem_y if not np.isnan(elem_y) else None for elem_y in list_Cloudy_img]  # 查询list_y里是否存在None值
    # if None in list_Cloudy_img:  # 有空值那么直接将数值赋为0
    #     pos = [i for i, x in enumerate(list_Cloudy_img) if x == None]  # 获取None值的索引
    #     list_Fit_img = [n for x_i, n in enumerate(list_Fit_img) if x_i not in pos]  # 清除y_Fit里的值
    #     list_Cloudy_img = list(filter(None, list_Cloudy_img))  # 清除y_Obs里的空值
    #
    # print(len(list_Fit_img), len(list_Cloudy_img))
    #
    # list_Fit_img = [elem_y if not np.isnan(elem_y) else None for elem_y in list_Fit_img]  # 查询list_y里是否存在None值
    # if None in list_Fit_img:  # 有空值那么直接将数值赋为0
    #     pos = [i for i, x in enumerate(list_Fit_img) if x == None]  # 获取None值的索引
    #     print(len(pos))
    #     list_Cloudy_img = [n for x_i, n in enumerate(list_Cloudy_img) if x_i not in pos]  # 清除y_Fit里的值
    #     print(len(list_Cloudy_img))
    #     list_Fit_img = list(filter(None, list_Fit_img))  # 清除y_Obs里的空值
    #
    # print(len(list_Fit_img),len(list_Cloudy_img))


    # 找出list_Fit_img中不是NaN的元素的索引
    valid_indices_Fit = np.where(~np.isnan(list_Fit_img))[0]

    # 找出list_Cloudy_img中不是NaN的元素的索引
    valid_indices_Cloudy = np.where(~np.isnan(list_Cloudy_img))[0]

    # 取交集，得到两个列表都不包含NaN的元素的索引
    valid_indices = np.intersect1d(valid_indices_Fit, valid_indices_Cloudy)

    # 使用索引获取新的不包含NaN的list_Fit_img和list_Cloudy_img
    list_Fit_img = [list_Fit_img[i] for i in valid_indices]
    list_Cloudy_img = [list_Cloudy_img[i] for i in valid_indices]

    print(len(list_Fit_img), len(list_Cloudy_img))

    list_a = []  # 置零重新参与计算

    sum = 0
    for index in range(0,len(list_Cloudy_img)):
        sum = sum + list_Cloudy_img[index] - list_Fit_img[index]


    # 调用
    four_index['ME_img'] = sum / len(list_Cloudy_img)
    MSE_img= mean_squared_error(list_Cloudy_img, list_Fit_img)
    four_index['MAE_img'] = mean_absolute_error(list_Cloudy_img, list_Fit_img)
    four_index['R_2_img'] = r2_score(list_Cloudy_img, list_Fit_img)
    four_index['RMSE_img'] = sqrt(MSE_img)

    print("四个指标计算结束！")

    return four_index;

