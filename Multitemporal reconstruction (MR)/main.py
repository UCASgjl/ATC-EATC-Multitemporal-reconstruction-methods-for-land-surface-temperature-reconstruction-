# Author : GJL
# CreateTime : 2023/7/1
# FileName : main
# Description : MR method
from img_class import *
from Similar_pixel_search import *

# ======================================================= 创建影像对象 =======================================================


input_cloud_file = "E:\\cloudy"
input_clear_file = "E:\\clear"
input_ndvi_file = "E:\\NDVI"
ref_dataset_file = "E:\\clear\\LC08_123035_20161128_warp.tif"  #保存LST-array结果时参考的影像
result_LST_file = "E:\\result"  #结果保存路径


import os

def check_file_paths(file_paths):
    """
    检查给定的文件路径是否都存在。

    参数:
    file_paths (list of str): 文件路径的列表。

    返回:
    bool: 如果所有路径都存在则返回 True，否则返回 False。
    list of str: 不存在的文件路径列表。
    """
    non_existent_paths = []
    all_exist = True

    for path in file_paths:
        if not os.path.exists(path):
            non_existent_paths.append(path)
            all_exist = False

    return all_exist, non_existent_paths

# 示例文件路径列表
file_paths = [
    input_cloud_file,
    input_clear_file,
    input_ndvi_file,
    ref_dataset_file,# 保存LST-array结果时参考的影像
    result_LST_file# 结果保存路径
]

# 检查文件路径是否存在
all_exist, non_existent_paths = check_file_paths(file_paths)

if all_exist:
    print("所有文件路径都存在。")
else:
    print(f"以下文件路径不存在: {', '.join(non_existent_paths)}")



# 1、云污染影像
cloud_file_all = os.listdir(input_cloud_file)
cloud_img_collection = {}

cloud_file_all_new = list()
for file_i in cloud_file_all:
    the_date = file_i[12:]  # 截取日期
    cloud_file_all_new.append(the_date)  # 该列表中每一个元素，形如“20160301_cloud.tif”

cloud_file_all_new.sort()  #  对影像按照时间重新排序
print(cloud_file_all_new)


# 创建原始云污染LST影像对象
for file_i_new in cloud_file_all_new:
    if file_i_new.endswith('.tif'):

        # LE07_127033_20161218_warp
        file_i_1 = 'LC08_127033_'
        file_i_2 = 'LC08_123032_'
        file_i_3 = 'LE07_123035_'

        if os.path.exists(input_cloud_file+"\\"+file_i_1+file_i_new):  # 判断是LE07还是LC08，以此按顺序打开影像
            cloud_img_collection['img' + str(file_i_new)] = Cloudy_IMG(input_cloud_file, file_i_1+file_i_new)  # 批量创建影像对象
        else :
            if os.path.exists(input_cloud_file+"\\"+file_i_2+file_i_new):
                cloud_img_collection['img' + str(file_i_new)] = Cloudy_IMG(input_cloud_file, file_i_2+file_i_new)
            else:
                cloud_img_collection['img' + str(file_i_new)] = Cloudy_IMG(input_cloud_file, file_i_3 + file_i_new)

        # print(cloud_img_collection['img' + str(file_i_new)].date)

print('cloud Img collection have been created!')

# 2、晴空影像
clear_file_all = os.listdir(input_clear_file)
clear_img_collection = {}

clear_file_all_new = list()
for file_i in clear_file_all:
    the_date = file_i[12:]  # 截取日期
    clear_file_all_new.append(the_date)

clear_file_all_new.sort()  #  对影像按照时间重新排序


# 创建原始晴空LST影像对象
for file_i_new in clear_file_all_new:
    if file_i_new.endswith('.tif'):

        # LE07_127033_20161218_warp
        file_i_1 = 'LC08_127033_'
        file_i_2 = 'LE07_127033_'
        file_i_3 = 'LE07_123032_'
        file_i_4 = 'LC08_123035_'
        file_i_5 = 'LE07_123035_'

        if os.path.exists(input_clear_file+"\\"+file_i_1+file_i_new):  # 判断是LE07还是LC08，以此按顺序打开影像
            clear_img_collection['img' + str(file_i_new)] = Clear_IMG(input_clear_file, file_i_1+file_i_new)  # 批量创建影像对象
        if os.path.exists(input_clear_file+"\\"+file_i_2+file_i_new):
            clear_img_collection['img' + str(file_i_new)] = Clear_IMG(input_clear_file, file_i_2+file_i_new)
        if os.path.exists(input_clear_file+"\\"+file_i_3+file_i_new):
            clear_img_collection['img' + str(file_i_new)] = Clear_IMG(input_clear_file, file_i_3 + file_i_new)
        if os.path.exists(input_clear_file + "\\" + file_i_4 + file_i_new):
            clear_img_collection['img' + str(file_i_new)] = Clear_IMG(input_clear_file, file_i_4 + file_i_new)
        if os.path.exists(input_clear_file + "\\" + file_i_5 + file_i_new):
            clear_img_collection['img' + str(file_i_new)] = Clear_IMG(input_clear_file, file_i_5 + file_i_new)

        # print(clear_img_collection['img' + str(file_i_new)].date)

print('clear Img collection have been created!')

# 3、NDVI影像
file_NDVI_all = os.listdir(input_ndvi_file)  # " 20160101.tif"
NDVI_img_collection = {}

# 创建原始NDVI影像对象
for file_NDVI_new in file_NDVI_all:
    if file_NDVI_new.endswith('.tif'):

        if os.path.exists(input_ndvi_file + "\\" + file_NDVI_new):
            NDVI_img_collection['img' + str(file_NDVI_new)] = NDVI_IMG(input_ndvi_file, file_NDVI_new)  # 批量创建影像对象

        # print(NDVI_img_collection['img' + str(file_NDVI_new)].date)
print('NDVI Img collection have been created!')
# 4、重建LST结果
Result_img_collection = {}
#创建重建后的LST影像对象集合
for date_i in cloud_file_all_new:

    d1 = date_i[:8]

    # print('正在创建',date_i,'的影像对象')
    Result_img_collection['img' + str(d1)] = Result_LST_IMG(str(d1))  # 批量创建LST重建后的影像对象

print('Result_img_collection have been created!')




# ===============================================云污染影像LST的重建=================================================
# 定义最小和最大窗口大小
Min_window_size = 31
Max_window_size = 150

# 定义期望的相似像素数量
Expected_similar_pixels = 100

# 逐个云污染影像进行LST重建
for file_i_cloud in cloud_file_all_new:
    if file_i_cloud.endswith('.tif'):

        '''
        LST_calculate() 为 Similar_pixel_serach.py 里的函数
        '''

        # 云污染LST
        Cloud_array = cloud_img_collection['img' + str(file_i_cloud)].vt_array

        # 晴空LST
        clear_date = list()
        for file_i in clear_file_all_new:
            the_date = file_i[:8]  # 截取日期
            clear_date.append(the_date)

        file_i_clear = find_closest_date(file_i_cloud, clear_date)  # 搜索时间上最近的晴空影像的日期
        print("207",file_i_clear)
        Clear_array = clear_img_collection['img' + str(file_i_clear) + '_warp.tif'].vt_array

        # NDVI影像
        file_i_ndvi = file_i_cloud[:8]
        NDVI_array = NDVI_img_collection['img' + str(file_i_ndvi) + '.tif'].ndvi_array  * 0.0001

        # 计算 LST
        file_i_LST = file_i_cloud[:8]
        Result_img_collection['img' + str(file_i_LST)].Result_LST_array= LST_calculate(file_i_LST, Cloud_array, Clear_array, NDVI_array, Min_window_size, Max_window_size, Expected_similar_pixels)

# 保存LST结果
arr2raster(ref_dataset_file, cloud_file_all_new, Result_img_collection,  result_LST_file)