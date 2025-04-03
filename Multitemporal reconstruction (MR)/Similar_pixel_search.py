# Author : GJL
# CreateTime : 2023/7/3
# FileName : Similar_pixel_serach
# Description : function
from osgeo import gdal
import numpy as np
from datetime import datetime

# 每个研究区的云污染影像LST的重建函数
def LST_calculate(file_LST, array1, array2, array3, min_window_size, max_window_size, expected_similar_pixels):

    array1 = np.nan_to_num(array1)  # 将NAN转化为0

    num = 500
    recon_array = np.zeros((num, num))
    # recon_array_path = gdal.Open('G:\\ATC_fusion_result\\yulin\\20160905_LST.tif')
    # recon_array = recon_array_path.ReadAsArray()

    # 遍历每个像元
    for i in range(0, 500):
        for j in range(0, 500):
            if array1[i][j] != 0:
                recon_array[i][j] =  array1[i][j]  # 当前像元有LST的话，不作重建，直接储存至 recon_array 对应位置
            else:
                # 假设为最大窗口，获取窗口中的数据，如果全为0，那么不必进行相似像元的搜索
                window_size = max_window_size

                row_start = max(0, i - window_size // 2)
                row_end = min(array1.shape[0], i + window_size // 2 + 1)
                col_start = max(0, j - window_size // 2)
                col_end = min(array1.shape[1], j + window_size // 2 + 1)

                window1 = array1[row_start:row_end, col_start:col_end]

                if np.all(window1 == 0):
                    recon_array[i][j] = 0
                else:
                    recon_array[i][j] = similar_pixel_search(i, j, array1, array2, array3, min_window_size, max_window_size, expected_similar_pixels)

                print('24行   ',file_LST,  i, j, recon_array[i][j])

    # i = 481  #452
    # j = 458  #476
    # recon_array[i][j] = similar_pixel_search(i, j, array1, array2, array3, min_window_size, max_window_size, expected_similar_pixels)
    # print(i, j, recon_array[i][j])

    return recon_array

# 搜索相似像元
def similar_pixel_search(i_old, j_old, array1, array2, array3, min_window_size, max_window_size, expected_similar_pixels):

    # window_size = min(i, j, 500 - i - 1, 500 - j - 1) * 2 + 1
    window_size = min_window_size

    while window_size <= max_window_size:

        i = i_old
        j = j_old

        # 计算窗口的边界
        row_start = max(0, i - window_size // 2)
        row_end = min(array1.shape[0], i + window_size // 2 + 1)
        col_start = max(0, j - window_size // 2)
        col_end = min(array1.shape[1], j + window_size // 2 + 1)

        # 获取窗口中的数据
        window1 = array1[row_start:row_end, col_start:col_end]
        window2 = array2[row_start:row_end, col_start:col_end]
        window3 = array3[row_start:row_end, col_start:col_end]

        # 获取现有窗口的行宽列宽
        window_row = len(window1)        # a = window1.shape[0]
        window_col = len(window1[0])        #rows, cols = arr.shape

        # 获取目标像元在现有窗口中的位置
        half_size = window_size // 2
        i= i - max(0, i - half_size)
        j= j - max(0, j - half_size)

        # 搜索窗口中的相似像元，并计算相似像素数量
        # 计算自适应阈值
        ref_center_pixel = window2[i, j]
        ndvi_center_pixel = window3[i, j]
        ref_diff = 0
        ndvi_diff = 0
        common_num = 0  # 公共像元个数

        common_pixel_mask = np.where(window1 != 0, 1, window1)
        window2 = window2 * common_pixel_mask

        # 初始化相似像素数量
        similar_pixel_count = 0
        similar_pixel_mask = np.zeros((window_row, window_col))  # 自定义掩膜mask
        for i_col in range(0, window_row):
            for j_row in range(0, window_col):
                if window2[i_col][j_row] == 0:
                    continue;
                else:
                    ref_diff = ref_diff + (np.abs(window2[i_col][j_row] - ref_center_pixel)) ** 2
                    ndvi_diff = ndvi_diff + (np.abs(window3[i_col][j_row] - ndvi_center_pixel)) ** 2
                    common_num = common_num + 1

        #当公共像元个数为0时，也就不必要搜索相似像元
        if common_num == 0:
            T_threshold = 0
            V_threshold = 0
        else:
            T_threshold = np.sqrt(ref_diff / common_num)
            V_threshold = np.sqrt(ndvi_diff / common_num)

        for i_col in range(0, window_row):
            for j_row in range(0, window_col):
                if window2[i_col][j_row] == 0:
                    continue;
                else:
                    # 计算相似像素数量
                    similar_pixels = np.abs(window2[i_col][j_row] - ref_center_pixel) <= T_threshold and np.abs(window3[i_col][j_row] - ndvi_center_pixel) <= V_threshold
                    if similar_pixels == True:
                        similar_pixel_count = similar_pixel_count + 1
                        similar_pixel_mask[i_col][j_row] = 1  # 该像元为相似像元时，掩膜上该位置赋值为1，否则默认为0

        if similar_pixel_count >= expected_similar_pixels:
            break;

        if similar_pixel_count < expected_similar_pixels:
            # 增加窗口大小
            window_size += 2


    #已达到窗口最大阈值，但还没有找够期望个数的相似像元
    if window_size >= max_window_size and similar_pixel_count < expected_similar_pixels:

        i = i_old
        j = j_old

        window_size = max_window_size
        # 计算窗口的边界
        row_start = max(0, i - window_size // 2)
        row_end = min(array1.shape[0], i + window_size // 2 + 1)
        col_start = max(0, j - window_size // 2)
        col_end = min(array1.shape[1], j + window_size // 2 + 1)

        # 获取窗口中的数据
        window1 = array1[row_start:row_end, col_start:col_end]
        window2 = array2[row_start:row_end, col_start:col_end]
        window3 = array3[row_start:row_end, col_start:col_end]

        # 获取现有窗口的行宽列宽
        window_row = len(window1)
        window_col = len(window1[0])

        # 获取目标像元在现有窗口中的位置
        half_size = window_size // 2
        i = i - max(0, i - half_size)
        j = j - max(0, j - half_size)

        # 搜索窗口中的相似像元，并计算相似像素数量
        # 计算自适应阈值
        ref_center_pixel = window2[i, j]
        ndvi_center_pixel = window3[i, j]

        T_threshold = 0.05
        V_threshold = 0.05 #====================================================更改阈值=================================================================

        similar_pixel_count = 0
        similar_pixel_mask = np.zeros((window_row, window_col))
        common_pixel_mask = np.where(window1 != 0, 1, window1)
        window2 = window2 * common_pixel_mask

        for i_col in range(0,window_row):
            for j_row in range(0,window_col):

                if window2[i_col][j_row]==0:
                    continue;
                else:
                    # 计算相似像素数量
                    similar_pixels = np.abs(window2[i_col][j_row] - ref_center_pixel) <= T_threshold and np.abs(window3[i_col][j_row] - ndvi_center_pixel) <= V_threshold
                    if similar_pixels == True:
                        similar_pixel_count = similar_pixel_count + 1
                        similar_pixel_mask[i_col][j_row] = 1  #该像元为相似像元时，掩膜上该位置赋值为1，否则默认为0

        if similar_pixel_count == 0:
            LST_reconstruct = 0
            return LST_reconstruct

    final_window_size = window_size
    final_similar_pixel_count = similar_pixel_count
    final_similar_pixel_mask = similar_pixel_mask

    LST_reconstruct = a_b_calculate(i, j, window1, window2, window3,  ref_center_pixel, ndvi_center_pixel, final_window_size, final_similar_pixel_count, final_similar_pixel_mask)

    return LST_reconstruct


# 检查相似像素数量，分情况计算线性模型里的 a 和 b
def a_b_calculate(i, j, window1, window2, window3, ref_center_pixel, ndvi_center_pixel, window_size, similar_pixel_count, similar_pixel_mask):

    window_row = len(window1)
    window_col = len(window1[0])

    # 处理相似像素
    ref_center_pixel2 = ref_center_pixel
    ndvi_center_pixel2 = ndvi_center_pixel

    kexi = 0.001

    window1 = window1 * similar_pixel_mask
    window2 = window2 * similar_pixel_mask
    window3 = window3 * similar_pixel_mask
    similar_pixel_D_mask = np.zeros((window_row, window_col))
    # 权重窗口 Di 的计算
    for m in range(window_row):
        for n in range(window_col):
            if window2[m][n] == 0:
                continue;
            else:
                similar_pixel_D_mask[m][n] = np.abs(ref_center_pixel2 - window2[m][n] + kexi) * np.abs(ndvi_center_pixel2 - window3[m][n] + kexi) * ( pow( (m - window_size // 2 ), 2) + pow( (n - window_size // 2), 2))
                # if m == 38 and n == 21:
                #     print(window2[m][n] ,window3[m][n],ref_center_pixel2, ndvi_center_pixel2, ( pow( (m - window_size // 2 ), 2) + pow( (n - window_size // 2), 2)))
                #     print(np.abs(ref_center_pixel2 - window2[m][n] + kexi), np.abs(ndvi_center_pixel2 - window3[m][n] + kexi), ( pow( (m - window_size // 2 ), 2) + pow( (n - window_size // 2), 2)))

    # result = np.unravel_index(np.argmin(similar_pixel_D_mask), similar_pixel_D_mask.shape)

    # non_zero_arr = similar_pixel_D_mask[similar_pixel_D_mask != 0]
    # min_val = np.min(non_zero_arr)
    # result = np.where(similar_pixel_D_mask == min_val)
    # similar_pixel_D_mask[result[0][0]][result[1][0]] = 0
    # print(result[0][0], result[1][0])

    non_zero_indices =  similar_pixel_D_mask != 0
    similar_pixel_D_mask[non_zero_indices] = 1 / similar_pixel_D_mask[non_zero_indices]  # 计算每一个非0元素的倒数

    # 筛选去除 D_mask 里的异常值
    result = np.where(similar_pixel_D_mask > 100)
    similar_pixel_D_mask[result] = 0
    window1[result] = 0
    window2[result] = 0
    window3[result] = 0

    # 归一化权重
    similar_pixel_W_mask = similar_pixel_D_mask / np.sum(similar_pixel_D_mask)

    # window1[result[0][0]][result[1][0]] = 0
    # window2[result[0][0]][result[1][0]] = 0
    # window3[result[0][0]][result[1][0]] = 0

    # 计算 斜率a 和 截距b
    similar_pixel_count = similar_pixel_count - len(result[0])
    T_s_mean_cloud = sum(sum(window1)) / similar_pixel_count
    T_s_mean_ref = sum(sum(window2)) / similar_pixel_count

    a = 0
    b = 0
    # if similar_pixel_count >= expected_similar_pixels:
    if similar_pixel_count >= 3:
        a_up = 0
        a_down = 0
        for m in range(window_row):
            for n in range(window_col):
                if window2[m][n] == 0:
                    continue;
                else:
                    a_up = a_up + similar_pixel_W_mask[m][n] * (window1[m][n] - T_s_mean_cloud) * (window2[m][n] - T_s_mean_ref)
                    a_down = a_down + similar_pixel_W_mask[m][n] * (window2[m][n] - T_s_mean_ref) * (window2[m][n] - T_s_mean_ref)

            # print(m, n, a_up, a_down)

        if a_up == 0 or a_down == 0:
            a = 0
        else:
            a = a_up / a_down
        b = T_s_mean_cloud - a * T_s_mean_ref

    if similar_pixel_count < 3:
       a = T_s_mean_cloud / T_s_mean_ref
       b = 0

    LST = a * ref_center_pixel2 + b

    return LST



# 查找时间最近的晴空影像的日期
def find_closest_date(date_cloud, clear_date_all):

    date_cloud = date_cloud[:8]
    date = datetime.strptime(date_cloud, '%Y%m%d')
    dates = [datetime.strptime(d, '%Y%m%d') for d in clear_date_all]
    closest_date = min(dates, key=lambda d: abs(d - date))
    return closest_date.strftime('%Y%m%d')

# 将数组转为栅格，设置LST影像的地理变换和坐标系，保存LST重建结果
def arr2raster(ref_dataset_file, cloud_file, result_LST_collection,  result_save_path):

    ref_dataset = gdal.Open(ref_dataset_file)
    for date_i in cloud_file:

        date_i = date_i[:8]
        my_LST_array = np.array(result_LST_collection['img' + str(date_i)].Result_LST_array)
        raster_file = result_save_path + '\\' + str(date_i) + '_LST.tif'  #=============================????????

        # 获取参考影像的地理变换
        geo_transform = ref_dataset.GetGeoTransform()
        # 获取参考影像的坐标参考系
        projection = ref_dataset.GetProjection()

        # 创建一个新的 GeoTIFF 数据集
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(raster_file, my_LST_array.shape[1], my_LST_array.shape[0], 1, gdal.GDT_Float32)

        # 设置地理变换为参考影像的地理变换
        dataset.SetGeoTransform(geo_transform)

        # 设置坐标参考系为参考影像的坐标参考系
        dataset.SetProjection(projection)

        # 将数组写入数据集
        dataset.GetRasterBand(1).WriteArray(my_LST_array)

        # 关闭数据集
        dataset = None
        del dataset



def single_arr2raster(ref_dataset_file, img_DOY, array,  result_save_path):

    ref_dataset = gdal.Open(ref_dataset_file)

    my_LST_array = np.array(array)
    raster_file = result_save_path + '\\' + str(img_DOY) + '.tif'

    # 获取参考影像的地理变换
    geo_transform = ref_dataset.GetGeoTransform()
    # 获取参考影像的坐标参考系
    projection = ref_dataset.GetProjection()

    # 创建一个新的 GeoTIFF 数据集
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(raster_file, my_LST_array.shape[1], my_LST_array.shape[0], 1, gdal.GDT_Float32)

    # 设置地理变换为参考影像的地理变换
    dataset.SetGeoTransform(geo_transform)

    # 设置坐标参考系为参考影像的坐标参考系
    dataset.SetProjection(projection)

    # 将数组写入数据集
    dataset.GetRasterBand(1).WriteArray(my_LST_array)

    # 关闭数据集
    dataset = None
    del dataset