# encoding: utf-8

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# 定义显示图像的函数
def show_picture(src):
    """
    定义显示图片的函数
    :param src:
    :return:
    """
    cv2.namedWindow(winname='pic', flags=0)
    cv2.imshow(winname='pic', mat=src)
    cv2.waitKey()
    cv2.destroyAllWindows()


"""
我国车牌种类很多， 但一般都会有统一的格式

"""
index = 92
plate_path_base = r'E:\code\112\models\picture_saved'

image = cv2.imread(os.path.join(plate_path_base, str(index) + '.png'), cv2.COLOR_BGR2GRAY)
# show_picture(image)
ret, thresh = cv2.threshold(src=image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 进行腐蚀操作
show_picture(thresh)
kernel_for_erosion = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
erosion = cv2.erode(src=thresh, kernel=kernel_for_erosion, iterations=1)
show_picture(erosion)
# 获取车牌的尺寸大小
height = erosion.shape[0]
width = erosion.shape[1]
height_center = height // 2
width_center = width // 2
# 从图像宽度的中心向上查找和向下查找
# 向上
up_part = 0
down_part = 0
left_part = 0
right_part = 0
# height_data = []
# width_data = []

# 从中间向上面找
# print('height_center: ', height_center)
for height_index in range(height_center, 0, -1):
    # print(height_index)
    temp = erosion[height_index, :]
    white_point_num = int(sum(temp) / 255)
    # height_data.append(white_point_num)
    # print(white_point_num)
    if white_point_num < 10 or height_index < 3:
        up_part = height_index
        # print('---------------------')
        # print(white_point_num)
        # print(height_index)
        # print('---------------------')
        break

# 从中间向下面找
for height_index in range(height_center, height, 1):
    temp = erosion[height_index, :]
    white_point_num = int(sum(temp) / 255)
    if white_point_num < 10 or height_index > height - 3:
        down_part = height_index
        break

for width_index in range(0, int(width_center * (1 - 0.7)), 1):
    temp_1 = erosion[:, width_index]
    temp_2 = erosion[:, width_index + 1]
    temp_3 = erosion[:, width_index + 2]
    temp_4 = erosion[:, width_index + 3]
    white_point_num_1, white_point_num_2, white_point_num_3, white_point_num_4 =\
        sum(temp_1) / 255, sum(temp_2) / 255, sum(temp_3) / 255, sum(temp_4) / 255
    white_point_num_1 = int(white_point_num_1)
    white_point_num_2 = int(white_point_num_2)
    white_point_num_3 = int(white_point_num_3)
    white_point_num_4 = int(white_point_num_4)
    # if white_point_num_2 - white_point_num_1 > 10:
    #     left_part = width_index
    #     break
    if white_point_num_1 == 0:
        continue
    else:
        if white_point_num_2 == 0:
            if white_point_num_3 == 0 and white_point_num_4 == 0:
                continue
            elif white_point_num_3 != 0 and white_point_num_4 == 0:
                continue
            elif white_point_num_3 == 0 and white_point_num_4 !=0:
                left_part = width_index + 2
                break
            else:
                left_part = width_index + 1
                break

        else:  # p1!=0  p2!=0
            if white_point_num_3 == 0 and white_point_num_4 == 0:
                continue
            elif white_point_num_3 != 0 and white_point_num_4 == 0:
                continue
            elif white_point_num_3 == 0 and white_point_num_4 != 0:
                left_part = width_index + 2
                break
            else:
                left_part = width_index
                break


for width_index in range(width - 1, int(width_center * (1 + 0.7)), -1):
    temp_1 = erosion[:, width_index]
    temp_2 = erosion[:, width_index - 1]
    temp_3 = erosion[:, width_index - 2]
    temp_4 = erosion[:, width_index - 3]
    white_point_num_1, white_point_num_2, white_point_num_3, white_point_num_4 = \
        sum(temp_1) / 255, sum(temp_2) / 255, sum(temp_3) / 255, sum(temp_4) / 255
    white_point_num_1 = int(white_point_num_1)
    white_point_num_2 = int(white_point_num_2)
    white_point_num_3 = int(white_point_num_3)
    white_point_num_4 = int(white_point_num_4)
    # if white_point_num_2 - white_point_num_1 > 15:
    #     right_part = width_index
    #     break
    if white_point_num_1 == 0:
        continue
    else:
        if white_point_num_2 == 0:
            if white_point_num_3 == 0 and white_point_num_4 == 0:
                continue
            elif white_point_num_3 != 0 and white_point_num_4 == 0:
                continue
            elif white_point_num_3 == 0 and white_point_num_4 != 0:
                right_part = width_index + 2
                break
            else:
                right_part = width_index + 1
                break
        else:  # p1!=0  p2!=0
            if white_point_num_3 == 0 and white_point_num_4 == 0:
                continue
            elif white_point_num_3 != 0 and white_point_num_4 == 0:
                continue
            elif white_point_num_3 == 0 and white_point_num_4 != 0:
                right_part = width_index + 2
                break
            else:
                right_part = width_index
                break
# 对得到的边界做一个经验判断：应该在合理的范围内才可以,否则。。。扯
left_proportion = 0.45
right_proportion = 0.65
up_proportion = 0.45
down_proportion = 0.65
if up_part > int(height * up_proportion):
    up_part = int(height * 0.15)
if down_part < int(height * down_proportion):
    down_part = int(height * 0.95)
if left_part > int(width * left_proportion):
    left_part = int(width * 0.15)
if right_part < int(width * right_proportion):
    right_part = int(width * 0.95)

processed_plate = erosion[up_part: down_part, left_part: right_part]
print(up_part, down_part, left_part, right_part)
show_picture(processed_plate)

# 膨胀
# kernel_1 = np.ones((3, 3), np.uint8)
# kernel_1 = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(3, 3))
# temp_image_1 = cv2.dilate(src=processed_plate, kernel=kernel_for_erosion, iterations=1)
# show_picture(temp_image_1)
#
# # kernel_2 = np.ones((5, 5), np.uint8)
# kernel_2 = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(3, 3))
# temp_image_2 = cv2.erode(src=temp_image_1, kernel=kernel_2, iterations=1)
# show_picture(temp_image_2)
# # show_picture(erosion[up_part: down_part, left_part: right_part])
"""
# 车牌字符分割
"""
height_new, width_new = processed_plate.shape
vertical_data = []
# print(height_new, width_new)
for width_index in range(width_new):  # 按计算白色像素点的数目
    temp = processed_plate[:, width_index]
    white_pixel_num = int(sum(temp) / 255)
    vertical_data.append(white_pixel_num)

value_max = max(vertical_data)
value_min = min(vertical_data)
leverage = (value_max + value_min) // 7
plt.plot(vertical_data)
plt.plot([leverage]*len(vertical_data), color='red')
plt.title("white pixel distribution")
plt.show()
# if vertical_data[0] == 0:


















