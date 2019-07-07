# encoding: utf-8
import cv2
# from car_plate_project.code.functions import show_picture
import numpy as np

# path = r'E:\code\112\opencv3_project\plate_recognition\plate_samples\5.png'


# 定义旋转图像的函数
def rotate_plate(image_path, show_img=False):
    """
    :param image_path:  输入的图像是灰度图像
    :param show_img:  bool
    :return:
    """
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    # 需要将图像反转：
    # 将文字变成浅色而将背景变成深色
    # 前景为浅色， 背景为黑色
    image_gray = cv2.bitwise_not(image_gray)
    show_picture(image_gray)
    # 获取二值化后的图像矩阵
    thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # print(thresh)
    coords = np.column_stack(np.where(thresh > 0))    # 找到像素值大于零的点，即前景的区域
    # print(coords)
    angle = cv2.minAreaRect(coords)[-1]   # 返回的是coords所包含的最小矩形（x_min, y_min, x_max, y_max）,还有偏转角度
    # print(angle)[-90, 0]之间的值
    if angle < -45:
        angle = -(angle + 90)
    # print(angle)
    if abs(angle) < 1.5:
        print('not rotated')
        rotated = image
        return rotated
    else:
        w, h = image_gray.shape[0], image_gray.shape[1]  # 确定图像的中心
        center = (w // 2, h // 2)
        # print(center)
        # 根据图像中心确定旋转矩阵
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
        # 进行一个放射变换对图像进行变换：
        rotated = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(h, w),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        rotated_temp = rotated.copy()
        cv2.putText(img=rotated_temp, text='{: .2f} degrees'.format(angle), org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7, color=(0, 0, 255), thickness=2)
    if show_img:
        show_picture(rotated_temp)
    return rotated


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


# rotate_plate(image_path=path, show_img=True)














