# encoding: utf-8
from PIL import Image
import os

base_path = r'E:\smart city\data_set\功能评测图像库'
base_out_path = r'E:\code\112\models\dataset\tesi_images'
dirs = os.listdir(base_path)

index = 1
flag = True
for dir_1 in dirs:
    temp_path_1 = os.path.join(base_path, dir_1)
    for dir_2 in os.listdir(temp_path_1):
        if flag:
            flag = False
            continue
        # print(dir_2)
        temp_path_2 = os.path.join(temp_path_1, dir_2)
        # print(temp_path_2)
        for dir_3 in os.listdir(temp_path_2):
            temp_path_3 = os.path.join(temp_path_2, dir_3)
            print(temp_path_3)
            for image in os.listdir(temp_path_3):
                temp = os.path.join(temp_path_3, image)
                img = Image.open(temp)
                img.save(base_out_path + '\\' + 'image' + str(index) + '.jpg')
                index += 1
                if index >= 100:
                    break

