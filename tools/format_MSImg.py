import os

import cv2
#data_dir = "D:\wenjian\质谱\代码\MassSpectrumCls\MassSpectrumCls\datas/rgb"
data_dir = "./datas/rgb"
#format_data_dir = "D:\wenjian\质谱\代码\MassSpectrumCls\MassSpectrumCls\datas/format"
format_data_dir = "./datas/format"
ROI_Pos = [0, 29, 2048, 2048]
if __name__ == '__main__':
    class_list = os.listdir(data_dir)
    for iclass in class_list:
        img_dir = os.path.join(data_dir, iclass)
        img_list = os.listdir(img_dir)

        format_img_dir=os.path.join(format_data_dir, iclass)
        if not os.path.exists(format_img_dir):
            os.makedirs(format_img_dir)
        for iimg_name in img_list:
            img = cv2.imread(os.path.join(img_dir, iimg_name))
            filepath=os.path.join(format_img_dir, iimg_name)
            x1, y1, x2, y2 = ROI_Pos
            cut_img = img[ y1: y2, x1: x2,:]
            cv2.imwrite(filepath, cut_img)
