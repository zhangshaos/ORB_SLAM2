"""测试图片缩小时的摩尔纹效应。
经过测试，多次放缩的摩尔纹效应小于一次放缩。
"""


import cv2
import numpy as np


RawDataDir = 'E:/Py_Projects/VCCPlaneRecon/datasets/generator/raw_data'


if __name__ == '__main__':
    oriFname = f'{RawDataDir}/0_scene.png'
    oriImg = cv2.imread(oriFname, cv2.IMREAD_COLOR)
    img256d = cv2.resize(oriImg, (256, 192))
    cv2.imwrite('./pic.png', oriImg)
    cv2.imwrite('./pic256d.png', img256d)
    img1440 = cv2.resize(oriImg, (1440, 1080))
    cv2.imwrite('./pic1440.png', img1440)
    img960 = cv2.resize(img1440, (960, 720))
    cv2.imwrite('./pic960.png', img960)
    img480 = cv2.resize(img960, (480, 360))
    cv2.imwrite('./pic480.png', img480)
    img256 = cv2.resize(img480, (256, 192))
    cv2.imwrite('./pic256.png', img256)
    pass