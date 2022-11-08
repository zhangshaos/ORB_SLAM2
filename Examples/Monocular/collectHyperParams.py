"""
根据设置的位置，自动生成 shenzhen_cameras.txt、shenzhen_images.txt 配置文件
"""


RawDataDir = 'E:/Py_Projects/VCCPlaneRecon/datasets/generator/raw_data'
TargetDir = 'E:/git_repos/ORB_SLAM2/Examples/Monocular'


import cv2
import os
import shutil


if __name__ == '__main__':
    camerasFname = f'{RawDataDir}/cameras.txt'
    shutil.copyfile(camerasFname, f"{TargetDir}/shenzhen_cameras.txt")
    imageCollection = []
    id = 0
    while True:
        fname = f'{RawDataDir}/{id}_scene.png'
        if os.path.exists(fname):
            imageCollection.append(fname)
            id += 1
        else:
            break
    with open('./shenzhen_images.txt', 'wt') as f:
        def strKey(s: str):
            s = ''.join(filter(str.isdigit, s))
            return -1 if len(s) == 0 else int(s)
        imageCollection.sort(key=strKey)
        for name in imageCollection:
            f.write(f'{name}\n')
    pass