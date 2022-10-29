"""
data/small_imgs 中原本图片的分辨率为 256x192 ，但是 SLAM 系统似乎 track 失败了。  
将所有图片扩大为 x3 倍。
"""


import cv2
import os
import shutil


if __name__ == '__main__':
    root, dirs, files = next(os.walk('./data/small_imgs'))
    imageCollection = []
    for f in files:
        img = cv2.imread(os.path.join(root, f), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f'./data/big_imgs/{f}', img)
        imageCollection.append(os.path.abspath(f'./data/big_imgs/{f}'))
    with open('./shenzhen_images.txt', 'wt') as f:
        def strKey(s: str):
            s = ''.join(filter(str.isdigit, s))
            return -1 if len(s) == 0 else int(s)
        imageCollection.sort(key=strKey)
        for name in imageCollection:
            f.write(f'{name}\n')
    pass