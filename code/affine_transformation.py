import cv2
import numpy as np

def affine():
    # 读取一张斯里兰卡拍摄的大象照片 (1920, 1080, 3)
    img = cv2.imread('./image/2.jpg')
    print(img.shape)
    # 沿着横纵轴放大1.6倍，然后平移(-150,-240)，最后沿原图大小截取，等效于裁剪并放大
    M_crop_elephant = np.array([
        [1, 0, -150],
        [0, 1,-240]
    ], dtype=np.float32)

    img_elephant = cv2.warpAffine(img, M_crop_elephant, (1080,1920))
    cv2.imwrite('./image/lanka_elephant.jpg', img_elephant)

    # x轴的剪切变换，角度15°
    theta = 15 * np.pi / 180
    M_shear = np.array([
        [1, np.tan(theta), 0],
        [0, 1, 0]
    ], dtype=np.float32)

    img_sheared = cv2.warpAffine(img, M_shear, (1080,1920))
    cv2.imwrite('./image/lanka_safari_sheared.jpg', img_sheared)

    # 顺时针旋转，角度15°
    M_rotate = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0]
    ], dtype=np.float32)

    img_rotated = cv2.warpAffine(img, M_rotate, (1080,1920))
    cv2.imwrite('./image/lanka_safari_rotated.jpg', img_rotated)

    # 某种变换，具体旋转+缩放+旋转组合可以通过SVD分解理解
    M = np.array([
        [1, 1.5, -400],
        [0.5, 2, -100]
    ], dtype=np.float32)

    img_transformed = cv2.warpAffine(img, M, (1080,1920))
    cv2.imwrite('./image/lanka_safari_transformed.jpg', img_transformed)

if __name__ == '__main__':
    affine()


