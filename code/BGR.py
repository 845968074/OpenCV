import matplotlib.pyplot as plt
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
def BRG():
    img = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]]
    ], dtype=np.uint8)

    plt.imsave('./image/img_plt.jpg', img)
    cv2.imwrite('./image/img_cv2.jpg', img)
def save_get_img():
    color_img=cv2.imread('./image/img_cv2.jpg')
    print(color_img.shape)
    gray_img=cv2.imread('./image/img_cv2.jpg',cv2.IMREAD_GRAYSCALE)
    print(gray_img,gray_img.shape)
    cv2.imwrite('./image/gray_img_cv2.png',gray_img)
    reload=cv2.imread('./image/gray_img_cv2.png',cv2.IMREAD_GRAYSCALE)
    print(reload,reload.shape)

def resize():
    # 读取一张四川大录古藏寨的照片
    img = cv2.imread('./image/2.jpg')
    print(img.shape)
    # 缩放成200x200的方形图像
    img_200x200 = cv2.resize(img, (200, 200))

    # 不直接指定缩放后大小，通过fx和fy指定缩放比例，0.5则长宽都为原来一半
    # 等效于img_200x300 = cv2.resize(img, (300, 200))，注意指定大小的格式是(宽度,高度)
    # 插值方法默认是cv2.INTER_LINEAR，这里指定为最近邻插值
    img_200x300 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)
    print(img_200x300.shape)
    # 在上张图片的基础上，上下各贴50像素的黑边，生成300x300的图像
    img_300x300 = cv2.copyMakeBorder(img, 50, 50, 0, 0,
                                     cv2.BORDER_CONSTANT,
                                     value=(0, 0, 0))
    print(img_300x300.shape)

    # 对照片中树的部分进行剪裁
    patch_tree = img[20:150, -180:-50]

    cv2.imwrite('./image/cropped_tree.jpg', patch_tree)
    cv2.imwrite('./image/resized_200x200.jpg', img_200x200)
    cv2.imwrite('./image/resized_200x300.jpg', img_200x300)
    cv2.imwrite('./image/bordered_300x300.jpg', img_300x300)

def HSV():
    img = cv2.imread('./image/2.jpg')
    print(img.shape)
    # 通过cv2.cvtColor把图像从BGR转换到HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # H空间中，绿色比黄色的值高一点，所以给每个像素+15，黄色的树叶就会变绿
    turn_green_hsv = img_hsv.copy()
    turn_green_hsv[:, :, 0] = (turn_green_hsv[:, :, 0] + 15) % 180
    turn_green_img = cv2.cvtColor(turn_green_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('./image/turn_green.jpg', turn_green_img)

    # 减小饱和度会让图像损失鲜艳，变得更灰
    colorless_hsv = img_hsv.copy()
    colorless_hsv[:, :, 1] = 0.5 * colorless_hsv[:, :, 1]
    colorless_img = cv2.cvtColor(colorless_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('./image/colorless.jpg', colorless_img)

    # 减小明度为原来一半
    darker_hsv = img_hsv.copy()
    darker_hsv[:, :, 2] = 0.5 * darker_hsv[:, :, 2]
    darker_img = cv2.cvtColor(darker_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('./image/darker.jpg', darker_img)

def Gamma():
    img = cv2.imread('./image/2.jpg')
    # 分通道计算每个通道的直方图
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])

    # 定义Gamma矫正的函数
    def gamma_trans(img, gamma):
        # 具体做法是先归一化到1，然后gamma作为指数值求出新的像素值再还原
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

        # 实现这个映射用的是OpenCV的查表函数
        return cv2.LUT(img, gamma_table)

    # 执行Gamma矫正，小于1的值让暗部细节大量提升，同时亮部细节少量提升
    img_corrected = gamma_trans(img, 0.5)
    cv2.imwrite('./image/gamma_corrected.jpg', img_corrected)

    # 分通道计算Gamma矫正后的直方图
    hist_b_corrected = cv2.calcHist([img_corrected], [0], None, [256], [0, 256])
    hist_g_corrected = cv2.calcHist([img_corrected], [1], None, [256], [0, 256])
    hist_r_corrected = cv2.calcHist([img_corrected], [2], None, [256], [0, 256])

    fig = plt.figure()

    pix_hists = [
        [hist_b, hist_g, hist_r],
        [hist_b_corrected, hist_g_corrected, hist_r_corrected]
    ]

    pix_vals = range(256)
    for sub_plt, pix_hist in zip([121, 122], pix_hists):
        ax = fig.add_subplot(sub_plt, projection='3d')
        for c, z, channel_hist in zip(['b', 'g', 'r'], [20, 10, 0], pix_hist):
            cs = [c] * 256
            ax.bar(pix_vals, channel_hist, zs=z, zdir='y', color=cs, alpha=0.618, edgecolor='none', lw=0)
        ax.set_xlabel('Pixel Values')
        ax.set_xlim([0, 256])
        ax.set_ylabel('Channels')
        ax.set_zlabel('Counts')

    plt.show()


if __name__ == '__main__':
    Gamma()