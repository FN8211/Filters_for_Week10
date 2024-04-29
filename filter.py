import cv2
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

img = cv.imread('e_cig_1 (1).jpg')

# 均值滤波
img_blur = cv.blur(img, (3, 3))  # (3,3)代表卷积核尺寸，随着尺寸变大，图像会越来越模糊
img_blur = cv.cvtColor(img_blur, cv.COLOR_BGR2RGB)  # BGR转化为RGB格式

# 高斯滤波
img_GaussianBlur = cv.GaussianBlur(img, (3, 3), 0, 0)  # 参数说明：(源图像，核大小，x方向的标准差，y方向的标准差)
img_GaussianBlur = cv.cvtColor(img_GaussianBlur, cv.COLOR_BGR2RGB)  # BGR转化为RGB格式

# 中值滤波
img_medianBlur = cv.medianBlur(img, 3)
img_medianBlur = cv.cvtColor(img_medianBlur, cv.COLOR_BGR2RGB)  # BGR转化为RGB格式

# 双边滤波
# 参数说明：(源图像，核大小，sigmaColor，sigmaSpace)
img_bilateralFilter = cv.bilateralFilter(img, 50, 100, 100)
img_bilateralFilter = cv.cvtColor(img_bilateralFilter, cv.COLOR_BGR2RGB)  # BGR转化为RGB格式

titles = ['img_blur','img_GaussianBlur', 'img_medianBlur', 'img_bilateralFilter']
images = [img_blur, img_GaussianBlur, img_medianBlur, img_bilateralFilter]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i]), plt.title(titles[i])
    plt.axis('off')
plt.show()


# 低通滤波
def Low_Pass_Filter(srcImg_path):
    # img = cv.imread('srcImg_path', 0)
    img = np.array(Image.open(srcImg_path))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 傅里叶变换
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    # 设置低通滤波器
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

    # 掩膜图像和频谱图像乘积
    f = fshift * mask

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv.idft(ishift)
    res = cv.magnitude(iimg[:, :, 0], iimg[:, :, 1])

    return res


# 高通滤波
def High_Pass_Filter(srcImg_path):
    # img = cv.imread(srcImg_path, 0)
    img = np.array(Image.open(srcImg_path))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 傅里叶变换
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    # 设置高通滤波器
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # 掩膜图像和频谱图像乘积
    f = fshift * mask

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv.idft(ishift)
    res = cv.magnitude(iimg[:, :, 0], iimg[:, :, 1])

    return res

def Mean_Binarization(srcImg_path):
    img = cv.imread(srcImg_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshold = np.mean(img_gray)
    print(threshold)
    img_gray[img_gray>threshold] = 255
    img_gray[img_gray<=threshold] = 0
    plt.imshow(img_gray, cmap='gray')
    plt.title('Mean_Binarization')
    plt.show()

def Hist_Binarization(srcImg_path):
    img = cv.imread(srcImg_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist = img_gray.flatten()
    plt.subplot(121)
    plt.hist(hist,256)

    cnt_hist = Counter(hist)
    print(cnt_hist)
    begin,end = cnt_hist.most_common(2)[0][0],cnt_hist.most_common(2)[1][0]
    if begin > end:
        begin, end = end, begin
    print(f'{begin}: {end}')

    cnt = np.iinfo(np.int16).max
    threshold = 0
    for i in range(begin,end+1):
        if cnt_hist[i]<cnt:
            cnt = cnt_hist[i]
            threshold = i
    print(f'{threshold}: {cnt}')
    img_gray[img_gray>threshold] = 255
    img_gray[img_gray<=threshold] = 0
    plt.subplot(122)
    plt.imshow(img_gray, cmap='gray')
    plt.show()


def Otsu(srcImg_path):
    img = cv.imread(srcImg_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    threshold_t = 0
    max_g = 0

    for t in range(255):
        front = img[img < t]
        back = img[img >= t]
        front_p = len(front) / (h * w)
        back_p = len(back) / (h * w)
        front_mean = np.mean(front) if len(front) > 0 else 0.
        back_mean = np.mean(back) if len(back) > 0 else 0.

        g = front_p * back_p * ((front_mean - back_mean) ** 2)
        if g > max_g:
            max_g = g
            threshold_t = t
    print(f"threshold = {threshold_t}")

    img[img < threshold_t] = 0
    img[img >= threshold_t] = 255
    plt.imshow(img, cmap='gray')
    plt.title('Otsu')
    plt.show()


img_Low_Pass_Filter = Low_Pass_Filter('e_cig_1 (1).jpg')
plt.subplot(121), plt.imshow(img_Low_Pass_Filter, 'gray'), plt.title('img_Low_Pass_Filter')
plt.axis('off')

img_High_Pass_Filter = High_Pass_Filter('e_cig_1 (1).jpg')
plt.subplot(122), plt.imshow(img_High_Pass_Filter, 'gray'), plt.title('img_High_Pass_Filter')
plt.axis('off')

plt.show()

Mean_Binarization('e_cig_1 (1).jpg')
Hist_Binarization('e_cig_1 (1).jpg')
Otsu('e_cig_1 (1).jpg')