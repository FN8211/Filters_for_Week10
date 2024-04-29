import cv2
import numpy as np

# 读取图像
image = cv2.imread('VOCdevkit/VOC2007/JPEGImages/2007_007818.jpg')
mask = cv2.imread('mask/2007_007818.png')

detected_boxes = [
    (324, 109, 376, 172),
    (83, 137, 131, 170),
    (0, 142, 38, 164),
    (136, 135, 180, 155),
    (185, 139, 222, 162),
    (39, 133, 84, 167),
    (1, 154, 49, 256)
]

# 转换图像为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用Canny边缘检测
lower_threshold = 128
upper_threshold = 200
edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)

# 创建一个与原始图像同样大小和类型的全黑图像
overlay = np.zeros_like(image)

# 将边缘检测结果（二值图像）复制到全黑图像上
# 使用绿色（BGR格式中的[0, 255, 0]）标记边缘
overlay[edges == 255] = [0, 255, 0]

# 将边缘图像和原始图像混合
result = cv2.addWeighted(mask, 0.5, overlay, 0.5, 0)
for box in detected_boxes:
    x1, y1, x2, y2 = box
    # 定义标记框的颜色，例如蓝色，使用BGR格式
    color = (255, 0, 0)
    # 定义标记框的厚度
    thickness = 2
    # 在图像上绘制矩形标记框
    result = cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

# 显示结果
cv2.imshow('Edges Overlaid on Original Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('img/edges.jpg', result)