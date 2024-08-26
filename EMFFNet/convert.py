import cv2

# 读取图像
image = cv2.imread(r'C:\Users\inspur\Downloads\EMFFNet-master\data\TestDataset\Nice1\images\0aa5f7804c34a359bbb402345d341253.jpeg')

# 检查图像是否正确读取
if image is None:
    print("Error: 图像未正确读取，请检查文件路径。")
    exit()

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用Canny边缘检测算法
edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)

# 显示原始图像
cv2.imshow('Original Image', image)

# 显示边缘图
cv2.imshow('Edges', edges)

# 等待用户按键，再继续执行
cv2.waitKey(0)

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()