import numpy as np
import cv2
import math

def hough_line_transform(edges):
    # 获取图像的宽度和高度
    height, width = edges.shape
    
    # 计算ρ的最大值 (最大距离)
    diag = int(np.sqrt(height**2 + width**2))
    max_rho = diag
    
    # 计算θ的范围 (从 0 到 180)
    max_theta = 180
    
    # 初始化累加器 (ρ, θ)
    accumulator = np.zeros((max_rho * 2, max_theta), dtype=int)
    
    # 角度步长 (θ) 的增量
    theta_step = np.pi / 180
    
    # 对每个边缘点 (x, y) 进行霍夫变换
    for y in range(height):
        for x in range(width):
            if edges[y, x] == 255:  # 只处理边缘点
                # 对每个θ值进行ρ计算
                for theta_index in range(max_theta):
                    theta = theta_index * theta_step
                    rho = int(x * np.cos(theta) + y * np.sin(theta)) + max_rho
                    accumulator[rho, theta_index] += 1
    
    return accumulator

def draw_lines(image, accumulator, threshold):
    # 从累加器中提取出最大的ρ和θ值，识别出直线
    height, width = image.shape
    lines = []
    for rho in range(accumulator.shape[0]):
        for theta in range(accumulator.shape[1]):
            if accumulator[rho, theta] > threshold:
                rho_value = rho - accumulator.shape[0] // 2
                theta_value = theta * np.pi / 180
                
                # 将ρ和θ转回笛卡尔坐标
                a = np.cos(theta_value)
                b = np.sin(theta_value)
                x0 = a * rho_value
                y0 = b * rho_value
                
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                lines.append(((x1, y1), (x2, y2)))
                
                # 画出这些线
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image, lines

# 读取图像
image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

# 使用 Canny 算法进行边缘检测
edges = cv2.Canny(image, 50, 150)

# 进行霍夫变换
accumulator = hough_line_transform(edges)

# 可视化和提取直线
threshold = 150  # 设定阈值
result_image, detected_lines = draw_lines(image.copy(), accumulator, threshold)

# 显示结果
cv2.imshow("Detected Lines", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
