import matplotlib.pylab as plt
import cv2
import numpy as np
import cannytest

cascade_src = 'cars.xml'
video_src = 'roadvidtimelapse.mp4'

car_cascade = cv2.CascadeClassifier(cascade_src)
fgbg = cv2.createBackgroundSubtractorMOG2()

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 1
    cv2.fillConvexPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is not None: 
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=3)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def filter_lines(lines):
    if lines is None:
        return []
    filtered_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 999 
            if 0.5 < abs(slope) < 2: 
                filtered_lines.append([[x1, y1, x2, y2]])
    return filtered_lines


def average_slope_intercept(lines):
    left_lines = []  
    right_lines = [] 
    left_weights = []
    right_weights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) 

            if slope < 0: 
                left_lines.append((slope, intercept))
                left_weights.append(length)
            elif slope > 0:  
                right_lines.append((slope, intercept))
                right_weights.append(length)

    def make_line(slope_intercept):
        if len(slope_intercept) > 0:
            slope, intercept = np.average(slope_intercept, axis=0, weights=left_weights if slope_intercept == left_lines else right_weights)
            y1, y2 = 600, 400  
            x1, x2 = int((y1 - intercept) / slope), int((y2 - intercept) / slope)
            return [[x1, y1, x2, y2]]
        return []

    left_lane = make_line(left_lines)
    right_lane = make_line(right_lines)

    return [left_lane, right_lane]

def process(image):
    if image is None:
        return None

    height, width = image.shape[:2]
    
    #  转换为灰度图并进行高斯模糊
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    #  Canny 边缘检测
    canny_image = cv2.Canny(blurred_image, 50, 150)
    # canny_image = cannytest.func(image) 我自己写的canny算法计算起来太慢了，希望能够改进之后直接调用
    kernel = np.ones((5, 5), np.uint8)
    dilated_canny = cv2.dilate(canny_image, kernel, iterations=1)

    # 设定兴趣区域
    roi_vertices = np.array([[[100, height], [width - 100, height], [width // 2, height // 2]]], np.int32)
    cropped_image = region_of_interest(dilated_canny, roi_vertices)

    # Hough 直线检测
    lines = cv2.HoughLinesP(cropped_image,
                            rho=1,
                            theta=np.pi/180,
                            threshold=50,  # 提高阈值
                            minLineLength=80,  # 过滤掉短线段
                            maxLineGap=100)  # 限制最大间隙

    # 过滤不符合角度的线
    filtered_lines = filter_lines(lines)

    # 绘制车道线
    lane_lines = average_slope_intercept(filtered_lines)
    image_with_lines = draw_the_lines(image, lane_lines)
    return image_with_lines


cap = cv2.VideoCapture(video_src)

while cap.isOpened():
    ret, img = cap.read()
    
    if not ret or img is None:
        break
        
    # 应用背景分割器
    fgbg.apply(img)
    
    # 处理图像以检测车道线
    img_with_lanes = process(img.copy())
    
    # 车辆检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    
    # 在车道线图像上绘制车辆边界框
    if img_with_lanes is not None:
        for (x, y, w, h) in cars:
            cv2.rectangle(img_with_lanes, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # 显示带有车道线和车辆检测的图像
        cv2.imshow('Lane and Vehicles Detection', img_with_lanes)
    
    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()