# 灰度化
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
def gray(img_path):
	"""
	计算公式：
	Gray(i,j) = [R(i,j) + G(i,j) + B(i,j)] / 3
	or :
	Gray(i,j) = 0.299 * R(i,j) + 0.587 * G(i,j) + 0.114 * B(i,j)
	"""
	
	# 读取图片
	img = plt.imread(img_path)
	# BGR 转换成 RGB 格式
	img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	# 灰度化
	img_gray = np.dot(img_rgb[...,:3],[0.299,0.587,0.114])
	return img_gray

# 去除噪音 - 使用 5x5 的高斯滤波器
def smooth(img_gray):
	# 生成高斯滤波器
	"""
	要生成一个 (2k+1)x(2k+1) 的高斯滤波器，滤波器的各个元素计算公式如下：
	H[i, j] = (1/(2*pi*sigma**2))*exp(-1/2*sigma**2((i-k-1)**2 + (j-k-1)**2))
	"""
	sigma = 1.4
	gaussian_sum = 0
	gaussian_kernel = np.zeros((5,5))
	for i in range(5):
		for j in range(5):
			gaussian_kernel[i,j] = np.exp((-np.square(i-3)-np.square(j-3))/(2*np.square(sigma)))/(2*math.pi*np.square(sigma))
			gaussian_sum += gaussian_kernel[i,j]
	# 归一化处理
	gaussian_kernel  /= gaussian_sum
	
	# 高斯滤波
	H,W = img_gray.shape
	img_gray_denoised = np.zeros((H-5,W-5))
	for i in range(H-5):
		for j in range(W-5):
			img_gray_denoised[i,j] = np.sum(img_gray[i:i+5,j:j+5]*gaussian_kernel)
	return img_gray_denoised
	
def gradients(img_gray_denoised):
    """
	输入：高斯模糊之后的图像 \n
	输出： dx: x方向的梯度 
           dy: y方向梯度 
           M: 梯度幅值（两个矢量和的模长） \n
           theta: 梯度方向
	"""
    H,W = img_gray_denoised.shape
    dx = np.zeros((H-1,W-1))
    dy = np.zeros((H-1,W-1))
    M = np.zeros((H-1,W-1))
    theta = np.zeros((H-1,W-1))

    for i in range(H-1):
        for j in range(W-1):
            dy[i,j] = img_gray_denoised[i+1,j] - img_gray_denoised[i,j]
            dx[i,j] = img_gray_denoised[i,j+1] - img_gray_denoised[i,j]
            M[i,j] = np.sqrt(np.square(dx[i,j])+np.square(dy[i,j]))
            theta[i,j] = math.atan(dy[i,j]/(dx[i,j]+0.000000001))
    return dx,dy,M,theta	

def NMS(M,dx,dy):
    d = np.copy(M)
    H,W = M.shape
    NMS = np.copy(d)
    NMS[0,:] = NMS[H-1,:] = NMS[:,0] = NMS[:,W-1] = 0
    for i in range(1,H-1):
        for j in range(1,W-1):
            if M[i,j] == 0:
                NMS[i,j] = 0
            else:
                gradX = dx[i,j]
                gradY = dy[i,j]
                gradTemp = d[i,j]

                if(np.abs(gradX)>np.abs(gradY)):
                    weight = np.abs(gradY) / np.abs(gradX)
                    g2 = d[i,j-1]
                    g4 = d[i,j+1]
                    if gradX * gradY > 0:
                        g3 = d[i+1,j+1]
                        g1 = d[i-1,j-1]
                    else:
                        g3 = d[i-1,j+1]
                        g1 = d[i+1,j-1]
                else:
                    weight = np.abs(gradX)/np.abs(gradY)
                    g2 = d[i-1,j]
                    g4 = d[i+1,j]
                    if gradX * gradY > 0:
                        g3 = d[i+1,j+1]
                        g1 = d[i-1,j-1]
                    else:
                        g3 = d[i+1,j-1]
                        g1 = d[i-1,j+1]
                gradTemp1 = weight * g1 + (1-weight) * g2
                gradTemp2 = weight * g3 + (1-weight) * g4
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    NMS[i,j] = gradTemp
                else:
                    NMS[i,j] = 0
    return NMS



def double_threshold(NMS):
    H,W = NMS.shape
    DT = np.zeros((H,W))
    TL = 0.1 * np.max(NMS)
    TH = 0.3 * np.max(NMS)

    for i in range(1,H-1):
        for j in range(1,W-1):
            if (NMS[i,j]<TL):
                DT[i,j] = 0
            elif (NMS[i,j]>TH):
                DT[i,j] = 1
            elif (NMS[i-1, j-1:j+2] < TH).any() or (NMS[i+1, j-1:j+2]).any() or (NMS[i, [j-1, j+1]] < TH).any():
                DT[i,j] = 1
    return DT

def func(image):
     gray_img = gray(image)
     smoothed_img = smooth(gray_img)
     dx,dy,M,theta = gradients(smoothed_img)
     after_img = NMS(M,dx,dy)
     last_img = double_threshold(after)
     return last_img

gray_image=gray('test.jpg')
smoothed_image = smooth(gray_image)
dx,dy,M,theta = gradients(smoothed_image)
after = NMS(M,dx,dy)
last = double_threshold(after)
# 转换数据格式为 0-255 并保存
final_output = (last * 255).astype(np.uint8)
cv2.imwrite("test_canny_myself.jpg", final_output)

# plt.figure(figsize=(10,10))
# plt.imshow(last,cmap='gray')
# plt.show()