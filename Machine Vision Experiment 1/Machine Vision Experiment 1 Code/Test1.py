import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os

#Question1: 读取图片
img_path1 = '/Users/guo2006/myenv/机器视觉/bxc.jpg'
img_path2 = '/Users/guo2006/myenv/机器视觉/background.jpg'
img_bxc = cv2.imread(img_path1) #opencv默认以BGR顺序读入
img_background = cv2.imread(img_path2)
if img_bxc is None or img_background is None:
    raise FileNotFoundError(f'未读取到图片，请检查路径')

#将BGR图像转换为RGB图像，使用matplotlib显示
img_bxc = cv2.cvtColor(img_bxc,cv2.COLOR_BGR2RGB)
#img_background = cv2.cvtColor(img_background,cv2.COLOR_BGR2RGB)
#如果使用matplotlib输出图像则要转换为RGB，使用opencv自带GUI则不用

#Question2: 显示
#plt.imshow(img_bxc)
#plt.axis('off')
#plt.title('BXC Original Picture')
#plt.show()
#还可以使用opencv自带的GUI
cv2.imshow('BXC Original Picture',img_bxc)
cv2.imshow('Background',img_background)
cv2.waitKey(0)
cv2.destroyAllWindows

#Question3: 打印图像尺寸、高度、宽度、通道数
height_1, width_1, channels_1 = img_bxc.shape
height_2, width_2, channels_2 = img_background.shape
#输出数据
print(f'图像1尺寸:{height_1} * {width_1} * {channels_1}')
print(f'图像1高度:{height_1},宽度:{width_1},通道数:{channels_1}')
print(f'图像2尺寸:{height_2} * {width_2} * {channels_2}')
print(f'图像1高度:{height_2},宽度:{width_2},通道数:{channels_2}')

#Question4: 转换图像为灰度图像
img_bxc_gray = cv2.cvtColor(img_bxc,cv2.COLOR_RGB2GRAY)#先前已经转换到RGB模式
img_background_gray = cv2.cvtColor(img_background,cv2.COLOR_BGR2GRAY)

cv2.imshow('BXC Gray Image',img_bxc_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(img_background_gray,cmap='gray')
plt.axis('on')
plt.show()

#Question5: 缩放、旋转、裁剪
#缩放
img_bxc_resize = cv2.resize(img_bxc,(300,200))#指定大小
plt.imshow(img_bxc_resize)
plt.axis('on')
plt.show()

scale = 0.2 #缩放0.2倍#指定比例
img_background_resize = cv2.resize(img_background,None,fx=scale,fy=scale)
cv2.imshow('BXC Resized Image',img_background_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
#旋转
img_path3 = '/Users/guo2006/myenv/机器视觉/before.jpg'
img_before = cv2.imread(img_path3)
if img_before is None:
    raise FileNotFoundError(f'图像路径错误，检查路径后重试')
else:
    h, w = img_before.shape[:2] #对.shape返回的三数单行矩阵切片
    center = (w//2, h//2) #选取图像中心作为旋转中心
    angle = 30 #逆时针 30°
    scale_1 = 1 #缩放倍数

    #计算旋转矩阵
    M = cv2.getRotationMatrix2D(center,angle,scale_1)
    #img_before_rotated = cv2.warpAffine(img_before,M,(w,h))
    
    #cv2.imshow('<Before> Rotated Picture',img_before_rotated)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() #四个角会被切掉
    
    #自动调整画布大小
    cos, sin = abs(M[0,0]), abs(M[0,1]) #|cosθ|与|sinθ|
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    #向M平移量tx,ty补一段差值
    M[0,2] += new_w/2 - center[0]   # center[0] 就是 w/2
    M[1,2] += new_h/2 - center[1]   # center[1] 就是 h/2
    #用更新后的矩阵 M 和扩大后的画布大小 (new_w, new_h) 做重采样
    img_before_rotated = cv2.warpAffine(img_before, M, (new_w, new_h))
    #显示
    cv2.imshow('<before> Rotated Image',img_before_rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#裁剪->数组切片
img_path4 = '/Users/guo2006/myenv/机器视觉/after.jpg'
img_after = cv2.imread(img_path4)
if img_after is None:
    raise FileNotFoundError(f'图像导入失败，检查路径后再试')
else:
    y1,y2 = 50,650#行范围
    x1,x2 = 10,780#列范围

    img_after_cropped = img_after[y1:y2,x1:x2]
    
    cv2.imshow('<after> Cropped Image',img_after_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#Question6: 加法、减法、乘法、除法运算
#加法->平均降噪
random.seed(0) #生成随机种子，让后面产生的随机噪声可重复

#图像加法函数
def show(title, *imgs):
    #把多张图水平拼成一行，统一高度后显示
    #*imgs可变长度参数，可以传任意多张图
    h0 = imgs[0].shape[0] #imgs[0]表示*imgs传入的第一张图，h0统一高度值
    
    resized_list = []
    for img in imgs: #对每张图给定长度，自动调节宽度
        resized_img = cv2.resize(img, (int(img.shape[1]*h0/img.shape[0]),h0))
        resized_list.append(resized_img)
    
    canvas = cv2.hconcat(resized_list) #水平方向把缩放后的图全部拼接
    cv2.imshow(title,canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#图像平均（降噪）
def add_noise(img, sigma=25):
    #添加高斯噪声
    noise = np.random.normal(0, sigma, img.shape).astype(np.uint8)
    #生成与图像同形状的 Gauss 噪声（均值为 0，方差 sigma²），并转成 0-255 的 uint8
    return cv2.add(img, noise) #饱和加法把噪声加到原图

img_path5 = '/Users/guo2006/myenv/机器视觉/bxc.jpg'
img_original = cv2.imread(img_path5)
height_bxc, width_bxc = img_original.shape[:2] #取前两个值
N = 50 #生成50张噪声图像

noise_img_list = []
for _ in range(N):
    noise_img_list.append(add_noise(img_original))

avg_noised = np.mean(noise_img_list, axis=0).astype(np.uint8)
#对每张图逐像素求平均，得到“理论噪声被抑制”的均值图，再转回uint8

show('加法-平均降噪', img_original, noise_img_list[0], avg_noised)
#加法->双重曝光(简单加法+权重)
background = cv2.imread(img_path2)
background = cv2.resize(background, (width_bxc, height_bxc)) #调整背景图片尺寸，与主体照片尺寸一致
double_exposure = cv2.addWeighted(img_original, 0.6, background, 0.4, 0)
#60 % 主体照片 + 40 % 背景 + 0 亮度偏移，实现“双重曝光”效果
show('加法-双重曝光',img_original,background,double_exposure)
#减法->静物差异检测
before = cv2.imread(img_path3)
after = cv2.imread(img_path4)
#修改为统一尺寸
height_exp, width_exp = before.shape[:2]
after = cv2.resize(after,(width_exp, height_exp))

img_diff_abs = cv2.absdiff(before, after) #逐像素做|img1 - img2|，得到差值图
img_diff_abs_gray = cv2.cvtColor(img_diff_abs, cv2.COLOR_BGR2GRAY)
#调用大津算法计算最佳阈值
thresh_used, mask = cv2.threshold(img_diff_abs_gray, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#thresh_used实际使用的阈值，mask二值化后的图像，0->阈值，255->最大值，二值化类型“大于阈值→255，否则→0”
print('Otsu自动阈值 =', thresh_used)

show('减法-差异', before, after, img_diff_abs)
show('减法-掩膜', img_diff_abs_gray, mask)
#乘法(掩膜抠图)
bxc = cv2.imread(img_path1)
h, w = bxc.shape[:2]
#二值掩膜：圆内全留，圆外全黑，硬边缘。
mask_bin = np.zeros((h, w),np.uint8) #创建一个特定形状和类型的新数组，其中所有元素的初始值都为0->此处为全黑画布
cv2.circle(mask_bin,(w//2,h//2),min(h, w)//3,255,-1)
#对象，圆心坐标，半径，颜色，-1=实心
result_bin = cv2.bitwise_and(bxc, bxc, mask=mask_bin)
#只有掩膜里 >0 的位置才把原图像素保留下来，其余位置变成黑
result_bin = cv2.cvtColor(result_bin, cv2.COLOR_BGR2RGB)

plt.imshow(result_bin)
plt.title('Binary Mask')
plt.axis('off')
plt.show()
#灰度掩膜：中心最亮，向外渐暗，柔边缘。
Y, X = np.ogrid[:h, :w] #生成两张“坐标网格”
dist = np.sqrt((X-w/2)**2+(Y-h/2)**2) #计算每个像素到图像中心的距离(距离图)
#把“距离”线性映射到 0–255 的灰度（掩膜规则）
mask_gray = np.clip((255-dist*(255/min(h,w)/2)),0,255).astype(np.uint8)
#将淹膜与原图叠加
result_gray = cv2.bitwise_and(bxc, bxc, mask= mask_gray)

show('乘法-二值掩膜', bxc, cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR), result_bin)
show('乘法-灰度掩膜', bxc, cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR), result_gray)
#除法
#常数除法——把整张图均匀变暗
new_bxc = cv2.imread(img_path1).astype(np.float32)
#先转成 0-255 的浮点数，方便后面除
dark_bxc = (new_bxc/3).clip(0,255).astype(np.uint8)
#每个像素值 ÷3，再截断回 0-255
dark_bxc = cv2.cvtColor(dark_bxc, cv2.COLOR_BGR2RGB)
plt.imshow(dark_bxc)
plt.title('Dark BXC Image')
plt.axis('off')
plt.show()

#创建光照模版 抵消光照不均（图像校正）
pic_height, pic_width = img_bxc_gray.shape[:2]
dY, dX = np.ogrid[:pic_height, :pic_width] #创建网格
light_illum = np.sqrt((dX-pic_width)**2+(dY-pic_height)**2) #到图片中心欧氏距离
light_illum = (light_illum / light_illum.max() * 0.8 + 0.2).astype(np.float32) #归一化到 0.2~1.0
#light_illum.max()取light_illum整张距离图中心的最大值（四角最远那点）
#使用除法抵消暗区
corrected_img = (img_bxc_gray / light_illum).clip(0,255).astype(np.uint8)
#除法 + 255截断 + 转换
show('除法-光照校正(灰度双通道图)', img_bxc_gray,(light_illum*255).astype(np.uint8), corrected_img)

pic_height, pic_width = bxc.shape[:2]
dY, dX = np.ogrid[:pic_height, :pic_width] #创建网格
light_illum = np.sqrt((dX-pic_width)**2+(dY-pic_height)**2) #到图片中心欧氏距离
light_illum = (light_illum / light_illum.max() * 0.8 + 0.2).astype(np.float32) #归一化到 0.2~1.0
#light_illum.max()取light_illum整张距离图中心的最大值（四角最远那点）

#扩充光线模版为3通道
light_illum_3ch = cv2.merge([light_illum, light_illum, light_illum])  # 变 (H,W,3)
#使用除法抵消暗区
corrected_img = (bxc / light_illum_3ch).clip(0,255).astype(np.uint8)
#除法 + 255截断 + 转换
show('除法-光照校正(彩色三通道图)', bxc,(light_illum_3ch*255).astype(np.uint8), corrected_img)

#Question7: 解码：ROI、解码、显示
#加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_img = cv2.imread(img_path1)
if face_img is None:
    raise FileNotFoundError(f'图像路径错误，检查路径后重试')
face_img_gray = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
#检测脸部->得到ROI掩膜
faces = face_cascade.detectMultiScale(face_img_gray, scaleFactor=1.2, minNeighbors=5)
#返回值faces是一个 N×4 的 numpy 数组，每行 (x, y, w, h) 代表一张人脸的左上角坐标和宽高。
#创建全黑掩膜
mask = np.zeros(face_img.shape[:2], dtype=np.uint8)   # 单通道全黑
#脸部涂白
for (x, y, w, h) in faces:
    cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    #绘制在mask上，左上角，右下角，255=白，实心填充
#生成随机密钥
random_key = np.random.randint(0,256,face_img.shape,dtype=np.uint8) #0-255随机
#生成的尺寸与face_img完全一样的三位数组
#只对人脸区域(mask>0区域)进行异或加密
encrypted = face_img.copy() #先整图复制一份，后面只改人脸部分，背景不动
encrypted[mask > 0] = cv2.bitwise_xor(face_img, random_key)[mask > 0]

cv2.imshow('Encrypted<Face ROI>',encrypted)
cv2.waitKey(0)
cv2.destroyAllWindows()
#在进行一次异或加密进行解密
decrypted = encrypted.copy()
decrypted[mask > 0] = cv2.bitwise_xor(encrypted, random_key)[mask > 0]

cv2.imshow('Decrypted<Face ROI>',decrypted)
cv2.waitKey(0)
cv2.destroyAllWindows()

#输出: cv2.imwrite('文件名',变量名)