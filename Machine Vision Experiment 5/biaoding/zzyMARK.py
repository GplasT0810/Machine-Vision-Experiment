import cv2
import numpy as np
import glob
import os

# 1. 准备对象点（棋格的真实世界坐标）
chessboard_size = (5, 8)  # 内角点数量（列，行）
square_size = 27.0

# 假设每个格子是1单位长度，实际标定可以输入真实尺寸
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
objp[:, :2] *= square_size

# 2. 存储对象点和图像点的数组
objpoints = []  # 真实世界中的3D点
imgpoints = []  # 图像中的2D点

# 3. 读取所有标定图像
folder = r'C:\Users\lenovo\Desktop\MV'
images = glob.glob(os.path.join(folder, '*.jpg')) \
       + glob.glob(os.path.join(folder, '*.png'))  # 两种后缀都兜底

if len(images) == 0:
    print("未找到标定图像，请检查路径是否正确")
    exit()

print(f"找到 {len(images)} 张标定图像")

# 4. 遍历每张图像进行角点检测
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"无法读取图像: {fname}")
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 查找棋格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        print(f"在 {fname} 中找到角点")
        objpoints.append(objp)
        
        # 提高角点检测精度
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        
        # 可视化角点（可选）
        img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(500)
    else:
        print(f"在 {fname} 中未找到角点")

cv2.destroyAllWindows()

# 5. 检查是否找到足够的图像进行标定
if len(objpoints) < 3:
    print("找到的可用图像太少，至少需要3张图像")
    exit()

print(f"使用 {len(objpoints)} 张图像进行标定")

# 6. 执行相机标定
print("开始相机标定...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n标定结果:")
print(f"重投影误差: {ret}")
print("相机内参矩阵:")
print(mtx)
print("畸变系数:")
print(dist)

# 7. 标定结果验证 - 计算重投影误差
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"\n平均重投影误差: {mean_error / len(objpoints)} 像素")

# 8. 保存标定结果
np.savez('camera_calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print("\n标定结果已保存到 camera_calibration.npz")

# 9. 使用标定结果进行畸变校正（验证）
if len(images) > 0:
    # 使用第一张图像进行演示
    img = cv2.imread(images[0])
    h, w = img.shape[:2]
    
    # 优化相机矩阵
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # 校正图像
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # 裁剪图像
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    # 显示结果
    cv2.imshow('Original Image', img)
    cv2.imshow('Undistorted Image', dst)
    
    # 保存校正后的图像
    cv2.imwrite('undistorted_image.jpg', dst)
    cv2.imwrite('original_image.jpg',img)
    print("畸变校正图像已保存为 undistorted_image.jpg")

    # ---------- 9. 带角点的畸变校正（示范用第一张） ----------
if len(images) > 0:
    img  = cv2.imread(images[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # （1）先在校正前图上重新检角点并画上去
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:                       # 如果找到就画
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.drawChessboardCorners(img, chessboard_size, corners, True)

    # （2）再去畸变
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # （3）裁剪黑边（可选）
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # （4）显示 & 保存
    cv2.imshow('Original (with corners)', img)
    cv2.imshow('Undistorted (with corners)', dst)

    cv2.imwrite('undistorted_with_corners.jpg', dst)
    print("已保存带角点的校正图：undistorted_with_corners.jpg")

    # 生成未校正 & 校正两张图（同尺寸，不裁剪）
    img_raw   = cv2.imread(images[0])
    h, w = img_raw.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), cv2.CV_16SC2)
    img_undist = cv2.remap(img_raw, map1, map2, cv2.INTER_LINEAR)

    # 计算差异并增强对比
    diff = cv2.absdiff(img_raw, img_undist)
    diff = cv2.convertScaleAbs(diff, alpha=1.5, beta=0)   # 放大 5 倍
    cv2.imshow('diff x5', diff)
    cv2.imwrite('diff_x5.jpg', diff)#白色为畸变修正后相对于原图改动的部分

    print("\n标定完成！")