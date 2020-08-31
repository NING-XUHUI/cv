# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %% 图像读取与显示
img = cv2.imread('cat.jpeg')
img

# %%
cv2.imshow('image', img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

# %%

cv2.startWindowThread()
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%

img.shape
img = cv2.imread('cat.jpeg', cv2.IMREAD_GRAYSCALE)
img.shape
# %%
cv2.imwrite('xxx.xx', img)

# %% 视频读取与显示
vc = cv2.VideoCapture('test.mp4')

# %%
if vc.isOpened():
    open, frame = vc.read()
else:
    open = False

# %%
while open:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('result', gray)
        if cv2.waitKey(1) & 0xFF == 27:
            break

vc.release()
cv2.destroyAllWindows()

# %% 截取图像数据
img = cv2.imread('cat.jpeg')
# %%
cat = img[0:200, 0:200]
cv2.imshow('cat', cat)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

# %% 颜色通道截取
b, g, r = cv2.split(img)
b
g
r
img = cv2.merge((b, g, r))
img.shape

# %%只保留r
cur_img = img.copy()
cur_img[:, :, 0] = 0
cur_img[:, :, 1] = 0
cv2.imshow('R', cur_img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

# %%g

cur_img = img.copy()
cur_img[:, :, 0] = 0
cur_img[:, :, 2] = 0
cv2.imshow('G', cur_img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

# %% 边界填充
