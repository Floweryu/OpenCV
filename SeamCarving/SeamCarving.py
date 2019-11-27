import cv2
import numpy as np


'''使用OpenCV函数计算能量值'''
def get_sobel(img):
    '''
    cv2.Sobel()求函数梯度
    原uint8位数不够，要转为16位
    0 , 1是求导的阶数
    '''
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    # 转为uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    # 将两个方向上的梯度组合
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst


'''
Find the gap with the smallest energy
'''
def dp_mininum_energe(img):
    row, col = img.shape
    print("row", row)
    print("col", col)
    dp = np.zeros((row, col))
    path = np.zeros((row, col))
    ans = np.ones((row, col), dtype=np.bool)
    dp[0] = img[0]      # 从第一行像素开始向下计算，img[0]表示第一行像素

    for i in range(1, row):
        for j in range(0, col):
            # j - 1 到 j + 1 范围内最小像素值下标
            ans_min = np.argmin(img[i - 1][max(0, j - 1):min(j + 2, col)])    # 这里求的是在切片中的相对位置,即[0, 2]之间      
            ans_min += max(0, j - 1)    # 加上列数，变成相对于图片的绝对位置
            dp[i][j] = int(img[i][j]) + int(img[i - 1][ans_min])
            path[i][j] = ans_min        # 记录最小像素值下标
    
    # 在最后一行找到最小像素值的下标
    min_pos = np.argmin(dp[row - 1][0 : col])
    # 从最后一行向上遍历
    for i in range(row - 1, -1, -1):
        ans[i][int(min_pos)] = False     # 该像素点被选了
        min_pos = path[i][int(min_pos)]  
    return ans


def cut_column(img, ans):
    row, col, c = img.shape
    ans = np.stack([ans] * 3, axis=2)
    img = img[ans].reshape((row, col - 1, 3))
    return img

if __name__ == "__main__":
    img = cv2.imread(r"D:\Learn_Files\OpenCV\SeamCarving\sample\test1.jpg")
    cv2.imshow("Initial Image", img)
    Xtimes = int(input("请输入X方向裁剪："))
    Ytimes = int(input("请输入Y方向裁剪："))
    while Xtimes > 0:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_sobel = get_sobel(img_gray)
        ans = dp_mininum_energe(img_sobel)
        img = cut_column(img, ans).astype(np.uint8)
        Xtimes -= 1
    # cv2.imshow("X-Energe Image", img)

    img = img.swapaxes(0, 1)    # 变换坐标轴的方向

    while Ytimes > 0:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_sobel = get_sobel(img_gray)
        ans = dp_mininum_energe(img_sobel)
        img = cut_column(img, ans).astype(np.uint8)
        Ytimes -= 1
    # cv2.imshow("Y-Energe Image", img)
    img = img.swapaxes(0, 1)    # 变换坐标轴的方向
    cv2.imshow("Energe Image", img)
    cv2.waitKey(0)
