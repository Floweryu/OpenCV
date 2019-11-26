import cv2
import numpy as np

'''
W——计算透射率时的收敛因子, 在[0, 1]之间，越大去雾效果越好,
但太接近于1又会出现亮斑,多次测试 W=0.91,0.90效果最好
'''
W = 0.91    
T0  = 0.1   # 投射率的阈值
hash = {}   # 把以像素为键，像素位置为值

'''引导滤波, 参考文章'''
def guided_filter(I, p, r=81, eps=0.001):
    mean_I = cv2.boxFilter(I, -1, (r, r))   # I的均值
    mean_p = cv2.boxFilter(p, -1, (r, r))   # p的均值
    mean_Ip = cv2.boxFilter(I * p, -1, (r, r))      # I *p 相乘后的均值
    cov_Ip = mean_Ip - mean_I * mean_p    # 每个部分 I, p 的协方差

    mean_II = cv2.boxFilter(I * I, -1, (r, r))  # 计算 I*I 的均值
    var_I = mean_II - mean_I * mean_I       # 计算 I 的方差
    # 计算 a , b 的值
    a = cov_Ip / (var_I + eps)
    b = mean_p - mean_I * a
    # 对包含像素点的所有a ,b 做平均
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    return mean_a * I + mean_b

'''
A是全球大气光成分
求出对RGB每个颜色通道的A值
'''
def get_A(img, width:int, length:int):
    '''
    计算亮度前0.01%的像素
    '''
    total_pixels = width * length       # 总像素
    take_pixels = int(0.0001 * total_pixels)   # 要取走的前 0.01% 像素
    count_pixels = 0    # 记录已取的像素个数
    location = []
    for i in range(255, -1, -1):         # 亮度高的像素优先
        leng_th = len(hash[i]) - 1       # 该亮度对应的个数
        while leng_th >= 0 and count_pixels < take_pixels:
            location.append(hash[i][leng_th])    # 将该亮度对应的像素全取走
            leng_th -= 1
            count_pixels += 1
        if count_pixels >= take_pixels:     
            break

    # 0.299*R + 0.587*G + 0.114*B
    # 根据RGB颜色权重求矩阵内积
    maxn = -1000000
    A = []
    for i in range(len(location)):
        x, y = location[i][0], location[i][1]
        temp = 0.144 * img[x][y][0] + 0.587 * img[x][y][1] + 0.299 * img[x][y][2]
        if temp > maxn:
            maxn = temp
            A = [x, y]
    A = np.array([[ img[x][y] ]])
    return A


'''最小值滤波'''
def min_filter(img, width:int, length:int, ksize, flag):  # ksize 核半径, flag 确定是否要求 hash
    if flag == True:        # True代表正常最小值滤波
        J_dark = img.copy()
        for i in range(0, 256):     # 初始化
            hash[i] = []
    else:       # False是dehaze函数中调用
        J_dark = np.zeros((width, length))
    
    for i in range(0, width):
        for j in range(0, length):
            up, down, left, right = i-ksize, i+ksize+1, j-ksize, j+ksize+1
            if up>=0 and down<=width and left>=0 and right<length:      # 中间部分
                J_dark[i][j] = img[up: down, left: right].min()
            elif up<0 and left<0 and right<length and down<width:       # 左上角
                J_dark[i][j] = img[0:down, 0:right].min()
            elif down>width and left<0 and up>=0 and right<length:      # 左下角
                J_dark[i][j] = img[up:width, 0:right].min()
            elif up<0 and right>length and down<width and left>=0:      # 右上角
                J_dark[i][j] = img[0:down, left:length].min()
            elif down>width and right>length and up>=0 and left>=0:     # 右下角
                J_dark[i][j] = img[up:width, left:length].min()
            elif left<0 and up>=0 and down<width and right<length:      # 正左
                J_dark[i][j] = img[up:down, 0:right].min()
            elif right>length and up>=0 and down<=width and left>=0:    # 正右
                J_dark[i][j] = img[up:down, left:length].min()
            elif up<0 and left>=0 and right<length and down<width:      # 正上
                J_dark[i][j] = img[0:down, left:right].min()
            elif down>width and up>=0 and left>=0 and right<length:     # 正下
                J_dark[i][j] = img[up:width, left:right].min()
            if flag == True:
                hash[J_dark[i][j]].append([i, j])       # 记录该最小值像素的位置，存到hash表中
    return J_dark


'''去雾'''
def dehaze(img, Gray, width, length, A, r=7):
    J_tmp = np.zeros((width, length))       # 存放各颜色通道的最小值
    t_x = np.zeros((width, length))
    '''
    根据公式求取投射率
    '''
    J_tmp = img / A             # 根据论文公式，除以全球大气成分A
    J_tmp = J_tmp.min(2)        # 颜色通道最小值
    t_x = min_filter(J_tmp, width, length, r, False)       # 根据公式再求最小值滤波
    t_x = 1 - W * t_x           # 根据公式求出透射率tx, W为收敛因子
    cv2.imshow("Transmission-Tx", (t_x * 255).astype(np.uint8))
    cv2.imwrite("DeFog\\Gongshi_tx.png", (t_x * 255).astype(np.uint8))

    '''
    根据引导滤波求取透射率
    '''
    t_x = guided_filter(Gray, t_x, r=81, eps=0.001)         # 用导向图计算投射率
    print("t_x:\n", t_x)
    cv2.imshow("GuidFilter-Tx", (t_x * 255).astype(np.uint8))
    cv2.imwrite("DeFog\\GuidFilter_tx.png", (t_x * 255).astype(np.uint8))

    for i in range(0, width):
        for j in range(0, length):
            for k in range(0, 3):
                # 根据公式求去雾后图片，T0是阈值，防止t_x过小
                img[i, j, k] = int(int((int(img[i, j, k]) - int(A[0, 0, k])) / max(t_x[i, j], T0)) + int(A[0, 0, k]))
                if img[i, j, k] > 255:
                    img[i, j, k] = 255
                if img[i, j, k] < 0:
                    img[i, j, k] = 0
                

if __name__ == "__main__":
    img = cv2.imread("D:\\Learn_Files\\OpenCV\DeFog\\sample\\train.jpg")
    cv2.imshow("Initial Image", img)
    width, length, d = img.shape   # 返回宽度，长度
    print("width", width)
    print("length", length)

    tmp = np.min(img, 2)        # 先求出图像像素RGB分量的最小值，存到灰度图中
    J_dark = min_filter(tmp, width, length, 7, True)    # 最灰度图进行最小值滤波
    print("hash:\n", hash)

    cv2.imshow("Gray-MinFilter", J_dark)
    cv2.imwrite("DeFog\\Gray_MinFilter.png", J_dark)

    A = get_A(img, width, length)       # 得到A值
    print("A:", A)

    t_x = dehaze(img, tmp / 255.0, width, length, A, 30)    # 去雾，这里
    cv2.imshow("DeFog Image", img)
    cv2.imwrite("DeFog\\DeFogImage.png", img)
    cv2.waitKey(0)
