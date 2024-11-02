import os
import numpy as np
import cv2
from sklearn.cluster import KMeans

def SURF(img):
    """
    使用SIFT（尺度不变特征变换）检测图像中的关键点并计算描述符。

    参数:
    img: 输入的图像，灰度图像

    返回值:
    kp: 关键点列表
    des: 描述符数组，与关键点列表一一对应，每个关键点都有一个描述符，用于在不同图像间匹配关键点。
    """
    # 创建SIFT对象，用于后续的关键点检测和描述符计算
    surf = cv2.SIFT_create()

    # 使用SIFT对象检测图像中的关键点并计算描述符
    # img为输入图像，None表示不使用掩码，即处理整幅图像
    kp, des = surf.detectAndCompute(img, None)

    # 返回检测到的关键点和计算出的描述符
    return kp, des



def ByFlann(img1, img2, kp1, kp2, des1, des2, flag="ORB"):
    """
    使用FLANN匹配算法进行特征点匹配。

    参数:
    img1: 第一张图像。
    img2: 第二张图像。
    kp1: 第一张图像的特征点。
    kp2: 第二张图像的特征点。
    des1: 第一张图像的特征描述符。
    des2: 第二张图像的特征描述符。
    flag: 指定使用的特征提取方法，默认为"ORB"。

    返回:
    matches: 特征点匹配结果。
    """
    if (flag == "SIFT" or flag == "sift"):
        # SIFT方法
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                            trees=5)
        search_params = dict(check=50)
    else:
        # ORB方法
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(check=50)
    # 定义FLANN参数
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.match(des1, des2)
    return matches

def RANSAC(img1, img2, kp1, kp2, matches):
    """
    使用RANSAC算法找出移动的物体。

    参数:
    img1: 第一张图像。
    img2: 第二张图像。
    kp1: 第一张图像的关键点。
    kp2: 第二张图像的关键点。
    matches: 两幅图像间关键点的匹配。

    返回:
    move_rect: 移动物体的矩形区域列表。
    """
    # 初始化矩形列表
    move_rect = []

    # 最小匹配点数量阈值
    MIN_MATCH_COUNT = 10

    # 根据Lowe的比例测试存储所有好的匹配
    matchType = type(matches[0])
    good = []
    if isinstance(matches[0], cv2.DMatch):
        # 如果匹配类型是cv2.DMatch，则直接使用matches
        good = matches
    else:
        # 否则，使用knnMatch的结果，并应用比例测试
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

    # 如果有足够的匹配点，则尝试找到单应性矩阵
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M: 3x3 变换矩阵.
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        # 如果没有足够的匹配点，打印消息并设置matchesMask为None
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    # 使用RANSAC中得到的good匹配点
    if good and len(good) > 0:
        # 提取匹配点的坐标
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # 计算光流
        # 注意：这里我们实际上是在已知匹配点的基础上计算光流，通常情况下我们会直接在两帧图像上计算光流
        nextPts, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, pts1, None)

        # 选择状态为1的点，即成功追踪的点
        good_new = nextPts[status == 1]
        good_old = pts1[status == 1]

        # 计算光流向量
        flow_vectors = good_new - good_old

        # 聚类分析
        n_clusters = 5 # 预估的运动物体数量加上背景
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
        kmeans.fit(flow_vectors)

        # 获取每个点所属的类别
        labels = kmeans.labels_

        # 统计每个类别的点数
        label_counts = np.bincount(labels)

        # 找到点数最多的类别，认为它是背景
        background_label = np.argmax(label_counts)

        # 其他类别认为是运动物体
        moving_object_labels = [i for i in range(n_clusters) if i != background_label]

        # 图像尺寸
        h, w, c = img1.shape
        move_obj_mask = np.zeros((h, w), dtype=np.uint8)

        # 根据标签绘制运动物体
        for i, label in enumerate(labels):
            if label in moving_object_labels:
                x, y = good_new[i].ravel()
                if round(y) < h and round(x) < w:
                    move_obj_mask[round(y)][round(x)] = 255

        # 形态学操作，增强运动物体的掩膜
        kernel = np.ones((4, 4), np.uint8)
        move_obj_mask = cv2.dilate(move_obj_mask, kernel, iterations=7)
        move_obj_mask = cv2.erode(move_obj_mask, kernel, iterations=10)
        kernel = np.ones((5, 5), np.uint8)
        move_obj_mask = cv2.dilate(move_obj_mask, kernel, iterations=9)

        # 找到运动物体的轮廓
        contours, _ = cv2.findContours(move_obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤掉小的轮廓
        min_contour_area = 100  # 可以根据实际情况调整
        filtered_mask = np.zeros_like(move_obj_mask)
        for contour in contours:
            if 10000>cv2.contourArea(contour) > min_contour_area:
                cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # 再次找到过滤后的轮廓
        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历所有轮廓
        for contour in contours:
            # 计算轮廓的最小外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            rect = [x, y, x + w, y + h]
            move_rect.append(rect)

    return move_rect


def preprocessing(img1, img2):
    """
    利用前两帧初步确定运动物体大致范围

    参数:
    img1: 第一帧
    img2: 第二帧

    返回:
    move_rect: 运动物体分为列表
    """
    # 使用SURF算法处理第一张图片，获取关键点和描述符
    kp1, des1 = SURF(img1)
    # 使用SURF算法处理第二张图片，获取关键点和描述符
    kp2, des2 = SURF(img2)
    # 通过FLANN匹配算法，根据关键点和描述符在两张图片间找到匹配点
    matches = ByFlann(img1, img2, kp1, kp2, des1, des2, "SIFT")
    move_rect = RANSAC(img1, img2, kp1, kp2, matches)
    return move_rect




if __name__ == '__main__':
    video_dir = "video/drones"
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    img1 = cv2.imread(os.path.join(video_dir, frame_names[9]))
    img2 = cv2.imread(os.path.join(video_dir, frame_names[10]))
    move_rect = preprocessing(img1, img2)
    print(move_rect)

