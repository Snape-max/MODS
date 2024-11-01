import os
import numpy as np
import cv2
from sklearn.cluster import KMeans

def SURF(img):
    surf = cv2.SIFT_create()
    kp, des = surf.detectAndCompute(img, None)
    return kp, des


def ByFlann(img1, img2, kp1, kp2, des1, des2, flag="ORB"):
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
    # 初始化矩形列表
    move_rect = []

    MIN_MATCH_COUNT = 10
    # store all the good matches as per Lowe's ratio test.
    matchType = type(matches[0])
    good = []
    if isinstance(matches[0], cv2.DMatch):
        # 搜索使用的是match
        good = matches
    else:
        # 搜索使用的是knnMatch
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M: 3x3 变换矩阵.
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
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

        h, w, c = img1.shape
        move_obj_mask = np.zeros((h, w), dtype=np.uint8)


        # 根据标签绘制运动物体
        for i, label in enumerate(labels):
            if label in moving_object_labels:
                x, y = good_new[i].ravel()
                if round(y) < h and round(x) < w:
                    move_obj_mask[round(y)][round(x)] = 255

        kernel = np.ones((4, 4), np.uint8)
        move_obj_mask = cv2.dilate(move_obj_mask, kernel, iterations=7)
        move_obj_mask = cv2.erode(move_obj_mask, kernel, iterations=10)
        kernel = np.ones((5, 5), np.uint8)
        move_obj_mask = cv2.dilate(move_obj_mask, kernel, iterations=9)

        contours, _ = cv2.findContours(move_obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤掉小的轮廓
        min_contour_area = 100  # 可以根据实际情况调整
        filtered_mask = np.zeros_like(move_obj_mask)
        for contour in contours:
            if 10000>cv2.contourArea(contour) > min_contour_area:
                cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        # 遍历所有轮廓
        for contour in contours:
            # 计算轮廓的最小外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            rect = [x, y, x + w, y + h]
            move_rect.append(rect)

    return move_rect

def preprocessing(img1, img2):
    kp1, des1 = SURF(img1)
    kp2, des2 = SURF(img2)
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

