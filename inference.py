# coding:utf-8
import os
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure

from src.loftr import LoFTR, default_cfg

import ransac
import blend
import k_means
from typing import List, Tuple, Union


def show_image(image: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).show()


# 变换、拼接 源:https://blog.csdn.net/qq_44019424/article/details/106010362
def drawMatches(img_left, img_right, kps_left, kps_right):
    H, status = cv2.findHomography(kps_right, kps_left, cv2.RANSAC)
    # 获取图片宽度和高度
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]
    """对imgB进行透视变换
    由于透视变换会改变图片场景的大小，导致部分图片内容看不到
    所以对图片进行扩展:高度取最高的，宽度为两者相加"""
    image = np.zeros((max(h_left, h_right), w_left + w_right, 3), dtype='uint8')
    # 初始化
    image[0:h_left, 0:w_left] = img_right
    """利用以获得的单应性矩阵进行变透视换"""
    image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))  # (w,h
    """将透视变换后的图片与另一张图片进行拼接"""
    image[0:h_left, 0:w_left] = img_left

    max_w = 2 * w_left
    for i in range(w_left, max_w):  # 计算右边界
        if np.max(image[:, i]) == 0:
            max_w = i
            break
    print(max_w)
    return image[:, :max_w]


# 你发的github上的代码，可以用他的拼接，但感觉效果也差不多
class Stitcher:
    def __init__(self, image1: np.ndarray, image2: np.ndarray):
        """输入图像和匹配，对图像进行拼接
        目前采用简单矩阵匹配和平均值拼合

        Args:
            image1 (np.ndarray): 图像一
            image2 (np.ndarray): 图像二
            matcher (Matcher): 匹配结果
            use_kmeans (bool): 是否使用kmeans 优化点选择
        """

        self.image1 = image1
        self.image2 = image2
        self.M = np.eye(3)

        self.image = None

    def stich(self, p1s, p2s, show_result=True, show_match_point=False, use_new_match_method=False, use_partial=False,
              use_gauss_blend=True):
        """对图片进行拼合

            show_result (bool, optional): Defaults to True. 是否展示拼合图像
            show_match_point (bool, optional): Defaults to True. 是否展示拼合点
        """
        # 得到匹配点
        self.image_points1, self.image_points2 = p1s, p2s

        if use_new_match_method:
            self.M = ransac.GeneticTransform(self.image_points1, self.image_points2).run()
        else:
            self.M, _ = cv2.findHomography(
                self.image_points1, self.image_points2, method=cv2.RANSAC)

        print("Good points and average distance: ", ransac.GeneticTransform.get_value(
            self.image_points1, self.image_points2, self.M))

        left, right, top, bottom = self.get_transformed_size()
        # print(self.get_transformed_size())
        width = int(max(right, self.image2.shape[1]) - min(left, 0))
        height = int(max(bottom, self.image2.shape[0]) - min(top, 0))
        print(width, height)
        # width, height = min(width, 10000), min(height, 10000)
        if width * height > 8000 * 5000:
            # raise MemoryError("Too large to get the combination")
            factor = width * height / (8000 * 5000)
            width = int(width / factor)
            height = int(height / factor)

        if use_partial:
            self.partial_transform()

        # 移动矩阵
        self.adjustM = np.array(
            [[1, 0, max(-left, 0)],  # 横向
             [0, 1, max(-top, 0)],  # 纵向
             [0, 0, 1]
             ], dtype=np.float64)
        # print('adjustM: ', adjustM)
        self.M = np.dot(self.adjustM, self.M)
        transformed_1 = cv2.warpPerspective(
            self.image1, self.M, (width, height))
        transformed_2 = cv2.warpPerspective(
            self.image2, self.adjustM, (width, height))

        self.image = self.blend(transformed_1, transformed_2, use_gauss_blend=use_gauss_blend)

        if show_match_point:
            for point1, point2 in zip(self.image_points1, self.image_points2):
                point1 = self.get_transformed_position(tuple(point1))
                point1 = tuple(map(int, point1))
                point2 = self.get_transformed_position(tuple(point2), M=self.adjustM)
                point2 = tuple(map(int, point2))

                # cv2.line(self.image, point1, point2, random.choice(colors), 3)
                cv2.circle(self.image, point1, 10, (20, 20, 255), 5)
                cv2.circle(self.image, point2, 8, (20, 200, 20), 5)
        # if show_result:
        #     show_image(self.image)

    def blend(self, image1: np.ndarray, image2: np.ndarray, use_gauss_blend=True) -> np.ndarray:
        """对图像进行融合

        Args:
            image1 (np.ndarray): 图像一
            image2 (np.ndarray): 图像二

        Returns:
            np.ndarray: 融合结果
        """

        mask = self.generate_mask(image1, image2)
        print("Blending")
        if use_gauss_blend:
            result = blend.gaussian_blend(image1, image2, mask, mask_blend=10)
        else:
            result = blend.direct_blend(image1, image2, mask, mask_blend=0)

        return result

    def generate_mask(self, image1: np.ndarray, image2: np.ndarray):
        """生成供融合使用的遮罩，由变换后图像的垂直平分线来构成分界线

        Args:
            shape (tuple): 遮罩大小

        Returns:
            np.ndarray: 01数组
        """
        print("Generating mask")
        # x, y
        center1 = self.image1.shape[1] / 2, self.image1.shape[0] / 2
        center1 = self.get_transformed_position(center1)
        center2 = self.image2.shape[1] / 2, self.image2.shape[0] / 2
        center2 = self.get_transformed_position(center2, M=self.adjustM)
        # 垂直平分线 y=-(x2-x1)/(y2-y1)* [x-(x1+x2)/2]+(y1+y2)/2
        x1, y1 = center1
        x2, y2 = center2

        # note that opencv is (y, x)
        def function(y, x, *z):
            return (y2 - y1) * y < -(x2 - x1) * (x - (x1 + x2) / 2) + (y2 - y1) * (y1 + y2) / 2

        mask = np.fromfunction(function, image1.shape)

        # mask = mask&_i2+mask&i1+i1&_i2
        mask = np.logical_and(mask, np.logical_not(image2)) \
               + np.logical_and(mask, image1) \
               + np.logical_and(image1, np.logical_not(image2))

        return mask

    def get_transformed_size(self) -> Tuple[int, int, int, int]:
        """计算形变后的边界
        计算形变后的边界，从而对图片进行相应的位移，保证全部图像都出现在屏幕上。

        Returns:
            Tuple[int, int, int, int]: 分别为左右上下边界
        """

        conner_0 = (0, 0)  # x, y
        conner_1 = (self.image1.shape[1], 0)
        conner_2 = (self.image1.shape[1], self.image1.shape[0])
        conner_3 = (0, self.image1.shape[0])
        points = [conner_0, conner_1, conner_2, conner_3]

        # top, bottom: y, left, right: x
        top = min(map(lambda x: self.get_transformed_position(x)[1], points))
        bottom = max(
            map(lambda x: self.get_transformed_position(x)[1], points))
        left = min(map(lambda x: self.get_transformed_position(x)[0], points))
        right = max(map(lambda x: self.get_transformed_position(x)[0], points))

        return left, right, top, bottom

    def get_transformed_position(self, x: Union[float, Tuple[float, float]], y: float = None, M=None) -> Tuple[
        float, float]:
        """求得某点在变换矩阵（self.M）下的新坐标

        Args:
            x (Union[float, Tuple[float, float]]): x坐标或(x,y)坐标
            y (float, optional): Defaults to None. y坐标，可无
            M (np.ndarray, optional): Defaults to None. 利用M进行坐标变换运算

        Returns:
            Tuple[float, float]:  新坐标
        """

        if isinstance(x, tuple):
            x, y = x
        p = np.array([x, y, 1])[np.newaxis].T
        if M is not None:
            M = M
        else:
            M = self.M
        pa = np.dot(M, p)
        return pa[0, 0] / pa[2, 0], pa[1, 0] / pa[2, 0]


# 定义一个类，方便调用
class loftrInfer(object):
    ''' 初始化之后，仅需调用run函数 '''

    def __init__(self, model_path="weights/indoor_ds.ckpt"):
        '''
            初始化，输入参数:
                    model_path: 模型地址
        '''
        self.matcher = LoFTR(config=default_cfg)  # 初始化模型
        self.matcher.load_state_dict(torch.load(model_path)['state_dict'])  # 下载训练好的模型文件，可选indoor_ds 、outdoor_ds
        self.matcher = self.matcher.eval().cuda()  # cuda验证

    def _infer_run(self, img0_raw, img1_raw):
        '''
            推理单对图片，输入参数:
                    img0_raw 、img1_raw    numpy.ndarray类型，单通道图像
                    返回值:
                    np_result/False     False 或 (n,5)推理结果，numpy.ndarray类型； 格式为(p1x,p1y,p2x,p2y,conf)
                    
        '''
        img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.  # 转torch格式，cuda ，归一到0-1
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
        batch = {'image0': img0, 'image1': img1}  # 模型输入为字典，加载输入

        # Inference with LoFTR and get prediction 开始推理
        with torch.no_grad():
            self.matcher(batch)  # 网络推理
            mkpts0 = batch['mkpts0_f'].cpu().numpy()  # (n,2) 0的结果 -特征点
            mkpts1 = batch['mkpts1_f'].cpu().numpy()  # (n,2) 1的结果 -特征点
            mconf = batch['mconf'].cpu().numpy()  # (n,)      置信度

        # 筛选，需要四个以上的匹配点才能得到单应性矩阵
        if mconf.shape[0] < 4:
            return False
        mconf = mconf[:, np.newaxis]  # 末尾增加新维度
        np_result = np.hstack((mkpts0, mkpts1, mconf))  # 水平拼接
        print(np_result.shape)
        list_result = list(np_result)

        def key_(a):
            return a[-1]

        list_result.sort(key=key_, reverse=True)  # 按得分从大到小排序
        np_result = np.array(list_result)
        return np_result

    def _points_filter(self, np_result, lenth=200, use_kmeans=True):
        '''
            进行特征值筛选，输入参数:
                    np_result  推理结果(n,5)
                    lenth       -1   不进行筛选，取全部
                                >0  取前nums个
                    use_kmeans   bool类型: 0 - 不使用
                                        1 -  使用聚类，取最多一类
        '''
        '''' 本来想直接取前多少个进行矩阵运算，但发现聚类后好些 '''
        lenth = min(lenth, np_result.shape[0])  # 选最大200个置信度较大的点对
        if lenth < 4: lenth = 4

        mkpts0 = np_result[:lenth, :2].copy()
        mkpts1 = np_result[:lenth, 2:4].copy()

        if use_kmeans:
            use_mkpts0, use_mkpts1 = k_means.get_group_center(mkpts0, mkpts1)  # 聚类，并返回同一类最多元素的匹配点
            print("一共：", mkpts0.shape)
            print("筛选与聚类后:", use_mkpts0.shape)
            if use_mkpts0.shape[0] < 4:
                return mkpts0, mkpts1
            return use_mkpts0, use_mkpts1
        return mkpts0, mkpts1

    def _draw_matchs(self, img1, img2, p1s, p2s, mid_space=10, if_save=False):
        '''
            画匹配点并显示，分别输入图像、特征点  ; mid_space间隔
            输入参数: 
                    img1,img2 彩色图像; p1s,p2s 分别的特征点位置
                    mid_space 左右图显示间隔
                    if_save 保存否
        '''
        h, w = img1.shape[:2]
        show = cv2.resize(img1, (2 * w + mid_space, h))
        show.fill(0)
        show[:, :w] = img1.copy()
        show[:, w + 10:] = img2.copy()
        p1s = p1s.astype(np.int)
        p2s = p2s.astype(np.int)

        for i in range(p1s.shape[0]):
            p1 = tuple(p1s[i])
            p2 = (p2s[i][0] + w + mid_space, p2s[i][1])
            cv2.line(show, p1, p2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
                     1)  # 画线
        cv2.namedWindow('show', 2)
        cv2.imshow('show', show)
        if if_save:
            cv2.imwrite('save.jpg', show)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def run(self, img0_bgr, img1_bgr, lenth=200, use_kmeans=True, if_draw=True, if_save=False, stitch_method=1):
        '''
            只需要调用该函数，完成推理+拼接
            输入参数 : 
                img0_bgr , img1_bgr  彩色图像，默认左右
                lenth       -1   不进行筛选，取全部
                            >0  取前nums个
                use_kmeans   bool类型: 0 - 不使用
                                    1 -  使用聚类，取最多一类
                if_draw      bool类型  是否绘制特征点匹配图像
                if_save      bool类型  是否保存特征点匹配图像
                stitch_method    拼接方法选择， 0 为l2n方法 ； 其他为简单方法
            返回值:
                image  拼接图像
        '''
        img0_bgr = cv2.resize(img0_bgr, (640, 480))  # 统一尺寸为640x480
        img1_bgr = cv2.resize(img1_bgr, (640, 480))

        img0_raw = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)  # 转灰度，网络输入的是单通道图
        img1_raw = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)

        np_result = self._infer_run(img0_raw, img1_raw)  # 推理
        if np_result is False:
            print("特征点数量不够！！！")
            return False
        mkpts0, mkpts1 = self._points_filter(np_result, lenth=lenth, use_kmeans=use_kmeans)  # 特征点筛选
        if if_draw:  # 显示匹配点对
            self._draw_matchs(img0_bgr, img1_bgr, mkpts0, mkpts1, mid_space=10, if_save=if_save)
        if stitch_method == 0:
            '''拼接,github方法，你推荐的l2net中的方法'''
            stitcher = Stitcher(img0_bgr, img1_bgr)
            stitcher.stich(p1s=mkpts0, p2s=mkpts1, use_partial=False, use_new_match_method=0, use_gauss_blend=0)
            image = (stitcher.image).copy()
        else:
            # 简单方法，变换右图进行拼接
            image = drawMatches(img0_bgr, img1_bgr, mkpts0, mkpts1)
        return image


if __name__ == "__main__":
    # 调用实例
    testInfer = loftrInfer(model_path="weights/indoor_ds.ckpt")
    img1_pth = "assets/scannet_sample_images/scene0768_00_frame-001095.jpg"
    img0_pth = "assets/scannet_sample_images/scene0768_00_frame-003435.jpg"
    img0_bgr = cv2.imread(img0_pth)  # 读取图片，bgr格式
    img1_bgr = cv2.imread(img1_pth)

    result = testInfer.run(img0_bgr, img1_bgr, lenth=200, use_kmeans=True, if_draw=True, if_save=False,
                           stitch_method=0)
    cv2.imshow('show', result)
    cv2.waitKey()
