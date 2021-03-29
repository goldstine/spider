# -*- coding:utf-8 -*-
"""
    author：goldstine
    version: 1.0
    date:     2021/3/29
    project name:k-means tools
    https://gist.github.com/iandanforth/5862470
"""

import math
import random

class Cluster(object):
    """
    聚类
    """
    def __init__(self,samples):
        if len(samples)==0:
            raise Exception("错误，没有样本点的空聚类!")

        #属于该聚类的样本点
        self.samples=samples

        #该聚类中样本点的维数
        self.n_dim=samples[0].n_dim

        #判断该聚类中所有的样本点维度是否相同
        for sample in samples:
            if sample.n_dim !=self.n_dim:
                raise Exception("错误：聚类中样本点的维度不一致！")

        #设置初始化的聚类中心
        self.centerid=self.cal_centerid()

    def __repr__(self):
        """
        输出对象信息
        :return:
        """
        return str(self.samples)

    def update(self,samples):
        """
        计算之前的聚类中心和更新之后的聚类中心的距离
        :param samples:
        :return:
        """
        old_centerid=self.centerid
        self.samples=samples
        self.centerid=self.cal_centerid()
        shift=get_distance(old_centerid,self.centerid)
        return shift



    def cal_centerid(self):
        """
        对一组样本点计算其中心点
        :return:
        """
        n_samples=len(self.samples)
        #获取所有样本点的坐标特征
        coords=[sample.coords for sample in self.samples]
        unzipped=zip(*coords)
        #计算每一个维度的均值
        centerid_coords=[math.fsum(d_list)/n_samples for d_list in unzipped]

        return Sample(centerid_coords)


class Sample(object):
    """
    样本点类
    """
    def __init__(self,coords):
        self.coords=coords     #样本点包含的坐标
        self.n_dim=len(coords)   #y样本点的维度

    #s输出对象信息
    def __repr__(self):
        return str(self.coords)

def get_distance(a,b):
    """
    返回样本点之间的欧氏距离
    参考：https://en.wikipedia.org/wiki/Euclidean_distance#n_dimensions
    :param a:
    :param b:
    :return:
    """
    if a.n_dim!=b.n_dim:
        #如果样本点维度不同
        raise Exception("错误：样本点维度不同，无法计算距离!")

    acc_diff=0.0
    for i in range(a.n_dim):
        square_diff=pow((a.coords[i]-b.coords[i]),2)
        acc_diff+=square_diff
    distance=math.sqrt(acc_diff)

    return distance


def gen_random_sample(n_dim,lower,upper):
    """
    生成随机样本点
    :param n_dim:
    :param lower:
    :param upper:
    :return:
    """
    sample=Sample([random.uniform(lower,upper) for _ in  range(n_dim)])
    return sample
