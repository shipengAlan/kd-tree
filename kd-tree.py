#! /usr/bin/env python
# coding=utf-8
# author: shipeng.alan
import numpy as np


class treeNode(object):
    """docstring for treeNode"""
    def __init__(self, d, s, l, r, p, lab):
        self.data = d
        self.split = s
        self.left = l
        self.right = r
        self.parent = p
        self.tag = 0
        self.label = lab

    def left(self, l):
        self.left = l

    def right(self, r):
        self.right = r


class kdTree(object):
    """docstring for kdTree"""
    def __init__(self, d):
        self.dimension = d

    def getSplitSeq(self, dataset):
        if dataset is None or dataset is []:
            return None
        # get split sequence [] by variance
        r, c = dataset.shape
        if r == 0 or c == 0:
            return []
        T_dataset = dataset.T
        var_list = []
        for i in range(c):
            var_list.append(T_dataset[i].var())
        var_arr = np.array(var_list)
        # 排序后原角标位置顺序，递增
        index_var = var_arr.argsort()
        # 递减 倒过来
        split_seq = index_var[::-1]
        return split_seq

    def createKD_Tree(self, dataset, labelset):
        split_seq = self.getSplitSeq(dataset)
        root = self.split_dimension(0, None, split_seq, dataset, labelset)
        return root

    def split_dimension(self, index, parent_tn, split_seq, dataset, labelset):
        if dataset is None:
            return None
        r, c = dataset.shape
        if r == 0 or c == 0:
            return None
        # split_seq是k各维度的分割先后顺序的index的list
        if index == len(split_seq):
            # k个维度的划分完毕重新计算划分顺序
            split_seq = self.getSplitSeq(dataset)
        index = index % len(split_seq)
        split = split_seq[index]
        index += 1
        T_dataset = dataset.T
        dimension_arr = T_dataset[split, :]
        #
        item_index = dimension_arr.argsort()
        s = dimension_arr.shape
        if s[0] == 0:
            return None
        mid_index = s[0]/2  # 中位数的下标
        #
        data = T_dataset[:, item_index[mid_index]]
        label = labelset[item_index[mid_index]]
        #
        dataset0 = None
        label0 = None
        # 子问题的数据集划分成两部分
        if mid_index >= 0:
            dataset0 = dataset[item_index[0:mid_index]]
            label0 = labelset[item_index[0:mid_index]]
        dataset1 = None
        label1 = None
        if mid_index+1 <= s[0]:
            dataset1 = dataset[item_index[mid_index+1:s[0]]]
            label1 = labelset[item_index[mid_index+1:s[0]]]
        p = treeNode(s=split, d=data, l=None, r=None, p=parent_tn, lab=label)
        p.left = self.split_dimension(index, p, split_seq, dataset0, label0)
        p.right = self.split_dimension(index, p, split_seq, dataset1, label1)
        return p

    def TravelTree(self, root):
        # 中，左，右 前序遍历
        all_node = []
        stack_list = []
        stack_list.append(root)
        all_node.append(root)
        header = ["-"]
        while len(stack_list):
            item = stack_list[-1]
            if item.tag == 0:
                print ''.join(header), stack_list[-1].data, stack_list[-1].split, stack_list[-1].label
                stack_list[-1].tag = 1
            if item.left is not None and item.left.tag == 0:
                item = item.left
                stack_list.append(item)
                header.append("-")
                all_node.append(item)
            else:
                if item.right is not None and item.right.tag == 0:
                    item = item.right
                    stack_list.append(item)
                    header.append("-")
                    all_node.append(item)
                else:
                    stack_list.pop()
                    header.pop()
        for i in range(len(all_node)):
            all_node[i].tag = 0

    def getDistance(self, node0, node1):
        """Euclidean distance"""
        return sum(abs(node0-node1)**2)**0.5

    def FindNearestNode(self, root, node):
        """Nearest Neighbor Search"""
        # 二叉查找
        current = root
        path_stack = []
        path_stack.append(current)
        while current.left is not None and current.right is not None:
            left = current.left
            right = current.right
            if current.data[current.split] >= node[current.split]:
                current = left
            else:
                current = right
            path_stack.append(current)
        minDistance = self.getDistance(path_stack[-1].data, node)
        # 回溯
        minNode = path_stack[-1]
        path_stack.pop()
        while len(path_stack):
            current = path_stack.pop()
            if abs(current.data[current.split] - node[current.split]) < minDistance:
                distance = self.getDistance(current.data, node)
                if distance < minDistance:
                    minDistance = distance
                    minNode = current
                if node[current.split] <= current.data[current.split]:
                    if current.right:
                        path_stack.append(current.right)
                else:
                    if current.left:
                        path_stack.append(current.left)

        return minNode.data, minDistance, minNode.label


if __name__ == '__main__':
    '''test kd tree'''
    kdt = kdTree(d=3)
    #root = kdt.createKD_Tree(np.array([[99,2,3],[4,5,7],[8,52,9],[10,25,2]]))
    root = kdt.createKD_Tree(np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]), np.array([1,2,3,4,5,6]))
    kdt.TravelTree(root)
    print kdt.FindNearestNode(root, np.array([2, 4.5]))