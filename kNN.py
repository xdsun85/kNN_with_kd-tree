'''
构建kd树，提高kNN算法的效率
    1. 使用对象方法封装kd树
    2. 每一个结点也用对象表示，结点的相关信息保存在实例属性中
    3. 使用递归方式创建树结构以及实现树的其它逻辑结构
'''
import numpy as np
import matplotlib.pyplot as plt
from tree_plotting import createPlot

class Node(object):
    # 结点对象
    def __init__(self, elem=None, lbl=None, dim=None, par=None, l_chd=None, r_chd=None):
        self.elem = elem  # 结点的值(样本信息)
        self.lbl = lbl  # 结点的标签
        self.dim = dim  # 结点的切分的目标维度(特征)。对二维空间就是x, y两轴的坐标：0或1
        self.par = par  # 父结点
        self.l_chd = l_chd  # 左子树
        self.r_chd = r_chd  # 右子树

class KDTree(object):
    # kd树
    def __init__(self, a_list, lbl_list):
        self.__length = 0  # 不可修改
        self.__root = self.__create(a_list, lbl_list)  # 根结点, 私有属性, 不可修改

    def __create(self, e_list, lbl_list, par_node=None):
        '''
        创建kd树
        :param e_list: 需要传入一个类数组对象(行数表示样本数，列数表示特征数)
        :param lbl_list: 样本的标签
        :param par_node: 父结点
        :return: 根结点
        '''
        arr = np.array(e_list)
        m, n = arr.shape
        lbl_arr = np.array(lbl_list).reshape(m, 1)
        if m == 0:  # 样本集为空
            return None
        # 求所有特征的方差，选择最大的那个特征作为切分超平面
        var_arr = np.var(arr, axis=0)       # 获取每一个特征的方差
        max_var = np.argmax(var_arr)    # 方差最大的特征序号，以它作为切分超平面

        # 样本按最大方差特征进行升序排序后，取出中位的样本
        ft_col_rk = arr[:, max_var].argsort()       # 按最大方差特征进行升序排序，返回排序序号
        med_idx = ft_col_rk[m // 2]        # 中位样本序号
        if m == 1:  # 样本为1时，返回自身
            self.__length += 1
            return Node(dim=max_var, lbl=lbl_arr[med_idx], elem=arr[med_idx], par=par_node, l_chd=None, r_chd=None)

        # 生成结点
        node = Node(dim=max_var, lbl=lbl_arr[med_idx], elem=arr[med_idx], par=par_node, )
        # 构建有序的子树
        l_tree = arr[ft_col_rk[: m // 2]]  # 左子树
        l_lbl = lbl_arr[ft_col_rk[: m // 2]]  # 左子树标签
        l_chd = self.__create(l_tree, l_lbl, node)
        if m == 2:  # 只有左子树，无右子树
            r_chd = None
        else:
            r_tree = arr[ft_col_rk[m // 2 + 1 :]]  # 右子树
            r_lbl = lbl_arr[ft_col_rk[m // 2 + 1 :]]  # 右子树标签
            r_chd = self.__create(r_tree, r_lbl, node)
            # 左右子树递归调用自己，返回子树根结点
        node.l_chd = l_chd
        node.r_chd = r_chd
        self.__length += 1

        return node

    @property
    def length(self):
        return self.__length

    @property
    def root(self):
        return self.__root

    def transfer_dict(self, node):
        '''
        查看kd树结构
        :param node: 需要传入根结点对象
        :return: 字典嵌套格式的kd树，字典的key是self.elem, 其余项作为key的值，似下面格式
        {
            (1, 2, 3):      # 坐标，以三维为例
            {
                lbl: 1,
                dim: 0,
                l_chd:
                {
                    (2, 3, 4):
                    {
                        lbl: 1,
                        dim: 1,
                        l_chd: None,
                        r_chd: None
                    },
                r_chd:
                {
                    (4, 5, 6):
                    {
                        lbl: 1,
                        dim: 1,
                        l_chd: None,
                        r_chd: None
                    }
                }
            }
        '''
        if node == None:
            return None
        kd_dict = {}
        kd_dict[tuple(node.elem)] = {}  # 将自身值作为key
        kd_dict[tuple(node.elem)]['lbl'] = node.lbl[0]
        kd_dict[tuple(node.elem)]['dim'] = node.dim
        kd_dict[tuple(node.elem)]['par'] = tuple(node.par.elem) if node.par else None
        kd_dict[tuple(node.elem)]['l_chd'] = self.transfer_dict(node.l_chd)
        kd_dict[tuple(node.elem)]['r_chd'] = self.transfer_dict(node.r_chd)
        return kd_dict

    def transfer_dict_simple(self, node):
        '''
        查看kd树结构
        :param node: 需要传入根结点对象
        :return: 字典嵌套格式的kd树，用于作图。似下面格式
        {
            '(1, -3)':      # 坐标，以二维为例
            {
                '[1]≤':
                {
                    '(10, -6)':
                    {
                        '[0]≤':
                        {
                            '(-4, -10)':
                            {
                                '[1]≤':  '(8, -22)',
                                '[1]>': '(-6, -5)'
                            }
                        },
                        '[0]>':
                        {
                            '(17, -12)':
                            {
                                '[0]≤':  '(15, -13)'
                            },
                        }
                    },
                },
                '[1]>':
                {
                    '(6, 5)':
                    {
                        '[0]≤':
                        {
                            '(-5, 12)':
                            {
                                '[1]≤': '(-2, -1)',
                                '[1]>': '(2, 13)'
                            }
                        },
                        '[0]>':
                        {
                            '(7, 15)':
                            {
                                '[1]≤': '(14, 1)'
                            }
                        }
                    }
                }
            }
        }
        '''
        if not node:
            return None
        if not node.l_chd and not node.r_chd:
            return tuple(node.elem)
        kd_dict = {}
        kd_dict[tuple(node.elem)] = {}  # 将自身值作为key
        # if node.l_chd:    # 如果想画空枝，把这两个if注掉
        kd_dict[tuple(node.elem)]['[%s]≤' % node.dim] = self.transfer_dict_simple(node.l_chd)
        # if node.r_chd:    # 如果想画空枝，把这两个if注掉
        kd_dict[tuple(node.elem)]['[%s]>' % node.dim] = self.transfer_dict_simple(node.r_chd)

        return kd_dict

    def transfer_list(self, node, kd_list=[]):
        '''
        将kd树转化为嵌套字典的列表输出
        :param node: 需要传入根结点
        :return: 返回嵌套字典的列表，格式如下
        [{
            elem: (9, 3),
            lbl: 1,
            dim: 0,
            par: None,
            l_chd: (3, 4),
            r_chd: (11, 11)
        },
        {
            elem: (3, 4),
            lbl: 1,
            dim: 1,
            par: (9, 3),
            l_chd: (7, 0),
            r_chd: (3, 15)
        }]
        '''
        if node == None:
            return None
        elem_dict = {}
        elem_dict['elem'] = tuple(node.elem)
        elem_dict['lbl'] = node.lbl[0]
        elem_dict['dim'] = node.dim
        elem_dict['par'] = tuple(node.par.elem) if node.par else None
        elem_dict['l_chd'] = tuple(node.l_chd.elem) if node.l_chd else None
        elem_dict['r_chd'] = tuple(node.r_chd.elem) if node.r_chd else None
        kd_list.append(elem_dict)
        self.transfer_list(node.l_chd, kd_list)
        self.transfer_list(node.r_chd, kd_list)
        return kd_list

    def find_nearest_leaf(self, tgt):
        '''
        找最近叶子结点(不一定是最近邻点)
        :param tgt: 需要预测的新样本target
        :return: 距离最近的叶子结点
        '''
        tgt = np.array(tgt)
        if self.length == 0:  # 空kd树
            return None
        # 递归找离target最近的那个叶结点
        node = self.__root
        if self.length == 1:  # 只有一个样本
            return node
        while True:         # 一路找到叶子结点
            cur_dim = node.dim
            if tgt[cur_dim] <= node.elem[cur_dim]:  # 进入左子树。一路找叶子，==也朝下找
                if not node.l_chd:  # 左子树为空，返回自身
                    return node
                node = node.l_chd
            else:    # 进入右子树    tgt[cur_dim] > node.elem[cur_dim]:
                if not node.r_chd:  # 右子树为空，返回自身
                    return node
                node = node.r_chd

    def knn_algo(self, tgt, k=1):
        '''
        找到距离测试样本最近的前k个样本
        :param tgt: 测试样本
        :param k: knn算法参数，定义需要参考的最近点数量，一般为1-5
        :return: 返回前k个样本的最大分类标签
        '''
        if self.length <= k:
            lbl_dict = {}
            # 获取所有lbl的数量
            for node in self.transfer_list(self.root):
                lbl_dict[node['lbl']] = lbl_dict.setdefault(node, 0) + 1
            sorted_lbl = sorted(lbl_dict.items(), key=lambda kv: kv[1], reverse=True)  # 给标签排降序
            return sorted_lbl[0][0]

        tgt = np.array(tgt)
        leaf = self.find_nearest_leaf(tgt)  # 找到最近的叶子结点
        if not leaf:  # 空树
            return None
        print('靠近点%s最近的叶结点为：%s' % (tgt, leaf.elem))
        dist = sum((tgt - leaf.elem) ** 2) ** 0.5  # 最近点与目标点的距离
        # 返回上一个父结点，判断以测试点为圆心，dist为半径的圆是否与父结点分隔超平面相割，若相割，则说明父结点的另一个子树可能存在更近的点
        nodes_list = [[dist, tuple(leaf.elem), leaf.lbl[0]]]  # 需要将距离与结点一起保存起来
        print('leaf', leaf.elem)

        # 回到父结点
        cur = leaf      # 从叶子结点开始
        while cur != self.root:
            par = cur.par      # 当前点还是cur, 还没上爬
            print('cur', cur.elem)
            print('par', par.elem)
            # 计算测试点与父结点的距离，与上面距离做比较
            par_dist = sum((tgt - par.elem) ** 2) ** 0.5
            print('par_dist', par_dist, tgt, par.elem)
            nodes_list.sort()
            far_dist = nodes_list[-1][0] if len(nodes_list) <= k else nodes_list[k - 1][0]
            print('far_dist', far_dist)
            print('进if:', len(nodes_list), far_dist, par_dist)
            if len(nodes_list) < k or par_dist < far_dist:  # 看有无必要再加par
                print('nodes_list加前', nodes_list)
                nodes_list.append([par_dist, tuple(par.elem), par.lbl[0]])
                print('nodes_list加后', nodes_list)
                nodes_list.sort()
                print('nodes_list排序后', nodes_list)
                far_dist = nodes_list[-1][0] if len(nodes_list) <= k else nodes_list[k - 1][0]
                print('far_dist', far_dist)

            # 判断父结点的另一个子树与结点列表中最大的距离构成的圆是否有交集
            print('超平面判断', tgt[par.dim], par.elem[par.dim], far_dist)
            if len(nodes_list) < k or far_dist > abs(tgt[par.dim] - par.elem[par.dim]):  # 说明父结点的另一个子树与圆有交集
                print('触发')
                # 说明父结点的另一子树区域与圆有交集
                other_chd = par.l_chd if par.l_chd != cur else par.r_chd  # 找另一个子树
                print('other_chd', other_chd.elem)
                # 测试点在该子结点超平面的左侧
                if other_chd:
                    print('搜哪边', tgt[par.dim], par.elem[par.dim])
                    if tgt[par.dim] <= par.elem[par.dim]:
                        self.l_srch(tgt, other_chd, nodes_list, k)
                    else:
                        self.r_srch(tgt, other_chd, nodes_list, k)  # 测试点在该子结点平面的右侧
            else:
                print('未触发')

            cur = par  # 爬一层
            print('爬一层\n')

        # 接下来取出前k个元素中最大的分类标签
        print('over')
        lbl_dict = {}
        nodes_list = nodes_list[: k]
        # 获取所有lbl的数量
        for node in nodes_list:
            lbl_dict[node[2]] = lbl_dict.setdefault(node[2], 0) + 1
        sorted_lbl = sorted(lbl_dict.items(), key=lambda kv: kv[1], reverse=True)  # 给标签排序
        return sorted_lbl[0][0], nodes_list

    def l_srch(self, tgt, node, nodes_list, k):
        '''
        按左中右顺序遍历子树结点，返回结点列表
        :param tgt: 传入的测试样本
        :param node: 子树结点
        :param nodes_list: 结点列表
        :param k: 搜索比较的结点数量
        :return: 结点列表
        '''
        print('left', node.elem)
        print('nodes_list排序前', nodes_list)
        nodes_list.sort()
        print('nodes_list排序后', nodes_list)
        far_dist = nodes_list[-1][0] if len(nodes_list) <= k else nodes_list[k - 1][0]
        print('far_dist', far_dist)
        if not node.l_chd and not node.r_chd:  # 叶结点
            dist = sum((tgt - node.elem) ** 2) ** 0.5
            print('dist', dist, tgt, node.elem)
            if len(nodes_list) < k or far_dist > dist:     #
                nodes_list.append([dist, tuple(node.elem), node.lbl[0]])
                print('nodes_list加', nodes_list)
            return
        self.l_srch(tgt, node.l_chd, nodes_list, k)
        # 每次进行比较前都更新nodes_list数据
        nodes_list.sort()
        print('nodes_list排序后', nodes_list)
        far_dist = nodes_list[-1][0] if len(nodes_list) <= k else nodes_list[k - 1][0]
        print('far_dist', far_dist)
        dist = sum((tgt - node.elem) ** 2) ** 0.5
        print('dist', dist)
        # 比较根结点
        if len(nodes_list) < k or far_dist > dist:
            nodes_list.append([dist, tuple(node.elem), node.lbl[0]])
            print('nodes_list加', nodes_list)
            nodes_list.sort()
            print('nodes_list', nodes_list)
            far_dist = nodes_list[-1][0] if len(nodes_list) <= k else nodes_list[k - 1][0]
            print('far_dist', far_dist)
        # 右子树
        print('超平面判断', node.elem, node.dim, tgt[node.dim], node.elem[node.dim])
        if len(nodes_list) < k or far_dist > abs(tgt[node.dim] - node.elem[node.dim]):  # 需要搜索右子树
            print('触发')
            if node.r_chd:
                self.l_srch(tgt, node.r_chd, nodes_list, k)
        else:
            print('未触发')

        return nodes_list

    def r_srch(self, tgt, node, nodes_list, k):
        '''
        按右中左顺序遍历子树结点
        :param tgt: 测试的样本点
        :param node: 子树结点
        :param nodes_list: 结点列表
        :param k: 搜索比较的结点数量
        :return: 结点列表
        '''
        print('right', node.elem)
        print('nodes_list', nodes_list)
        nodes_list.sort()
        print('nodes_list', nodes_list)
        far_dist = nodes_list[-1][0] if len(nodes_list) <= k else nodes_list[k - 1][0]
        print('far_dist', far_dist)
        if not node.l_chd and not node.r_chd:  # 叶结点
            dist = sum((tgt - node.elem) ** 2) ** 0.5
            print('dist', dist)
            if len(nodes_list) < k or far_dist > dist:     #
                nodes_list.append([dist, tuple(node.elem), node.lbl[0]])
                print('nodes_list加', nodes_list)
            return
        if node.r_chd:
            self.r_srch(tgt, node.r_chd, nodes_list, k)

        nodes_list.sort()
        far_dist = nodes_list[-1][0] if len(nodes_list) <= k else nodes_list[k - 1][0]
        print('far_dist', far_dist)
        dist = sum((tgt - node.elem) ** 2) ** 0.5
        print('dist', dist)
        # 比较根结点
        if len(nodes_list) < k or far_dist > dist:     #
            nodes_list.append([dist, tuple(node.elem), node.lbl[0]])
            print('nodes_list加', nodes_list)
            nodes_list.sort()
            print('nodes_list排序后', nodes_list)
            far_dist = nodes_list[-1][0] if len(nodes_list) <= k else nodes_list[k - 1][0]
            print('far_dist', far_dist)
        # 左子树
        print('超平面判断', tgt[node.dim], node.elem[node.dim], far_dist)
        if len(nodes_list) < k or far_dist > abs(tgt[node.dim] - node.elem[node.dim]):  # 需要搜索左子树
            print('触发')
            self.r_srch(tgt, node.l_chd, nodes_list, k)
        else:
            print('未触发')

        return nodes_list

if __name__ == '__main__':
    # arr = np.array([[19, 2], [7, 0], [13, 5], [3, 15], [3, 4], [3, 2], [8, 9], [9, 3], [17, 15], [11, 11]])
    arr = np.array([[6, 5], [1, -3], [-6, -5], [-4, -10], [-2, -1], [-5, 12], [2, 13], [17, -12], [8, -22], [15, -13], [10, -6], [7, 15], [14, 1]])
    # arr = np.array([[6.27, 5.5], [1.24, -2.86], [17.05, -12.79], [-6.88, -5.4], [-2.96, -0.5], [7.75, -22.68], [10.8, -5.03], [-4.6, -10.55], [-4.96, 12.61], [1.75, 12.26], [15.31, -13.16], [7.83, 15.7], [14.63, -0.35]])
    # arr = np.random.randint(0, 20, size=(10000, 2))
    lbl = np.array([[0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
    # lbl = np.random.randint(0, 3, size=(10000, 1))

    kd_tree = KDTree(arr, lbl)
    kd_dict = kd_tree.transfer_dict(kd_tree.root)
    print('kd_dict', kd_dict)
    kd_dict_simple = kd_tree.transfer_dict_simple(kd_tree.root)
    print('kd_dict_simple', kd_dict_simple)
    createPlot(kd_dict_simple)
    kd_list = kd_tree.transfer_list(kd_tree.root)
    print('kd_list', kd_list)

    # tgt = np.array([-1, -5])
    tgt = np.array([-2, 5])
    # tgt = np.array([10, -6])
    # tgt = np.array([9, -22])
    k = 3
    lbl, nodes_list = kd_tree.knn_algo(tgt, k=k)
    print('点%s的最接近的前%s个点为：%s' % (tgt, k, nodes_list))
    print('点%s的标签：%s' % (tgt, lbl))

    if len(arr[0]) == 2:       # 二维的话，画图
        plt.scatter(arr[:, 0], arr[:, 1])
        for pt in arr:
            plt.text(pt[0] + 0.8, pt[1] - 0.5, str(pt))

        plt.scatter(tgt[0], tgt[1], color='C3', marker='*')
        plt.text(tgt[0] + 0.8, tgt[1] - 0.5, str(tgt))

        pts_nbr = np.array([j[1] for j in nodes_list])
        plt.scatter(pts_nbr[:, 0], pts_nbr[:, 1], color='C2')

        plt.axis('equal')
        plt.show()
