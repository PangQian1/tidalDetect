import csv

class UnionFind:
    # 记录连通分量个数 int
    count = 0
    # 存储若⼲棵树 int[]
    parent = []
    # 记录树的“重量”,即每棵树有几个节点 int[]
    size = []
    #记录网格编号和索引i的对应关系
    grid = []

    def __init__(self, path):
        #根据读入的文件初始化森林
        with open(path, 'r') as file:
            reader = csv.reader(file)
            for r in reader:  # r是一个list
                self.size.append(1) #初始每棵树都只有根节点
                self.parent.append(self.count)
                self.grid.append(int(r[0]))
                self.count = self.count + 1
                
    # 将 p 和 q 连通 
    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if (rootP == rootQ):
            return

        # ⼩树接到⼤树下⾯，较平衡
        if (self.size[rootP] > self.size[rootQ]):
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
        else:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]

        self.count = self.count - 1

    #判断 p 和 q 是否互相连通
    def connected(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        # 处于同⼀棵树上的节点，相互连通
        return rootP == rootQ

    #返回节点 x 的根节点
    def find(self, x):
        while (self.parent[x] != x):
            # 进⾏路径压缩
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def getCount(self):
        return self.count

    #遍历所有的节点，按照连通性对grid内容分类
    def classification(self):
        dict = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if(root in dict.keys()):
                list = dict[root]
                list.append(self.grid[i])
                dict[root] = list
            else:
                dict[root] = [self.grid[i]]

        return dict
