import networkx as nx
import matplotlib.pyplot as plt
import csv

def createGraphByRMid(filePath):
    g = nx.DiGraph()  # 创建空的有向图
    preAdjDict = {} #存储了每个节点的前驱节点，每个key(node)对应一个list，解决有向图中邻接点不保存前驱节点的问题
    with open(filePath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:
            # if(judgeRoadAttr(r[2])):
            #     print(r[0]+","+r[2])
            #     continue
            g.add_node(r[5])
            g.add_node(r[6])
            eNode = None
            sNode = None
            if(r[4] == '2'):
                g.add_edge(r[5], r[6], linkID=r[0], kindNum=r[1], kind=r[2], width=r[3], dir=r[4], len=r[7], laneNum=r[8])
                eNode = r[6]
                sNode = r[5]
            else:
                g.add_edge(r[6], r[5], linkID=r[0], kindNum=r[1], kind=r[2], width=r[3], dir=r[4], len=r[7], laneNum=r[8])
                eNode = r[5]
                sNode = r[6]
            if (eNode in preAdjDict.keys()):
                list = preAdjDict[eNode]
                list.append(sNode)
                preAdjDict[eNode] = list
            else:
                list = [sNode]
                preAdjDict[eNode] = list
    return g, preAdjDict

#判断道路种别代码，是否含有我们不需要的link类型，如果是，返回true，否则返回false
def judgeRoadAttr(kind):
    otherKindList = ['0a','0b','0e','01','03','05','13','11']
    kindList = kind.split('|')
    for item in kindList:
        if(item[:2] == '00'):
            return True
        if(item[2:] in otherKindList):
            return True
    return False

def reversalGraph(g, preAdjDict):
    revG = nx.DiGraph()
    nodes = g.nodes()
    for node in nodes:
        adjNodeDict = g[node]  # 所有邻接点
        for key in adjNodeDict:
            edge = adjNodeDict[key]
            revG.add_node(edge['linkID'], kindNum=edge['kindNum'], kind=edge['kind'], width=edge['width'],
                          dir=edge['dir'], len=edge['len'], laneNum=edge['laneNum'])
            #有一些node可能没有前驱
            if(node in preAdjDict.keys()):
                for preNode in preAdjDict[node]:
                    revG.add_edge(g[preNode][node]['linkID'], edge['linkID'])
    return revG


if __name__ == '__main__':
    #filePath = 'D:\\program\\congestion\\test.csv'
    filePath = 'D:\\program\\congestion\\originMapWithOutBiDirec.csv'
    g, preAdjDict = createGraphByRMid(filePath)
    print(g['531572'])
    print(preAdjDict['531574'])
    print(g['531572']['531574'])
    revG = reversalGraph(g, preAdjDict) #得到以边为节点的有向图
    print(revG['19468718'])
    print(revG.nodes['19468716'])
    #
    # # 1. 创建文件对象
    # f = open('D:\\program\\congestion\\intersectionLink.csv', 'w', encoding='utf-8', newline ='')
    # # 2. 基于文件对象构建 csv写入对象
    # csv_writer = csv.writer(f)
    #
    # #查看交叉点link周边都是什么类型的link
    # kindDict = {}
    # maxAdjNum = 0 #最大邻居数
    # for node in revG.nodes():
    #     if(maxAdjNum < len(revG[node])):
    #         maxAdjNum = len(revG[node])
    #     kind = revG.nodes[node]['kind']
    #     kindList = kind.split('|')
    #     for item in kindList:
    #         if(item == '0604'):
    #             for key in revG[node]:
    #                 adjEdgeKind = revG.nodes[key]['kind']
    #                 if adjEdgeKind in kindDict.keys():
    #                     value = kindDict[adjEdgeKind]
    #                     value += 1
    #                     kindDict[adjEdgeKind] = value
    #                 else:
    #                     kindDict[adjEdgeKind] = 1
    #     #print(revG[node])
    # for key in kindDict:
    #     csv_writer.writerow([key, str(kindDict[key])])
    #
    # # 5. 关闭文件
    # f.close()
    # print('节点个数：' + str(len(revG)))
    # print('最大邻居数：' + str(maxAdjNum))
    # # #print(g.edges())
    # # print(revG.nodes())
    # # #print('imformation of all nodes', revG.nodes.data())

    #打印邻接点
    #print(g['531572'])
    #打印边的信息
    #print(g['531572']['531574'])

    #可视化展示
    # nx.draw(revG)
    # plt.show()




# g.add_node(1)
# #一次添加一个节点列表
# g.add_nodes_from([2,3,4])
# g.nodes()
# g.add_edge(2,3)
# g.add_edges_from([(1,2),(1,3)])
# g.edges()

#print('information of all nodes',g.nodes.data())
#print('information of all nodes',g.edges.data())


# #将图转成邻接矩阵
# B = nx.adjacency_matrix(g)
# print(B)
# B1 = B.todense()
# print(B1)

# #可视化展示
# nx.draw(g)
# plt.show()