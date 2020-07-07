import networkx as nx
import matplotlib.pyplot as plt
import csv

if __name__ == '__main__':
    g = nx.Graph()  # 创建空的无向图

    filePath = 'C:\\Users\\98259\\Desktop\\6.9学习相关文档\样本数据\\fiftMin\\sample - 副本.csv'
    count = 0
    with open(filePath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:
            for i in range(63):
                g.add_edge(r[i], r[i+1])

            count += 1
            if(count == 2):
                print(count)
                break

    #可视化展示
    nx.draw(g)
    plt.show()


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