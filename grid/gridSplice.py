import csv
import grid.UF as gu

#根据网格获取道路名称
def getName(bjTopologyPath, gridLinkPeerPath, gridSplicePath, resPath):
    #link 和 name 对应字典
    nameDict = {}
    with open(bjTopologyPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            nameDict[r[0]] = r[6]

    gridLinkPeerDict = {}
    with open(gridLinkPeerPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            gridLinkPeerDict[r[0]] = r[1:]

    f = open(resPath, 'w', encoding='utf-8', newline='')  # newline解决空行问题
    csv_writer = csv.writer(f)
    with open(gridSplicePath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:
            nameList = []
            for i in range(len(r)):
                for j in range(len(gridLinkPeerDict[r[i]])):
                    link = gridLinkPeerDict[r[i]][j]
                    nameList.append(nameDict[link])
            csv_writer.writerow(nameList)

    f.close()
"""
union-find算法
对已经提取出来的具有潮汐现象的网格进行“连通性”处理，以拼接成一段完整的道路
"""
def splice(gridPath, gridSplicePath):
    f = open(gridSplicePath, 'w', encoding='utf-8', newline='')  # newline解决空行问题
    csv_writer = csv.writer(f)
    uf = gu.UnionFind(gridPath)
    for i in range(len(uf.grid)-1):
        for j in range(i+1, len(uf.grid)):
            if (abs(uf.grid[j] - uf.grid[i]) == 1 or abs(uf.grid[j] - uf.grid[i]) == 1782 or
                    abs(uf.grid[j] - uf.grid[i]) == 1783 or abs(uf.grid[j] - uf.grid[i]) == 1784):
                uf.union(i, j)
                #print(str(uf.grid[i]) + ' ' + str(uf.grid[i+1]))

    dic = uf.classification()
    for key in dic.keys():
        csv_writer.writerow(dic[key])

    f.close()

if __name__ == '__main__':

    #读文件
    gridLinkPeerPath = 'E:/G-1149/trafficCongestion/网格化/gridLinkPeer_14.csv'
    gridPath = 'E:/G-1149/trafficCongestion/网格化/tidal/rnn_14.csv'
    bjTopologyPath = "E:/G-1149/trafficCongestion/bjTopology.csv"

    #写文件
    resPath = 'E:/G-1149/trafficCongestion/网格化/tidal/name.csv'
    gridSplicePath = 'E:/G-1149/trafficCongestion/网格化/tidal/gridSplice.csv'

    splice(gridPath, gridSplicePath)
    getName(bjTopologyPath, gridLinkPeerPath, gridSplicePath, resPath)



