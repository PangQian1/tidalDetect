import csv
import grid.UF as gu

'''
根据拼接后的网格，提取道路起止点，道路长度，道路名称
'''
def getTidalRoadAttr(bjTopologyPath, gridTidalPath, gridPoiPath, gridSplicePath, resPath):
    #link 和 name 对应字典
    nameDict = {}
    with open(bjTopologyPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            nameDict[r[0]] = r[6]

    #网格编号和网格内的linkPeer，不少于一对，一般不会超过两对
    gridLinkPeerDict = {}
    with open(gridTidalPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            gridLinkPeerDict[r[0]] = r[1:]

    #网格编号，经度:纬度
    gridPoiDict = {}
    with open(gridPoiPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:
            gridPoiDict[r[0]] = r[1]

    f = open(resPath, 'w', encoding='utf-8', newline='')  # newline解决空行问题
    csv_writer = csv.writer(f)
    with open(gridSplicePath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:
            nameList = []
            length = len(r)
            for i in range(length):
                for j in range(len(gridLinkPeerDict[r[i]])):
                    link = gridLinkPeerDict[r[i]][j]
                    if(link != ''):#由于对预测后的网格文件进行了排序处理，Excel自动格式对齐，因此出现了空白填充
                        if(nameDict[link] not in nameList):
                            nameList.append(nameDict[link])
            csv_writer.writerow([gridPoiDict[r[0]], gridPoiDict[r[length-1]], length*100, '-'.join(nameList)])

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
            if (abs(uf.grid[j] - uf.grid[i]) == 1782 or abs(uf.grid[j] - uf.grid[i]) == 1783 or
                    abs(uf.grid[j] - uf.grid[i]) == 1784 or abs(uf.grid[j] - uf.grid[i]) == 1):
                uf.union(i, j)
                #print(str(uf.grid[i]) + ' ' + str(uf.grid[i+1]))

    dic = uf.classification()
    for key in dic.keys():
        csv_writer.writerow(dic[key])

    f.close()

if __name__ == '__main__':

    #读文件
    gridTidalPath = 'E:/G-1149/trafficCongestion/网格化/tidal/gridTidal_rnn_14.csv'
    gridPoiPath = 'E:/G-1149/trafficCongestion/网格化/gridPoi.csv'
    bjTopologyPath = "E:/G-1149/trafficCongestion/bjTopology.csv"

    #写文件
    resPath = 'E:/G-1149/trafficCongestion/网格化/tidal/name.csv'
    gridSplicePath = 'E:/G-1149/trafficCongestion/网格化/tidal/gridSplice.csv'

    #splice(gridTidalPath, gridSplicePath)
    getTidalRoadAttr(bjTopologyPath, gridTidalPath, gridPoiPath, gridSplicePath, resPath)



