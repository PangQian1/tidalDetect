import csv

'''
计算结果有待验证
获取长路段的status 
'''
def longRoad(linkStatusPath, roadPath, linkAttrPath, resPath):
    #link 和 length 对应字典
    lengthDict = {}
    with open(linkAttrPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            lengthDict[r[0]] = r[4]

    # link 和 status 对应字典
    linkStatusDict = {}
    with open(linkStatusPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            linkStatusDict[r[0]] = r[1:]

    statusNum = 16
    f = open(resPath, 'w', encoding='utf-8', newline='')  # newline解决空行问题
    csv_writer = csv.writer(f)
    noData = {}
    with open(roadPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:
            list = []
            for i in range(statusNum):
                statusSum = 0.0
                totalLen = 0.0
                for j in range(len(r)):
                    if(r[j] in linkStatusDict):
                        totalLen = totalLen + float(lengthDict[r[j]])
                        statusSum = statusSum + (float(lengthDict[r[j]]) * float(linkStatusDict[r[j]][i]))
                    else:
                        noData[r[j]] = 1
                list.append(statusSum/totalLen)
            csv_writer.writerow(list)

    f.close()
    print('缺失的link数目' + str(len(noData)))

'''
根据mapInfo提取的局部道路，提取上下行分开的平行路段
注意方向根据道路的不同，会有所差别
'''
def getParallelLink(longRoadMidPath, linkAttrPath, roadPath, dir1, dir2, dir3, dir4):
    #link 和 DIR 对应字典
    dirDict = {}
    with open(linkAttrPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            dirDict[r[0]] = r[1]

    f = open(roadPath, 'w', encoding='utf-8', newline='')  # newline解决空行问题
    csv_writer = csv.writer(f)
    resListFir = []
    resListSec = []
    with open(longRoadMidPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            linkID = r[1].replace('"', '')
            if(dirDict[linkID] == dir1 or dirDict[linkID] == dir2):
                resListFir.append(linkID)
            elif(dirDict[linkID] == dir3 or dirDict[linkID] == dir4):
                resListSec.append(linkID)
            else:
                print('error')
    csv_writer.writerow(resListFir)
    csv_writer.writerow(resListSec)
    f.close()

def getJingzangSta(linkStatusPath, roadPath, resPath):
    # link 和 status 对应字典
    linkStatusDict = {}
    with open(linkStatusPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            linkStatusDict[r[0]] = r[1:]

    # linkPeer list
    linkPeer = []
    with open(roadPath, 'r') as file:
        reader = csv.reader(file)
        list = []
        for r in reader:  # r是一个list
            list.append(r)
        for i in range(34):
            linkPeer.append([list[0][i], list[1][i]])

    f = open(resPath, 'w', encoding='utf-8', newline='')  # newline解决空行问题
    csv_writer = csv.writer(f)
    for i in linkPeer:
        if(i[0] in linkStatusDict and i[1] in linkStatusDict):
            list = linkStatusDict[i[0]] + linkStatusDict[i[1]]
            csv_writer.writerow(list)

    f.close()


if __name__ == '__main__':
    date = '14'
    roadName = '京藏高速'
    #dir1 和 dir2 是一个方向
    dir1 = '0'
    dir2 = '1'
    dir3 = '2'
    dir4 = '3'

    #读文件
    linkStatusPath = 'E:/G-1149/trafficCongestion/网格化/linkStatus_' + date + '_完整.csv'
    linkAttrPath = 'E:/G-1149/trafficCongestion/网格化/linkAttr.csv'
    longRoadMidPath = 'E:/G-1149/trafficCongestion/北京地图数据/beijing/dealedMap/长路段验证/' + roadName + '.MID'

    #写文件
    roadPath = 'E:/G-1149/trafficCongestion/长路段判定/' + roadName + '.csv'
    #getParallelLink(longRoadMidPath, linkAttrPath, roadPath, dir1, dir2, dir3, dir4)

    resPath = 'E:/G-1149/trafficCongestion/长路段判定/' + roadName + '_' + date + 'status.csv'
    #longRoad(linkStatusPath, roadPath, linkAttrPath, resPath)

    #京藏高速13日数据特征时序构造
    jingzang = 'E:/G-1149/trafficCongestion/长路段判定/jingzang.csv'
    getJingzangSta(linkStatusPath, roadPath, jingzang)


