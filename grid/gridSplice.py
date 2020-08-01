import csv

def getName(bjTopologyPath, gridLinkPeerPath, gridPath, resPath):
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
    with open(gridPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            linkList = gridLinkPeerDict[r[0]]
            for i in range(len(linkList)):
                csv_writer.writerow([linkList[i], nameDict[linkList[i]]])

    f.close()

def splice(gridPath, gridSplicePath):
    f = open(gridSplicePath, 'w', encoding='utf-8', newline='')  # newline解决空行问题
    csv_writer = csv.writer(f)
    with open(gridPath, 'r') as file:
        reader = csv.reader(file)
        spliceList = [[int((next(reader))[0])]]
        print(spliceList)
        for r in reader:  # r是一个list
            gridNum = int(r[0])
            flag = False
            for i in range(len(spliceList)):
                for j in range(len(spliceList[i])):
                    if(abs(gridNum-spliceList[i][j]) == 1 or abs(gridNum-spliceList[i][j]) == 1782 or
                    abs(gridNum-spliceList[i][j]) == 1783 or abs(gridNum-spliceList[i][j]) == 1784):
                        spliceList[i].append(gridNum)
                        flag = True
                        break



    f.close()

if __name__ == '__main__':
    gridLinkPeerPath = 'E:/G-1149/trafficCongestion/网格化/gridLinkPeer_14.csv'
    gridPath = 'E:/G-1149/trafficCongestion/网格化/tidal/rnn_14.csv'
    bjTopologyPath = "E:/G-1149/trafficCongestion/bjTopology.csv"

    resPath = 'E:/G-1149/trafficCongestion/网格化/tidal/name_1.csv'
    gridSplicePath = 'E:/G-1149/trafficCongestion/网格化/tidal/gridSplice.csv'

    #getName(bjTopologyPath, gridLinkPeerPath, gridPath, resPath)
    #splice(gridPath, gridSplicePath)

    print(1 == 2)
    print(1 == 1)

