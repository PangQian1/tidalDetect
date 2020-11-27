"""
网格编号至网格坐标的转换，同时提取了道路名称，用以可视化（因为可视化需要坐标作为支持）
"""

import csv
import config

def numToCoor(gridTidalPath, bjTopologyPath,  gridPoiPath, outPath):
    #link 和 name 对应字典
    nameDict = {}
    with open(bjTopologyPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            nameDict[r[0]] = r[6]

    #网格和经纬度的对应，利用网格里link的坐标来代表(具有一定随机性，潮汐路段的起止点本身就不是那么明确的）
    coorDict = {}
    with open(gridPoiPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            if(r[0] not in coorDict.keys()):
                coorDict[r[0]] = r[1]


    f = open(outPath, 'w', encoding='utf-8', newline='')  # newline解决空行问题
    csv_writer = csv.writer(f)
    with open(gridTidalPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:
            num = int(r[0])
            nameList = []
            x = 0
            y = 0
            if((num % config.max_X) == 0):
                x = config.max_X
                y = int(num / config.max_X)
            else:
                x = num % config.max_X
                y = int(num / config.max_X) + 1
            for i in range(1, len(r)):
                if(r[i] != ''): #由于对预测后的网格文件进行了排序处理，Excel自动格式对齐，因此出现了空白填充
                    name = nameDict[r[i]]
                    if(name not in nameList):
                        nameList.append(name)
            csv_writer.writerow([x, y,coorDict[r[0]] , '-'.join(nameList)])

    f.close()


if __name__ == '__main__':

    #读文件
    gridTidalPath = 'E:/G-1149/trafficCongestion/网格化/tidal/gridTidal_rnn_new13.csv'
    bjTopologyPath = "E:/G-1149/trafficCongestion/bjTopology.csv"
    gridPoiPath = 'E:/G-1149/trafficCongestion/网格化/gridPoi.csv'

    #写文件
    coorPath = 'E:/G-1149/trafficCongestion/网格化/tidal/coordinate_new13.csv'

    numToCoor(gridTidalPath, bjTopologyPath,  gridPoiPath, coorPath)