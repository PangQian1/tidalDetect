"""
网格编号和网格坐标的转换
"""

import csv
import config

def numToCoor(gridTidalPath, bjTopologyPath, outPath):
    #link 和 name 对应字典
    nameDict = {}
    with open(bjTopologyPath, 'r') as file:
        reader = csv.reader(file)
        for r in reader:  # r是一个list
            nameDict[r[0]] = r[6]

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
                if(r[i] != ''):
                    name = nameDict[r[i]]
                    if(name not in nameList):
                        nameList.append(name)
            csv_writer.writerow([x, y, '-'.join(nameList)])

    f.close()


if __name__ == '__main__':

    #读文件
    gridTidalPath = 'E:/G-1149/trafficCongestion/网格化/tidal/gridTidal_rnn_14.csv'
    bjTopologyPath = "E:/G-1149/trafficCongestion/bjTopology.csv"

    #写文件
    coorPath = 'E:/G-1149/trafficCongestion/网格化/tidal/coordinate_1.csv'

    numToCoor(gridTidalPath, bjTopologyPath, coorPath)