import csv

#f = open('data/test_4_two.csv', 'w', encoding='utf-8', newline='')  # newline解决空行问题
f = open('E:/G-1149/trafficCongestion/长路段判定/jingzang_two.csv', 'w', encoding='utf-8', newline='')  # newline解决空行问题

csv_writer = csv.writer(f)
noData = {}
#with open('data/test_4.csv', 'r') as file:
with open('E:/G-1149/trafficCongestion/长路段判定/jingzang.csv', 'r') as file:
    reader = csv.reader(file)
    for item in reader:
        csv_writer.writerow(item[0:16])
        csv_writer.writerow(item[16:32])
f.close()

