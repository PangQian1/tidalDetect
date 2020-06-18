import networkx as nx
import matplotlib.pyplot as plt
import csv

g=nx.DiGraph()#创建空的有向图
with open('D:\\program\\congestion\\test.csv', 'r') as file:
    reader = csv.reader(file)
    
    for row in reader:
        g.add_node(row[5])
        g.add_node(row[6])
        g.add_edge(row[5],row[6])

g.add_node('531572',attr='edd')
#g.add_node(1)
#g.add_nodes_from([2,3,4])
#g.nodes()
#g.add_edge(2,3)

print(g.nodes['531574'])
print(g['531574'])
#g.add_edges_from([(1,2),(1,3)])
#g.edges()

nx.draw(g)
plt.show()