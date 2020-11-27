# -*- coding: utf-8 -*-
"""
@author: tom
"""

import numpy
import math
import matplotlib.pyplot as plt


def sigmoid(x):
    a = []
    for item in x:
        a.append(1.0 / (1.0 + math.exp(-item)))
    return a


x = numpy.arange(-100, 100, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.show()

x = numpy.arange(-100, 100, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.show()