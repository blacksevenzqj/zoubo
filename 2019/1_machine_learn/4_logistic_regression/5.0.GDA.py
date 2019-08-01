#!/usr/bin/python
# -*- coding:utf-8 -*-

import math


if __name__ == "__main__":
    learnig_rate = 0.1
    x = 1
    for i in range(200):
        x -= learnig_rate * x**x * (math.log(x)+1)
        print(i, x, 1/x)
