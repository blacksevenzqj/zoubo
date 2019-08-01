# coding:utf-8

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print(np.sqrt(6 * np.sum(1 / (np.arange(1, 10000) ** 2))))

    x = np.arange(1, 20)
    print(np.sum(1 / x.cumprod()) + 1)