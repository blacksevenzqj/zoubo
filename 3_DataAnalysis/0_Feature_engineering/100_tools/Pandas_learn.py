import numpy as np
import pandas as pd

# 1、列对齐
data1 = pd.DataFrame(np.ones((6,6),dtype=float),columns = ['a','b','c','d','e','f'],index = pd.date_range('6/12/2012',periods =6))
data2 = pd.DataFrame(np.ones((6,3),dtype = float) *2 ,columns=['a','b','c'],index=pd.date_range('6/13/2012',periods=6))
data1,data2 = data1.align(data2, join = 'inner', axis = 1)


