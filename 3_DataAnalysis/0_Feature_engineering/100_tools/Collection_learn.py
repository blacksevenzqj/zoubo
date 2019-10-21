

# 1、字典：
dict1 = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
dict2 = {'a': 1, 'b': 2, 'c': 5, 'e': 6}

# 所有差异（key 和 value）
differ = set(dict1.items()) ^ set(dict2.items())
print(differ)
# 输出:{('c', 3), ('e', 6), ('c', 5), ('d', 4)}

# 查看字典a和字典b相同的键值对（key 和 value 都相同）
print(dict1.items() & dict2.items())

# 两个字典共有的key
diff = dict1.keys() & dict2.keys()
print(diff)
# 查看字典a 和字典b 的不共有的key
print(dict1.keys() ^  dict2.keys())
# 查看在字典a里面而不在字典b里面的key
print(dict1.keys() - dict2.keys())
# 查看在字典b里面而不在字典a里面的key
print(dict2.keys() - dict1.keys())

# 相同key，不同value（不包含 不同key）
diff_vals = [(k, dict1[k], dict2[k]) for k in diff if dict1[k] != dict2[k]]
print(diff_vals)
# 输出:[('c', 3, 5)]



# 2、Set：
# https://blog.csdn.net/Chihwei_Hsu/article/details/81416818
# 2.1、交集
# 方法一:
a=[2,3,4,5]
b=[2,5,8]
tmp = [val for val in a if val in b]
print(tmp)
# 方法二
print(list(set(a).intersection(set(b))))
# 方法三：
print(list(set(a) & set(b)))

# 2.2、差集
# 方法一：
print(list(set(b).difference(set(a)))) # b中有而a中没有的
# 方法二：
print(list(set(b) - (set(a))))

# 2.3、补集
print(list(set(b) ^ set(a))) # 一样
print(list(set(a) ^ set(b)))