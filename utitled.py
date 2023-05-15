# ls = [5, 1, 4]
# length = len(input)
# num_op = 2
#
# for i in range(num_op):
#     min_num = min(ls)
#     min_index = ls.index(min_num)
#     max_num = max(ls)
#     max_index = ls.index(max_num)
#
#     ls[max_index] = min_num

# a = input()
# b = input()
# a = "5 3 2 1 1"
# b = "5 3 2 2 1"
#
# if len(a) > len(b):
#     for i in range((len(a) - len(b)) // 2):
#         b = "0" + " " + b
#
# a = a.split(" ")
# b = b.split(" ")
#
# result = []
#
# for num in range(len(a)):
#     sum = eval(a[-1]) + eval(b[-1])
#     if sum >= num + 2:  # 进制
#         add = 1  # 进位的值
#         try:
#             a[-2] = str(eval(a[-2]) + add) # a的下一位加1
#             result.append(0 + sum - (num + 2))  # 当前这一位的结果
#             a.pop()
#             b.pop()
#         except:
#             result.append(0 + sum - (num + 2))
#             result.append(1)
#
#     else:
#         result.append(sum)
#         a.pop()
#         b.pop()
#
# print(result[::-1])
# print(" ".join(result[::-1]))


import numpy as np
a = np.array([0, 3, 3, 2, 1, 1])
b = np.array([0, 0, 3, 3, 2, 1])

jinzhi = np.array([i + 2 for i in range(len(a))][::-1])
add = np.array([i for i in range(len(a))][::-1])
out = a + b
booleam = np.array([1 for i in range(len(a))], dtype=np.bool)
while booleam.any():
    booleam = out >= jinzhi  # 要进位的为True
    out = out % jinzhi # 余数
    result = add * booleam
    print(bool)

print(out)
