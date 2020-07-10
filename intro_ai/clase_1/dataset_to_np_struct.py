import numpy as np


# class DsToNpStruct(object)
#
#     instance = None
#     def __new__(cls, dataset):
#         if DsToNpStruct.instance is None:
#             print("__new__ object created")
#             DsToNpStruct.instance = super(DsToNpStruct, cls).__new__(cls)
#             return DsToNpStruct.instance
#         else:
#             return DsToNpStruct.instance
#
#     def __init__(self, dataset):
#         print("__init__")

# # with open('G:\My Drive\AI\CURSO\intro_ia\ratings.csv') as csv_file:
# csv_file = 'ratings.csv'
# f = open(csv_file, 'r')
# csv_reader = csv.reader(f, delimiter=',')
# # data = list(csv.reader(f))
# ncol = len(next(csv_reader))
# # row_count = sum(1 for row in csv_reader)
# f.seek(0)
# print(ncol)
# # print(row_count)

structure = np.dtype({'names': ('userId', 'movieId', 'rating', 'timestamp'),
                      'formats': ('u1', 'i4', 'f8', 'i4')})

my_data = np.genfromtxt('rank.csv', delimiter=',')
row_count = np.size(my_data)
a = np.empty(int(row_count/4), dtype=structure)
for i in list(range(int(row_count/4))):
    a[i]['userId'] = my_data[i, 0]
    a[i]['movieId'] = my_data[i, 1]
    a[i]['rating'] = my_data[i, 2]
    a[i]['timestamp'] = my_data[i, 3]


print(a)


