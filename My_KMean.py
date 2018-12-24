import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from math import sqrt
import warnings
from collections import Counter
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets

# Tạo Gaussian Distribution dataset (x,y)
mean = [0, 0]
cov =  [[1, 0], [0, 1]]
x, y = np.random.multivariate_normal(mean, cov, 1000).T
X = np.array(list(zip(x, y)))                                       # dataset
num_clusters = 3                                                    # với k = 3

C_x = np.random.randint(np.min(X), np.max(X), size=num_clusters)    # tọa đô x của centroid ngẫu nhiên
C_y = np.random.randint(np.min(X), np.max(X), size=num_clusters)    # tọa đô y của centroid ngẫu nhiên
C = np.array(list(zip(C_x, C_y)), dtype=np.float64)                 # điểm centroid

# Xem giản đồ phân bố ban đầu của tập dữ liệu đầu vào
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c='blue', s=6)
plt.scatter(C_x, C_y, marker='*', s=150, c='g')
plt.show()

##################################################################

# Phân cụm đầu tiên cho dataset
level1 = KMeans(num_clusters, random_state=0)                   # gọi giải thuật kMean
rs_lv1 = level1.fit(X)                                          # kết quả trả ra của kMean
lbl = rs_lv1.labels_                                            # lấy nhãn từng điểm
ctr = rs_lv1.cluster_centers_                                   # centroid của cụm

# Tạo bảng 2 chiều chứa kết quả (data + label)
result1 = pd.DataFrame(X, columns= ['x','y'])
result1['Labels'] = level1.labels_                              # Gán nhãn cụm cho bảng kết quả

print("Lần phân cụm đầu tiên:")
for n in range(num_clusters):
    Cluster_group = result1.loc[lbl == n]                     # xem data tương ứng của từng cụm trong bảng
    print(Cluster_group)

# Xem giản đồ thường sau khi phân cụm đầu tiên
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:,0], X[:,1], c=result1['Labels'], s=5)
# plt.scatter(ctr[:, 0], ctr[:, 1], marker='*', s=200, c='g')

# Xem giản đồ với colorbar sau khi phân cụm đầu tiên
ax = result1.plot.scatter(x='x', y='y', c=result1['Labels'], colormap='plasma', s=5, figsize=(8, 6))
ax.scatter(ctr[:, 0], ctr[:, 1], marker='*', s=200, c='darkblue')    # xem vị trí centroid
plt.show()

##################################################################

# Phân cụm lần thứ 2 cho 3 cụm của lần chia đầu tiên , k = 3
lst = []
for i in range(num_clusters):
    cluster_data = result1[result1['Labels'] == i].drop(['Labels'], axis=1)
    level2 = KMeans(num_clusters)
    level2.fit(cluster_data)
    cluster_data['Labels2'] = level2.labels_+ (i+1)*10
    lst.append(cluster_data)

print("Lần phân cụm thứ 2:")
# plt.show()

##################################################################

# Phân cụm lần 3 cho 9 cụm đã chia được tại lần chia thứ  2
rs = []
for i in range(num_clusters):               # lấy 3 cụm parent của lần chia thứ 2, mỗi cụm parent có 3 cụm con
    data = lst[i]
    label_set = list(data.Labels2.unique())
    for label in label_set:
        cluster_data = data[data['Labels2'] == label].drop(['Labels2'], axis=1)
        level3 = KMeans(n_clusters=3)
        level3.fit(cluster_data)
        cluster_data['Labels3'] = level3.labels_ + (label)*10
        rs.append(cluster_data)              # rs là 9 dataframe, mỗi datafram là 9 cluster của lần chia thứ 2
                                             # trong mỗi dataframe chứa 3 cụm mới

print("Lần phân cụm thứ 3:")
# print(rs )

rs_final = pd.concat(rs)

##################################################################

