import numpy as np

x = np.array([1.0,2.0,3.0])
y = np.array([4.0,5.0,6.0])
z = x+y
j = x-y
k = x * y
m = x / y
X = np.array([[51, 55], [14, 19], [0, 4]])
for row in X :
    print(row)
    for item in row :
        print(item)
print(z)
print(j)
print(k)
print(m)