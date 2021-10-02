#Numpy cheatsheet
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(type(a))

two = np.array([[1,2,3,4,5], [6,7,8,9,6]], dtype='int16')# or dtype='int32' - memory efficient way
print(two)

# Getting the shape
print(two.ndim)# Geting the dimension
print(two.shape)#gettind the shap ein (Rows, Columns)

print(two.dtype)# to prin the data type(no. bytes allocated to an item for faster manipulation)
print("Number of bytes-8 bits as one byte though it is stored as int16 it shows 2: ",two.itemsize)

print(a.size) #gives the number of elements where as .itemsize gives number of bytes

print(a.nbytes)#gives the total number of bytes in a array where as itemsize gives for the siingle element

# floats are bigger than integers 

print(two[1, 2])#Accessing the elements in a 2d array
print(two[0,:])#Accessing the whole row
print(two[:, 3])#accessing the whole column
print(two[0, 1:4:2])#Accessing by steps

#3d arrays
t = np.array([[[1, 2, 3 ], [4, 5, 6]],[[12, 34 ,55 ], [98, 89, 78]]])
print(type(t))
print(t)

#Accessing 3d array
print(t[1, 0, :])

#Initializing different types of arrays

#All 0's matrix
zeros = np.zeros((2,3), dtype='int16')
print(zeros)

#All 1's matrix
ones = np.ones((2,3), dtype='int32')
print(ones)

#All randoms's matrix
random = np.full((2,3), 99, dtype='int16')
print(random)

#All randoms's by copying the shape of another array matrix
random = np.full_like(t, 9) # or np.full(t.shape, 7)
print(random)

# Random array generator

#gives random numbers [floats]
x = np.random.rand(6, 5)
print(x)

#gives random integers[no-floats]
x = np.random.randint(9,size=10)# or sixe may be(2,3)
print(x)

# To generate a identity matrix
y = np.identity(7)
print(y)

#Reapeating an array 
arr = np.array([[1,2,3]])
threeD = np.repeat(arr, 3, axis=0) # if axis is 1 it will replicate into a 1d array
print(threeD)

# To copy the array
b = np.full(a.shape, 7)
c = b.copy()#safest way to copy rather than just b = c which will become hard to manipulate c without changin b

print(b,"is same as", c)

## LINEAR ALGEBRA

#matrix multiplication

aa = np.ones((2,3))
bb = np.full((3, 2), 7)

# print(aa*bb) It will throw an error because their shapes are not identical
xx = np.matmul(aa, bb)
print(xx)

# to find the determinent of a matrix
print(np.linalg.det(xx))


## Statistics

# min
cc = np.array([1,2,3,4])
print(np.min(cc))

dd = np.random.randint(9, size=(3,4))
print(dd)
print(np.min(dd, axis=0)) # to get the minimum of each column
print(np.min(dd, axis=1)) # to get the minimum of each row

# Stacking

# Vertically stacking vectors
v1 = np.ones((2,2))
v2 = np.zeros((2,2))

print(np.vstack([v1,v2,v1,v2]))

# Horizontal  stack
h1 = np.ones((2,4))
h2 = np.zeros((2,2))

print(np.hstack((h1,h2)))


#loading a file in numpy
filedata = np.genfromtxt('data.txt', delimiter=',') #there is no file exist
filedata = filedata.astype('int32')
print(filedata)