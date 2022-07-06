import numpy as np

a = np.array([1,2,5], dtype='int32') # specify data type; int16, int8, int32
print(a)

c = np.array((1,2,3))
print(c)

b = np.array([[1,2,3],[4,5,6]])
print(b)

# Get Dimension
bdim = b.ndim
adim = a.ndim
print(f'Dimension of a = {adim}, b = {bdim}')

# Get Shape
bshape = b.shape
ashape = a.shape
print(f'Shape of  a = {ashape}, b = {bshape}')

# Get Type
atype = a.dtype
btype = b.dtype
print(f'Type of  a = {atype}, b = {btype}')

# Get size
asize = a.itemsize
print(f'Size of a = {asize}')

# Get total size
atotal = a.size * a.itemsize
print(f'Total Size of a = {atotal}')
# OR
atot = a.nbytes
print(f'Total Size of a = {atot}')

 # Accessing/Changing specific elements, rows, columns, etc.

a = np.array([[1,2,3,4,5,6,7],[12,34,34,67,12,78,25]])
print(a)

# Get a specific element [r, c]
print(a[0,-7])
print(a[0,0])
# Both of these print statements print same thing.
# Indexing starts at 0
#  0  1  2  3  4  5  6 
# -7 -6 -5 -4 -3 -2 -1 

# Get a specific row
print(a[0,:])

# Get a specific column
print(a[:,1])

# [startindex:endindex:stepsize]
print('Fancy !')
print(a[0, 1:6:2]) # endindex is excluded
print(a[1,0:-1:2])

# Change element
a[1,0] = 77
print(a,'\n')

# Change column
a[:,2] = [17,7]
print(a,'\n')

a[:,2] = [89,]
print(a,'\n')

# Change row
a[0,:] = [72,23,56,11,89,90,13]
print(a,'\n')

# 3D Example
b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(b)
print(f'Size of b = {b.size}')
print(f'Shape of b = {b.shape}')
print(f'Nbyte of b = {b.nbytes}')
print(f'Item size of b = {b.itemsize}')

# Get specific element ( work outside in)
print(b[0,1,1])

# replace
b[:,1,:] = [[9,9],[8,8]]
print(f'After replacement \n {b}')

# Initializing different types of array 
# All zeros matrix
z = np.zeros((2,3,3))
print(z)

# All ones matrix
o = np.ones((2,2,2))
print(f'Ones \n{o}')

# Any other number
num = np.full((2,2,2), 99, dtype='int32')
print(f'Any other number \n {num}')

# Any other number ( full_like)
n = np.full_like(a, 7)
print(f'full_like \n{n}')

# Random decimal numbers
ran = np.random.rand(4,2)
ransample = np.random.random_sample(a.shape)
print(f'Random numbers between 0 and 1 \n{ran}')
print(f'Random Sample like a \n{ransample}')

# Random integers
ranint = np.random.randint(4,7, size=(3,3)) # between 4 and 7
print(f'Random int between 4 and 7 \n{ranint}')

# Identity matrix
idmat = np.identity(5)
print(f'Identity matrix \n{idmat}')

# repeat
arr = np.array([[1,2,3]])
r1 = np.repeat(arr, 3, axis=0) # axis = 1
print(r1)

# Task for Today
tsk  = np.zeros((5,5))
tsk[0,:]=1
tsk[:,0]=1
tsk[4,:]=1
tsk[:,4]=1
tsk[2,2]=9
print(f'Task \n{tsk}')

# Alternative solution
output = np.ones((5,5))
z = np.zeros((3,3))
z[1,1] = 9
output[1:-1, 1:-1] = z
print(output)

# Be careful while copying arrays 
a = np.array([1,2,3])
b = a # variable 'b' points to the same thing  as variable 'a', we didn't make a copy of 'a'
b[0] = 100
print(a)

# Solution
a = np.array([1,2,3])
b = a.copy()
b[0] = 100
print(a)

# Mathematics
a = np.array([1,2,3,4])
print(a)

a += 2
print(a)

a -= 2
print(a)

a *= 2
print(a)

a = a/2
print(a)

b = np.array([5,10,15,20])
print(f'a + b = \t{a + b}')
print(f'a * b = \t{a * b}')
print(f'a / b = \t {b / a}')

# Take sin
print(f'sin :  {np.sin(b)}')

# Linear Algebra
print('Linear Algebra')
a = np.ones((2,3))
print(a)
b = np.full((3,2), 2)
print(b)

print(f'Product a * b = \t {np.matmul(a,b)}')

# Determinant
c = np.identity(3)
det = np.linalg.det(c)
print(f'Determinant = {det}')

# Statistics
stats = np.array([[1,2,3],[4,5,6]])
print(f'Statistics = {stats}')
print(f'Minimum = {np.min(stats, axis = 1)}')
print(f'Maximum = {np.max(stats, axis = 1)}')


# Reorganizing arrays
print('Reorganizing Arrays')
before = np.array([[1,2,3,4],[5,6,7,8]])
print(f'Before : {before}')

after = before.reshape((4,2))
print(f'After : {after}')

# Vertically stacking vectors
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])

stack = np.vstack([v1,v2])
print(f'Vertical stack :\n {stack}')

# Horizontal stack
h1 = np.ones((2,2))
h2 = np.zeros((2,2))
stack = np.hstack([h1,h2,h1,h2])
print(f'Horizontal stack :\n {stack}')

# Miscellaneous
# Load data from file 
filedata  = np.genfromtxt('datatxt.txt', delimiter=',')
filedata = filedata.astype('int32')
print(f'Data from file : \n{filedata}')

# Boolean masking and advanced Indexing
chk = filedata > 5
print(chk)

fd = filedata[filedata > 7]
print(fd)

# You can index with list in Numpy
a = np.array([1,2,3,4,5,6,7,8,9])
print(a[[2,3,7]])

b = np.any(filedata > 10, axis = 0)
print(b)

b = np.all(filedata > 10, axis=0)
print(b)

c = ((filedata > 5) & (filedata < 13))
print(c)
