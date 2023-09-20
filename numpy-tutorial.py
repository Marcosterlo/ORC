import numpy as np

x = np.zeros(3)
y = np.array([3., 6., 12.])
A = np.zeros([3, 3])
I = np.identity(3)
B = np.empty((3, 4))*np.nan

'''
 Full of trash variables, in case we wanto to be more 
 efficient in terms of memory
 
 nan too is used (not a number), is the result of 
 division by 0.
 Usually creating a new vector or matrix he creates it
 as empty and then multiplies it by np.nan. In this 
 way we can understand at the end of the script if some
 parts of the matrices remain nan it means they have
 not been used
'''

def increment(x):
    x = np.copy(x) # This was used to work on a copy of the object passed, 
                   # a quanto pare non serve più perché python lavora sui 
                   # reference di default, se troviamo np.copy nei suoi script
                   # ora sappiamo perché
    x = x+1 # if it is a vector numpy adds 1 to each element of the vector
    return x

z = increment(y)

'''
Some useful functions:
    np.max()
    np.min()
    np.mean()
    np.abs()
    np.linalg.norm()
'''

# Gives random values between 0 and 1 from a uniform distribution
C = np.random.rand(3, 3)

# Inverse of a matrix
D = np.linalg.inv(C)

# Matricial multiplication with @
print(C @ D) # except for numerical uncertainties it's an identity matrix

# Function for pseudo-inverse (from the left )
E = np.random.rand(3, 2)
print(np.linalg.pinv(E) @ E)

# this is a change
