import time

start_time = time.time()

y = 5

if (y > 4):
    print("Maggiore di 4")
elif (y <= 4):
    print("Minore o uguale a 4")

for i in range(0, 10, 2):
    print(i)
    if (i > 5):
        break

print(i) # Loop variable still existent outside of for loop

i = 0
while (i<10):
    print(i)
    i += 1

# LISTS
x = [2, 8, 19, 43, 36]
print(x)

for i in x:
    print(i)

print("The first element of the list is:", x[0])
print("The last element of the list is:", x[-1])
print("The penultimo element of the list is:", x[-2])
print("The length of the list is:", len(x))
print("A slice of the list from 0 to 2 is:", x[0:3])
print("A slice of the list from 2 to 4 is:", x[2:4])
print("A slice of the list from 3 to 4 is:", x[3:4])
print("All the elements starting from index 2:", x[2:])

y = [8, 3, 6]
z = x + y
z.append(78)
z = z + [78]
print(z)

# TUPLE can't be modified

x = (4, 8, 12)

# DICTIONARY
d = {}
d["red"] = [1, 0, 0]
d["green"] = [0, 1, 0]
d["blue"] = [0, 0, 1]
d["white"] = [0, 0, 0]
d["black"] = [1, 1, 1]

print(d)

# directly create a dictionary with values
e = {'red': [1, 0, 0], 'green': [0, 1, 0]}

print(e)

print("The RGB code of white is", d["white"])

d["none"] = "The string None can be tricy"

print(d)

# function definition
def sum(a, b=0): # doing so the second argument is optional and is 0 by default # doing so the second argument is optional and is 0 by default
    s = a + b
    return (s, a, b)

# to return more than one value use tuples

res = sum(4, 9)

print("4 + 9 =", res)
print("4 + 0 =", sum(4))


'''
OBJECT ORIENTED PROGRAMMING
'''

# Classes

class Point2D: 
    
    def __init__(self, x, y): # self is equivalent to this in C++
        self.x = x
        self.y = y
        
    def increment(self):
        self.x += 1
        self.y += 1
    
    def print(self):
        print("x =", self.x, " y =", self.y)

    def sum(self, x1, y1):
        self.x += x1
        self.y += y1 

p = Point2D(3, 6)

p.print()
p.increment()
p.print()
p.sum(-4, 3)
p.print()


# assert(x==0) # Come C

# Accessing time
print("The current time is:", time.time())

final_time = time.time()

print("Execution time:", final_time - start_time)