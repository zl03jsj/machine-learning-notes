a = 'hello world!!!'
print(a)

words = ['cat', 'window', 'hello']
for w in words:
    print w, len(w)

print('-----------------------')
for w in words[0:len(words)]:
    if len(w) > 4:
        words.insert(0, w)
    print(w)

print(words)

print('-----------------------')
def fib(n):
    print "*****a fibonacci series up ", n, "*****"
    a, b = 0, 1
    # for i in range(0, n, 1):
    while a < n:
        print a,
        a, b = b, a+b
    print "\n"

fib(2000)

def printSpliteLine(str) :
    specialstr = "-----"
    print specialstr, str, specialstr

##### tuple #####
printSpliteLine("tuple")

tple = (10, 20, "hello")
print tple

a, b, c = tple
print a , b , c
print '\n'

##### set #####
printSpliteLine("set")
a = set('abcd')
b = set('cdef')
print 'a = ', a
print 'b = ', b
print 'a | b = ', a|b
print 'a - b = ', a-b
print 'a & b = ', a&b
print 'a ^ b = ', a^b
print '\n'


import numpy as np
from numpy import *
##### array #####
# a = np.array([[1., 2, 3], [4, 5, 6]])
a = np.arange(0, 15, 2).reshape(2, 4)
print 'a = ', a
print 'a\'s dtype = ', a.dtype, ', ndim = ', a.ndim, ', shape = ', a.shape, ', size = ', a.size
print 'a\'s dtype = ', a.dtype, ', each item size = ', a.itemsize, ', data cache = ', a.data

print '................'
a = np.array([[1, 2], [3, 4]], dtype=complex)
print 'a = ', a
print 'a\'s dtype = ', a.dtype, ', ndim = ', a.ndim, ', shape = ', a.shape, ', size = ', a.size
print 'a\'s dtype = ', a.dtype, ', each item size = ', a.itemsize, ', data cache = ', a.data



