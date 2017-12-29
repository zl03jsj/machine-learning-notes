import mxnet.ndarray as nd
import mxnet.autograd as ag

x = nd.array([[1, 2], [3, 4]])
x.attach_grad()

with ag.record():
    y = 2 * x
    z = x * y

z.backward()

print(x.grad == 4 * x)
print('x.grad = ', x.grad)

# x = nd.array([0, 1, 2, 3, 4])
# x.attach_grad()
# z = nd.zeros(x.shape)
#
# with ag.record():
#     y = x * x * 2
#
# y.backward()
#
# print('x.grad', x.grad)
