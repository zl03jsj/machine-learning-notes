import mxnet.ndarray as nd
import mxnet.autograd as ag

x = nd.array([0, 1, 2, 3, 4])
x.attach_grad()

z = nd.zeros(x.shape)

with ag.record():
    y = x * x * 2

y.backward()

print('x.grad', x.grad)
