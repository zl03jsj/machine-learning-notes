from mxnet import autograd
from mxnet import ndarray as nd
import matplotlib.pyplot as plt
import random
from MXNET import utils


# def subplot():
#     import mumpy as np
#     def f(t):
#         return np.exp(-t) * np.cos(2 * np.pi * t)
#
#     t1 = np.arange(0, 5, 0.1)
#     t2 = np.arange(0, 5, 0.02)
#
#     # plt.figure(12)
#     plt.subplot(221)
#     plt.plot(t1, f(t1), 'bo', t2, f(t2), 'r--')
#
#     plt.subplot(222)
#     plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
#
#     plt.subplot(212)
#     plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
#
#     plt.show()

# subplot()


def create_samples(num_inputs, num_examples, params):
    w = params[0]
    b = params[1]
    # x = nd.random_normal(shape=(num_examples, num_inputs), loc=2, scale=0.1)
    x = nd.random_normal(shape=(num_examples, num_inputs), loc=2, scale=1)
    y = w[0] * x[:, 0] + w[1] * x[:, 1] + b
    y += (0.01 * nd.random_normal(shape=y.shape))
    # plt.subplot(131)
    # plt.scatter(x[:, 0].asnumpy(), x[:, 1].asnumpy())
    # plt.subplot(132)
    # plt.scatter(x[:, 0].asnumpy(), y.asnumpy())
    # plt.subplot(133, )
    # plt.scatterer(x[:, 1].asnumpy(), y.asnumpy())
    # plt.show()
    return x, y


def data_iter(x, y, batch_size):
    import random
    num_examples = len(x)
    idx = list(range(num_examples))
    random.shuffle(idx)

    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_examples)])
        yield nd.take(x, j), nd.take(y, j)


def net(x, params):
    return nd.dot(x, params[0]) + params[1]


def plot(losses, x, params, real_params, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')

    fg2.set_title('Estimated vs real function')
    fg2.plot(x[:sample_size, 1].asnumpy(),
             net(x[:sample_size, :], params).asnumpy(), 'or', label='Estimated')
    fg2.plot(x[:sample_size, 1].asnumpy(),
             net(x[:sample_size, :], real_params).asnumpy(), '*g', label='Real')
    fg2.legend()
    plt.show()


def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2


def main():
    print(__name__)
    if __name__ != "__main__": return

    num_inputs = 2
    num_examples = 1000
    true_w = nd.array([2, -3.4])
    true_b = 4.2
    real_params = [true_w, true_b]

    x, y = create_samples(num_inputs, num_examples, real_params)

    w = nd.random_normal(shape=(num_inputs, 1))
    b = nd.zeros((1,))
    params = [w, b]

    for param in params:
        param.attach_grad()

    epochs = 1000
    learning_rate = .001
    niter = 0
    losses = []
    moving_loss = 0
    smoothing_constant = .01

    for e in range(epochs):
        total_loss = 0

        print('------------------\n', params )
        for data, label in data_iter(x, y, 10):
            # print(data, label)
            with autograd.record():
                output = net(data, params)
                loss = square_loss(output, label)

            loss.backward()
            print('grads = ', params[0].grad, params[1].grad)

            utils.sgd(params, learning_rate)
            total_loss += nd.sum(loss).asscalar()

            # 记录每读取一个数据点后，损失的移动平均值的变化；
            niter += 1
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss

            # correct the bias from the moving averages
            est_loss = moving_loss / (1 - (1 - smoothing_constant) ** niter)

            if (niter + 1) % 200 == 0:
                losses.append(est_loss)
                print("Epoch %s, batch %s. Moving avg of loss: %s. Average loss: %f" % (
                    e, niter, est_loss, total_loss / num_examples))
                plot(losses, x, params,  real_params)


main()

# true_w, w
# true_b, b
exit()
