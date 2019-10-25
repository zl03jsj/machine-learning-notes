from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd
import matplotlib.pyplot as plt


def create_samples(count_samples, params):
    count_inputs = len(params[0])
    x = nd.random_normal(shape=(count_samples, count_inputs))
    y = nd.zeros(shape=count_samples) + params[1]
    for i in range(0, x.shape[1]):
        y += params[0][i] * x[:, i]
    y += (0.001 * nd.random_normal(shape=y.shape))
    return x, y


def net(x, params):
    return nd.dot(x, params[0]) + params[1]


def plot(losses, x, params, real_params, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, list(losses), '-r')

    fg2.set_title('Estimated vs real function')
    fg2.plot(x[:sample_size, 1].asnumpy(),
             net(x[:sample_size, :], params).asnumpy(), 'or', label='Estimated')
    fg2.plot(x[:sample_size, 1].asnumpy(),
             net(x[:sample_size, :], real_params).asnumpy(), '*g', label='Real')
    fg2.legend()
    plt.show()


def main():
    print(__name__)
    if __name__ != '__main__': return

    samples_count = 1000
    params = [nd.array([2, -3.4]), 4.2]

    batch_size = 10
    x, y = create_samples(samples_count, params)
    data_set = gluon.data.ArrayDataset(x, y)
    data_iter = gluon.data.DataLoader(data_set, batch_size)

    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))
    net.initialize()

    square_loss = gluon.loss.L2Loss()
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': 0.1})

    help(trainer.step)

    epochs = 5
    batch_size = 10
    losses = []
    niter = 0
    moving_loss = 0
    smoothing_constant = 0.01

    for e in range(epochs):
        total_loss = 0
        for data, label in data_iter:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()

            print('grads = ', net[0].weight.data(), net[0].bias.data())

            trainer.step(batch_size)

            niter += 1

            curr_loss = nd.mean(loss).asscalar()
            total_loss += curr_loss
            moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss
            est_loss = moving_loss / (1 - (1 - smoothing_constant) ** niter)

            # if (niter + 1) % 5 == 0:
            #     losses.append(est_loss)
            #     print("Epoch %s, batch %s. Moving avg of loss: %s. Average loss: %f" % (
            #         e, niter, est_loss, total_loss / samples_count))
            #     plot(losses, x, [net[0].weight.data()[0], net[0].bias.data()], params)


    dense = net[0]
    print(dense.weight.data())
    print(dense.bias.data())


main()
