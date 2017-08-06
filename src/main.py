import mnist_loader
import network


def naive_network():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.stochastic_gradient_descent(training_data=training_data, epochs=30, mini_batch_size=10,
                                    eta=3.0, test_data=test_data)


if __name__ == '__main__':
    naive_network()
