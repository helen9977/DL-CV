import Network3
from Network3 import Network, ConvPoolLayer, FullyConnectionLayer, SoftmaxLayer, ReLU


#import mnist_loader


'''
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print("training data : ")
print(type(training_data))
print(len(training_data))
print(training_data[0][0].shape)
print(training_data[0][1].shape)

print("validation data : ")
print(len(validation_data))

print("test data : ")
print(len(test_data))

#  定义中间层的神经元
net = Network.Network([784, 30, 10])

# def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None)
# 开始训练网络，同时做出预测

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


net = Network2.Network([784, 30, 10], cost=Network2.CrossEntropyCost)

def SGD(self, training_data, epochs, mini_batch_size, eta, lamda: 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False
            ):


net.SGD(training_data, 30, 10, 0.5, 5.0, evaluation_data=validation_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)

'''
training_data, validation_data, test_data = Network3.load_data_shared()
mini_batch_size = 10

# def __init__(self, filter_shape, image_shape, poolsize=(2, 2),activation_fn=sigmoid)

net = Network([
        ConvPoolLayer(filter_shape=(20, 1, 5, 5),
                      image_shape=(mini_batch_size, 1, 28, 28),
                      poolsize=(2, 2),
                      activation_fn=ReLU,
                      ),
        ConvPoolLayer(filter_shape=(40, 20, 5, 5),
                      image_shape=(mini_batch_size, 20, 12, 12),
                      poolsize=(2, 2),
                      activation_fn=ReLU,
                      ),
        FullyConnectionLayer(
                n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5
                ),
        FullyConnectionLayer(
                n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5
                ),
        SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
    mini_batch_size
)


# SGD(self, training_data, epochs, mini_batch_size, eta,validation_data, test_data, lmbda=0.0):
net.SGD(training_data, 40, mini_batch_size, 0.03, validation_data, test_data)
