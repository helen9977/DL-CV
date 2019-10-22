"""
使用svm 完成 手写数字识别
准确率: 9435/10000
"""

import mnist_loader

from sklearn import svm


if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data()

    clf = svm.SVC() # svc 用于分类     svr用于回归
    clf.fit(training_data[0], training_data[1])

    predictions = [int(a) for a in clf.predict(test_data[0])]

    num_correct = sum(int(a == y)for a, y in zip(predictions, test_data[1]))

    print('Baseline classifier using an SVM.')
    print('%s of %s values correct.' % (num_correct, len(test_data[1])))