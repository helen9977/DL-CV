"""
根据手写数字的明暗程度来判断类型
举个例子,1的亮度比8的亮度要高.
图片的平均亮度使用每个像素点求和求平均的方式
这不是一个好的方式,但是值得尝试
准确率 2225 /10000
"""
from collections import defaultdict

import mnist_loader


def avg_darkness(training_data):
    digit_count = defaultdict(int)
    darkness = defaultdict(float)
    for image, digit in zip(training_data[0], training_data[1]):
        digit_count[digit] += 1
        darkness[digit] += sum(image)

    avgs = defaultdict(float)
    for digit, n in digit_count.items():
        avgs[digit] = darkness[digit]/n

    return avgs


def guess_digit(image, avgs):
    darkness = sum(image)
    distance = {k: abs(v-darkness) for k, v in avgs.items()}

    return min(distance, key=distance.get)


if __name__ == "__main__":
    training_data, validation_data, test_data = mnist_loader.load_data()
    avgs = avg_darkness(training_data)
    num_correct = sum(int(guess_digit(test, avgs) == y)
                      for test, y in zip(test_data[0], test_data[1]))
    print("Baseline classifier using average darkness of image")
    print("%s of %s values correct." % (num_correct, len(test_data[1])))