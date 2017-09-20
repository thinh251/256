# coding=utf-8
import math


class Perception(object):
    PERCEPTRON_ALG = 'perception'
    WINNOW_ALG = 'winnow'

    def __init__(self):
        super(Perception, self).__init__()
        self.features = None
        # self.learningrate = 0.1
        self.theta = 0
        self.weights = None  # The param use in threshold function
        self.target = 0  # The target use in threshold function
        self.expression = ''  # The expression used in NBF
        self.features_num = 0  # Number of features

    def activation(self, x, activationmethod):
        if activationmethod == 'threshold':
            return 1 if x > self.theta else 0
        elif activationmethod == 'relu':
            return max(0, self.theta)
        elif activationmethod == 'tanh':
            return 1/2 +  1/2*(math.tanh((x-self.theta)/2)

    # Activation function with threshold, params and target is parsed from ground_file
    # For example:
    #   TF
    #   +15             //this is target
    #   +10 -5 +30      //this is params
    # f(x) = 1 iff 10x1 - 5x2 + 30x3 >= 15
    @staticmethod
    def threshold_func(params, ft, target):
        sump = 0
        for i in range(0, len(params)):
            sump += params[i] * ft[i]
        return 1 if sump >= target else 0

    # Activation function in NBF case
    @staticmethod
    def nested_boolean_func():
        # +1 OR -2 AND +5 AND +3
        # f(x1,x2,x3,x4,x5):=(((x1 ∨ ¬x2) ∧ x5) ∧ x3)
        # TODO:
        return 1

    def predict(self, features, activation):
        # Predict by threshold function
        if activation == 1:
            # params and target is initialized when parsing the ground_file
            guess = self.threshold_func(self.weights, features, self.target)
        # Predict by nested boolean function
        else:
            # TODO: not done with NBF yet, this is just a sketch
            guess = self.nested_boolean_func()

        return guess

    def train(self, features, target, train_alg):
        # TODO: Assume that threshold function is used, I'm not very clear for tha activation step
        guess = self.predict(features, 1)
        error = target - guess
        if train_alg == Perception.PERCEPTRON_ALG:
            self.perceptron_train(features, target, error)
        elif train_alg == Perception.WINNOW_ALG:
            self.winnow_train(error)
        else:
            raise NotImplementedError('This training algorithm is not implemented')

    def test(self, features, target):
        # TODO: Assume that threshold function is used, I'm not very clear for tha activation step
        guess = self.predict(features, 1)
        error = math.fabs(target - guess)
        txt = ''
        for t in range(0, len(features)):
            txt += str(features[t]) + ", "
        txt += ": " + str(guess) + " : " + str(target) + " : " + str(error)
        print txt
        return error

    def perceptron_train(self, features, target, error):
        # False positive prediction, set w = w - x and set theta = theta + 1
        txt = ""
        if error > 0:
            for i in range(0, len(self.weights)):
                self.weights[i] = self.weights[i] - features[i]
                self.theta += 1
                txt += str(features[i]) + ", "
            txt += ": " + str(target) + ": UPDATE"
        # False negative prediction, set w = w + x and set theta = theta -1
        elif error < 0:
            for j in range(0, len(self.weights)):
                self.weights[j] = self.weights[j] + features[j]
                self.theta -= 1
                txt += str(features[j]) + ", "
            txt += ": " + str(target) + ": UPDATE"
        else:
            for k in range(0, len(self.weights)):
                txt += str(features[k]) + ", "
            txt += ": " + str(target) + ": no update"
        print txt

    def winnow_train(self, error, alpha):
        # On a false positive prediction, for all i, set wi := α ** −xi * wi
        if error > 0:
            for i in range(0, len(self.weights)):
                self.weights[i] = (alpha ** (-self.features[i])) * self.weights[i]
        # On a false negative prediction, for all i, set wi := α ** xi * wi
        if error < 0:
            for i in range(0, len(self.weights)):
                self.weights[i] = (alpha ** self.features[i]) * self.weights[i]



