# coding=utf-8
import math


class Neuron(object):
    PERCEPTRON_ALG = 'perceptron'
    WINNOW_ALG = 'winnow'

    def __init__(self):
        super(Neuron, self).__init__()
        self.features = None
        # self.learningrate = 0.1
        self.theta = 0      # Assume theta = 0 for perceptron update rule
        self.alpha = 1.8    # Assume alpha > 0 for winnow update rule
        self.weights = None  # The param use in threshold function
        self.target = 0  # The target use in threshold function
        self.expression = ''  # The expression used in NBF
        self.features_num = 0  # Number of features

    def activation(self, x, activationmethod):
        if activationmethod == 'threshold':
            return 1 if x >= self.theta else 0
        elif activationmethod == 'relu':
            return max(0, x - self.theta)
        elif activationmethod == 'tanh':
            return 1/2 + 1/2*(math.tanh((x-self.theta)/2))

    # Activation function with threshold, params and target is parsed from ground_file
    # For example:
    #   TF
    #   +15             //this is target
    #   +10 -5 +30      //this is params
    # f(x) = 1 iff 10x1 - 5x2 + 30x3 >= 15
    def threshold_func(self, params, ft, target):
        sump = 0
        params = map(int, params)
        for i in range(0, len(params)):
            sump += params[i] * ft[i]
            # print 'Param: ', params[i], ' feature: ', ft[i], 'sump: ', sump

        #print "Sump: ", sump, 'Target: ', target
        if sump >= target:
            #  print "Return 1 Sump: ", sump, 'Target: ', target
            return 1
        else:
            return 0
        # return 1 if sump >= target else 0

    # Activation function in NBF case
    def nested_boolean_func(self, expression, features):
        # +1 OR -2 AND +5 AND +3
        # f(x1,x2,x3,x4,x5):=(((x1 ∨ ¬x2) ∧ x5) ∧ x3)
        expression = expression.replace('-', 'not ')
        expression = expression.replace('+', '')
        for i in range(len(features)):
            # if str(i) in expression:
                expression = expression.replace(str(i+1), str(features[i]))
        return eval(expression)

    def predict(self, features, param, function_type, target):
        # Predict by threshold function
        if function_type == 'TF':
            # params and target is initialized when parsing the ground_file
            guess = self.threshold_func(param, features, target)
            # Predict by nested boolean function
        elif function_type == 'NBF':
            guess = self.nested_boolean_func(param, features)
        return guess

    def train(self, features, train_alg, param, function_type, target, activation_method):
        guess = self.predict(features, param, function_type, target)

        # Calculate the real output
        if train_alg == Neuron.PERCEPTRON_ALG:
            self.perceptron_train(features, guess, activation_method)
        elif train_alg == Neuron.WINNOW_ALG:
            self.winnow_train(features, guess, activation_method)
        else:
            raise NotImplementedError('This training algorithm is not implemented')

    def test(self, features, param, function_type, target, activation_method):
        guess = self.predict(features, param, function_type, target)
        # print 'Guess: ', guess
        x = 0
        for f in range(0, len(features)):
            x += features[f] * self.weights[f]
        error = math.fabs(x - guess)
        txt = ''
        for t in range(0, len(features)):
            txt += str(features[t]) + ", "
        txt += ": " + str(guess) + " : " + str(x) + " : " + str(error)
        print txt
        return error

    def perceptron_train(self, features, guess, activation_method):
        # False positive prediction, set w = w - x and set theta = theta + 1
        x = 0
        for f in range(0, len(features)):
            x += features[f] * self.weights[f]
        x = self.activation(x, activation_method)
        txt = ""
        if x > 0 and x != guess:
            for i in range(0, len(self.weights)):
                self.weights[i] = self.weights[i] - features[i]
                self.theta += 1
                txt += str(features[i]) + ", "
            txt += ": " + str(x) + ": UPDATE"
        # False negative prediction, set w = w + x and set theta = theta -1
        elif x < 0 and x != guess:
            for j in range(0, len(self.weights)):
                self.weights[j] = self.weights[j] + features[j]
                self.theta -= 1
                txt += str(features[j]) + ", "
            txt += ": " + str(x) + ": UPDATE"
        elif x == guess:
            for k in range(0, len(self.weights)):
                txt += str(features[k]) + ", "
            txt += ": " + str(x) + ": no update"
        print txt

    def winnow_train(self, features, guess, activation_method):
        # On a false positive prediction, for all i, set wi := α ** −xi * wi
        x = 0
        for f in range(0, len(features)):
            x += features[f] * self.weights[f]
        x = self.activation(x, activation_method)
        txt = ""
        if x > 0 and x != guess:
            for i in range(0, len(self.weights)):
                self.weights[i] = (self.alpha ** (-int(features[i]))) * self.weights[i]
            txt += ": " + str(x) + ": UPDATE"
        # On a false negative prediction, for all i, set wi := α ** xi * wi
        elif x < 0 and x != guess:
            for i in range(0, len(self.weights)):
                
                self.weights[i] = (self.alpha ** features[i]) * self.weights[i]
            txt += ": " + str(x) + ": UPDATE"
        elif x == guess:
            for k in range(0, len(self.weights)):
                txt += str(features[k]) + ", "
            txt += ": " + str(x) + ": no update"
        print txt
