import os
import random
import math

def is_number(txt):
    try:
        int(txt)
        return True
    except ValueError:
        try:
            float(txt)
            return True
        except ValueError:
            return False


def parse_ground_file(ground_file):
    gfile = open(ground_file)
    func_type = gfile.readline().strip()
    if func_type == "NBF":
        expression = gfile.readline()
        token = expression.split(" ")
        x = []
        for t in token:
            if is_number(t):
                x.append(int(t))
        n = max(x)
        return [expression.lower(), n, func_type, 0]
    elif func_type == "TF":
        target = gfile.readline()
        param = gfile.readline().split(" ")
        n = len(param)  # n is needed to figure out the number of features
        return [param, n, func_type, target]
    else:
        print "NOT PARSEABLE"
        exit()


def validate_arguments(arguments):
    if len(arguments) < 8:
        print ('Missing arguments')
        return False
    if arguments[1] != 'relu' and arguments[1] != 'tanh' and arguments[1] != 'threshold':
        print ('Activation type is not supported')
        return False
    if arguments[2] != 'perceptron' and arguments[2] != 'winnow':
        print ('Training algorithm is not supported')
        return False
    if not os.path.exists(arguments[3]) or (os.path.getsize(arguments[3]) <= 0):
        print ('Ground file does not exist or empty')
        return False
    if arguments[4] != 'bool' and arguments[4] != 'sphere':
        print "Distribution attribute is either bool or sphere"
        return False
    if not is_number(arguments[5]):
        print arguments[5]
        print ('Number of train should be a number')
        return False
    if not is_number(arguments[6]):
        print ('Number of test should be a number')
        return False
    if not is_number(arguments[7]):
        print ('Epsilon should be a number')
        return False
    # All the test passed
    return True


def gen_vector(distribution, n, func_type):
    vector = []
    if distribution == "bool" or func_type == "NBF":
        for i in range(0, n):
            bit = random.randint(0, 1)
            vector.append(bit)
        return vector
    elif distribution == "sphere" and func_type == "TF":
        norm = 0
        for i in range(0, n):
            bit = random.random()
            norm += bit**2
            vector.append(bit)
        norm = norm**(1/2)
        for i in range(len(vector)):
            vector[i] = vector[i]/norm
        print ' vector in gen', vector
        return vector


def generate_test(number_of_test, number_of_feature):
    inputs = [[random.randint(0, 1) for i in range(number_of_feature)] for j in range(number_of_test)]
    # print 'Feature from generate example function:', features
    return inputs
