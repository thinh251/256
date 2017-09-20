import os
import random


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
        nf = len(expression.split(" ")) / 2  # n is needed to figure out the number of features
        return [expression, nf]
    elif func_type == "TF":
        target = gfile.readline()
        params = gfile.readline().split(" ")
        nf = len(params)  # n is needed to figure out the number of features
        return [target, params, nf]
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
    # TODO: validate distribution
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

def gen_vector(distribution, n):
    vector = []
    if distribution == "bool":
        for i in range(n):
            bit = random.randint(0, 1)
            vector.append(bit)
        return vector
    elif distribution == "sphere":
        norm = 0
        for i in range(n):
            bit = random.randint()
            norm += bit^2
            vector.append(bit)
        norm = norm^(1/2)
        return vector/norm

def generate_test(number_of_test, number_of_feature):
    inputs = [[random.randint(0, 1) for i in range(number_of_feature)] for j in range(number_of_test)]
    output = [random.randint(0, 1) for i in range(number_of_test)]
    # print 'Feature from generate example function:', features
    return inputs, output
