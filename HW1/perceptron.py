import random
import sys
import utils

from neuron import Neuron

# print 'Arguments list:', sys.argv

if utils.validate_arguments(sys.argv):
    args = sys.argv
    activation_method = args[1]
    training_alg = args[2]
    ground_file_name = args[3]
    distribution = args[4]
    num_train = int(args[5])
    num_test = int(args[6])
    epsilon = float(args[7])

    neuron = Neuron()

    param, n, func_type, target = utils.parse_ground_file(ground_file_name)
    # Load the target, and number of feature from ground file (this is for threshold function case)
    neuron.target = float(target)
    neuron.features_num = int(n)

    # Generate random weights
    neuron.weights = [random.random() for i in range(n)]
    print "Weights:", neuron.weights
    # Generate examples
    example_features = []
    for i in range(0, num_train):
        vector = utils.gen_vector(distribution, n, func_type)
        example_features.append(vector)
    # print 'Example features:', example_features

    print 'Start training....\n'
    for e in example_features:
        # Generate example data
        # example_features = utils.gen_vector(distribution, n, func_type)
        # print 'Weight before training:', neuron.weights
        neuron.train(e, training_alg, param, func_type, float(target), activation_method)
        # print 'Weight after trained:', neuron.weights, '\n'
    print 'Training completed!!!'
    print 'Theta after training:', neuron.theta

    print '============== Start Testing =============\n'
    # Generate test data
    test_features = []
    for i in range(0, num_test):
        vector = utils.gen_vector(distribution, n, func_type)
        test_features.append(vector)
    # print 'Test data:', test_features

    sum_error = 0
    for t in test_features:
        sum_error += neuron.test(t, param, func_type, float(target), activation_method)
    average_error = sum_error / num_test
    print 'Average Error:', average_error
    print 'Epsilon: ', epsilon
    if average_error <= epsilon:
        print 'TRAINING SUCCEEDED'
    else:
        print 'TRAINING FAILED'
