import sys
import utils

from perceptron import Perception

print 'Arguments list:', sys.argv

if utils.validate_arguments(sys.argv):
    args = sys.argv
    activation_method = args[1]
    training_alg = args[2]
    ground_file_name = args[3]
    distribution = args[4]
    num_train = int(args[5])
    num_test = int(args[6])
    epsilon = float(args[7])
    neuron = Perception()

    t, p, n = utils.parse_ground_file(ground_file_name)
    # Load the weight, target, and number of feature from ground file (this is for threshold function case)
    neuron.target = float(t)
    neuron.weights = [float(f) for f in p]
    neuron.features_num = int(n)

    print "Weights:", neuron.weights
    example_features, example_outputs = utils.generate_example(distribution, num_train, neuron.features_num)

    print 'Start training....\n'
    for i in range(0, num_train):
        # print "Features: ", example_features[i],
        # print 'Output: ', example_outputs[i]
        print 'Weight before training:', neuron.weights
        neuron.train(example_features[i], example_outputs[i], Perception.PERCEPTRON_ALG)
        print 'Weight after trained:', neuron.weights, '\n'
    print 'Training completed!!!'

    print '============== Start Testing =============\n'
    test_features, test_outputs = utils.generate_test(num_test, neuron.features_num)
    sum_error = 0
    for i in range(0, num_test):
        sum_error += neuron.test(test_features[i], test_outputs[i])

    # TODO: double check with group mate whether it's divided by number of tests or number of failed tests
    average_error = sum_error / num_test
    print 'Average Error:', average_error
    print 'Epsilon: ', epsilon
    if average_error <= epsilon:
        print 'TRAINING SUCCEEDED'
    else:
        print 'TRAINING FAILED'
