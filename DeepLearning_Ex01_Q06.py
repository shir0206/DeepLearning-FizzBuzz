import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("Deep Learning - Ex01 - Q06")

NUM_DIGITS = 10

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def prime_encode(n):

    number_primes = prime_factors(n)

    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    size = len(all_primes)
    factors_encode = [0] * size

    for prime in number_primes:
        if prime not in all_primes:
            break
        else:
            prime_pos = all_primes.index(prime)
            if(prime_pos < all_primes[size-1]):
                factors_encode[prime_pos] += 1
            else:
                factors_encode[len(factors_encode)-1] += 1
    return factors_encode

# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])

def fizz_buzz_result(i):
    if   i % 15 == 0: return 'fizzbuzz'
    elif i % 5  == 0: return 'buzz'
    elif i % 3  == 0: return 'fizz'
    else:             return i

def check_digit(i):
    if str.isdigit(i): return 'Digit'
    else: return i

# Generate training data.
# Our goal is to produce fizzbuzz for the numbers 1 to 100.
# So it would be unfair to include these in our training data.
# Accordingly, the training data corresponds to the numbers 101 to (2 ** NUM_DIGITS - 1).
# trX = numbers from 101 to 1023 in binary for training/"studying"
# trY = the "output" of the training/"studying" - the catagorize of fizz/buzz/fizzbuzz/none in binary
trX = np.array([prime_encode(i) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** NUM_DIGITS)])

# We'll want to randomly initialize weights.
def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.01))

# Our model is a standard 1-hidden-layer multi-layer-perceptron with ReLU activation.
# The softmax (which turns arbitrary real-valued outputs into probabilities)
# gets applied in the cost function.
def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

# Our variables. The input has width NUM_DIGITS, and the output has width 4.
X = tf.compat.v1.placeholder("float", [None, NUM_DIGITS])
Y = tf.compat.v1.placeholder("float", [None, 4])

# How many units in the hidden layer.
NUM_HIDDEN = 100

# Initialize the weights.
w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 4])

# Predict y given x using the model.
py_x = model(X, w_h, w_o)

# We'll train our model by minimizing a cost function.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.compat.v1.train.GradientDescentOptimizer(0.05).minimize(cost)

# And we'll make predictions by choosing the largest output.
predict_op = tf.argmax(py_x, 1)

# Finally, we need a way to turn a prediction (and an original number)
# into a fizz buzz output
def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

# Each epoch we'll train in batches of 128 inputs
BATCH_SIZE = 128

# Train the model.
# Launch the graph in a session
with tf.compat.v1.Session() as sess:

    # grab a tensorflow session and initialize the variables.
    tf.compat.v1.global_variables_initializer().run()

    # 1000 epochs of training
    # Epoch = Going once through the entire training data.
    for epoch in range(10000):
        # Shuffle the data before each training iteration.
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

        # Train inputs of 101 to 1024, in batches of 128 inputs.
        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        # Print the current accuracy on the training data.
        print("Epoch #",epoch, " Accuracy:",np.mean(np.argmax(trY, axis=1) == sess.run(predict_op, feed_dict={X: trX, Y: trY})),)

    # And now for some fizz buzz
    numbers = np.arange(1, 101)
    teX = np.array([prime_encode(i) for i in numbers])
    teY = sess.run(predict_op, feed_dict={X: teX})
    output = np.vectorize(fizz_buzz)(numbers, teY)

    print("\nOutput #06: \n", output)

    the_correct = np.array([fizz_buzz_result(i) for i in range(1, 101)])

# Check output result vs correct result and print the classifier accuracy
check_result = np.frompyfunc(check_digit, 1, 1)

output_result = check_result(np.array(output))
correct_result = check_result(np.array(the_correct))

print("\nThe Classifier Accuracy: " + str(accuracy_score(correct_result, output_result)))

# Generate confusion matrix
conf_matrix = confusion_matrix(correct_result, output_result)
print("\nConfusion Matrix:\n", str(conf_matrix))

labels = ['Number', 'fizz', 'buzz', 'fizzbuzz']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_matrix)
plt.title('Confusion Matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Output Result')
plt.ylabel('Correct Result')
plt.show()
