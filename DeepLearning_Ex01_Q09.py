import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
def fizz_buzz_prime_encode(prime):
    if (prime[1] > 0 and prime[2] > 0):
        return np.array([0, 0, 0, 1])
    elif (prime[1] == 0 and prime[2] > 0):
        return np.array([0, 0, 1, 0])
    elif (prime[1] > 0 and prime[2] == 0):
        return np.array([0, 1, 0, 0])
    else:
        return np.array([1, 0, 0, 0])

def fizz_buzz_result(i):
    if   i % 15 == 0: return 'fizzbuzz'
    elif i % 5  == 0: return 'buzz'
    elif i % 3  == 0: return 'fizz'
    else:             return i

def fizz_buzz_result_code(encode_result):
    encode_result_list = encode_result.tolist()
    list = []
    for result in encode_result_list:
        list.append(result.index(1))
    return list

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
trY = np.array([fizz_buzz_prime_encode(prime_encode(i))          for i in range(101, 2 ** NUM_DIGITS)])

#init weights of the last epoch, after running training with 100% accuracy
def init_weights_w_o():
    weights =   [[-3.733603,   3.8169994, -2.4922888,  2.4283876],
                 [ 4.2530627, -0.8944688, -1.008752,  -2.3385623],
                 [-3.0656772, -3.3521671,  3.864877,   2.5300734]]
    return tf.Variable(np.array(weights, dtype=float))

#init w_h weights of the last epoch, after running training with 100% accuracy
def init_weights_w_h():
    weights =   [[ 2.8464245e-02,  1.3385745e+00,  1.4537486e-04],
                 [ 6.3857436e+00, -1.8323094e+00,  1.2662148e-01],
                 [ 5.2814104e-02, -1.6129370e+00,  6.4842734e+00],
                 [-6.6229717e-07,  1.8772777e+00, -7.0314951e-02],
                 [-2.1282917e-02,  1.7452775e+00, -3.4460749e-02],
                 [-2.5101673e-02,  1.6849409e+00, -2.9270058e-02],
                 [-2.1898899e-02,  1.5763744e+00, -3.3776768e-02],
                 [-2.7350739e-02,  1.5400741e+00, -3.1757951e-02],
                 [-3.9761696e-02,  1.4494705e+00, -2.5453875e-02],
                 [-4.2063192e-02,  1.1602138e+00, -2.8603597e-02]]
    return tf.Variable(np.array(weights, dtype=float))

# Our model is a standard 1-hidden-layer multi-layer-perceptron with ReLU activation.
# The softmax (which turns arbitrary real-valued outputs into probabilities)
# gets applied in the cost function.
def model(X, w_h, w_o):
    X = tf.cast(X, tf.float32)
    w_h =tf.cast(w_h, tf.float32)
    w_o = tf.cast(w_o, tf.float32)

    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

# Our variables. The input has width NUM_DIGITS, and the output has width 4.
X = tf.compat.v1.placeholder("float", [None, NUM_DIGITS])
Y = tf.compat.v1.placeholder("float", [None, 4])

# How many units in the hidden layer.
NUM_HIDDEN = 3

# Initialize the weights.
w_h = init_weights_w_h()
w_o = init_weights_w_o()

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

# Launch the graph in a session
with tf.compat.v1.Session() as sess:

    # grab a tensorflow session and initialize the variables.
    tf.compat.v1.global_variables_initializer().run()

    # Print the current accuracy on the training data.
    print("Accuracy:",np.mean(np.argmax(trY, axis=1) == sess.run(predict_op, feed_dict={X: trX, Y: trY})),)
    print("Weights-w_o:\n", w_o.eval(sess))
    print("Weights-w_h:\n", w_h.eval(sess))

    # And now for some fizz buzz
    numbers = np.arange(1, 101)
    teX = np.array([prime_encode(i) for i in numbers])
    teY = sess.run(predict_op, feed_dict={X: teX})

    output = np.vectorize(fizz_buzz)(numbers, teY)

    print("\nOutput #09: \n", output)

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
