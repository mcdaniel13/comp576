from imageio import imread
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mp

if tf.__version__.split(".")[0] == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()


# --------------------------------------------------
# setup

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], "SAME")


def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")


ntrain = 1000  # per class
ntest = 100  # per class
nclass = 10  # number of classes
imsize = 28
nchannels = 1
batchsize = 100

Train = np.zeros((ntrain * nclass, imsize, imsize, nchannels))
Test = np.zeros((ntest * nclass, imsize, imsize, nchannels))
LTrain = np.zeros((ntrain * nclass, nclass))
LTest = np.zeros((ntest * nclass, nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = './CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
        im = imread(path)  # 28 by 28
        im = im.astype(float) / 255
        itrain += 1
        Train[itrain, :, :, 0] = im
        LTrain[itrain, iclass] = 1  # 1-hot lable
    for isample in range(0, ntest):
        path = './CIFAR10/Test/%d/Image%05d.png' % (iclass, isample)
        im = imread(path)  # 28 by 28
        im = im.astype(float) / 255
        itest += 1
        Test[itest, :, :, 0] = im
        LTest[itest, iclass] = 1  # 1-hot lable

sess = tf.InteractiveSession()

# tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_data = tf.placeholder(tf.float32, [None, imsize, imsize, nchannels])
tf_labels = tf.placeholder(tf.float32, [None, nclass])  # tf variable for labels

# --------------------------------------------------
# model
# create your model

# Convolutional layer with kernel 5 x 5 and 32 filter maps followed by ReLU
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(tf_data, W_conv1) + b_conv1)

# Max Pooling layer subsampling by 2
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer with kernel 5 x 5 and 64 filter maps followed by ReLU
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# Max Pooling layer subsampling by 2
h_pool2 = max_pool_2x2(h_conv2)

# Fully Connected layer that has input 7*7*64 and output 1024
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully Connected layer that has input 1024 and output 10 (for the classes)
# Softmax layer (Softmax Regression + Softmax Nonlinearity)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# --------------------------------------------------
# loss
# set up the loss, optimization, evaluation, and accuracy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_labels, logits=y_conv))
optimizer = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --------------------------------------------------
# optimization
sess.run(tf.global_variables_initializer())

sess.run(tf.initialize_all_variables())
# setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_xs = np.zeros((batchsize, imsize, imsize, nchannels))
# setup as [batchsize, the how many classes]
batch_ys = np.zeros((batchsize, nclass))

accuracy_list = []
train_loss_list = []

for i in range(1000):  # try a small iteration size once it works then continue
    perm = np.arange(ntrain * nclass)
    np.random.shuffle(perm)
    feed = {tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5}
    for j in range(batchsize):
        batch_xs[j, :, :, :] = Train[perm[j], :, :, :]
        batch_ys[j, :] = LTrain[perm[j], :]
    if i % 10 == 0:
        # calculate train accuracy and print it
        print("@it = " + str(i) + ", accuracy = " + str(accuracy.eval(feed_dict=feed)) + " train loss = " + str(
            cross_entropy.eval(feed_dict=feed)))

    # Accuracy and loss
    accuracy_list.append(accuracy.eval(feed_dict=feed))
    train_loss_list.append(cross_entropy.eval(feed_dict=feed))

    # First convolution layer's weights
    w1 = W_conv1.eval()

    # dropout only during training
    optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})

# --------------------------------------------------
# test

# Plot train/test accuracy and train loss
print("test accuracy %g" % accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))

plt.plot(range(len(accuracy_list)), accuracy_list, label="accuracy")
plt.show()

plt.plot(range(len(train_loss_list)), train_loss_list, label="loss")
plt.show()

# Visualize the first convolutional layer's weights
fig = plt.figure()
for i in range(32):
    ax = fig.add_subplot(8, 4, 1 + i)
    ax.imshow(w1[:, :, 0, i], cmap='gray')
    plt.axis('off')
plt.show()

# Show the statistics of the activation in the convolutional layers
a1 = h_conv1.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
a2 = h_conv2.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})

print("a1: mean = " + str(np.mean(np.array(a1))) + " var = " + str(np.var(np.array(a1))))
print("a2: mean = " + str(np.mean(np.array(a2))) + " var = " + str(np.var(np.array(a2))))

sess.close()
