import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import  shuffle

## Getting the data from CIFAR data set
height = 32
width = 32
depth = 3
numClasses = 10


W_conv1 = 0.0
W_conv2 = 0.0
b_conv1 = 0.0
b_conv2 = 0.0

size_conv1 = 5
size_conv2 = 5
depth_conv1 = 64
depth_conv2 = 64

W_fc1   = 0.0
W_fc2   = 0.0

b_fc1   = 0.0
b_fc2   = 0.0

size_fc1 = 384
size_fc2 = 192

W_out = 0.0
b_out = 0.0

epochs = 10
initialLearningRate = 0.0001
batchSize = 500

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def reshapeData(x):
    ##sess = tf.InteractiveSession()
    ## Data will be loaded into [depth height width]
    ## -1 since batch mode
    y = tf.reshape(x,[-1,depth,height,width])
    ## w.r.t each data in batch data [depth height width] -> [height width depth]
    ## i.e [batch depth height width] -> [batch height width depth]
    z = tf.transpose(y, [0, 2, 3, 1])
    z = tf.cast(z,tf.float32)
    ##print(z.eval())
    return z

def convLayer(input,phase=True):

    ## Weights and bias of the convolution layer
    global W_conv1, W_conv2, b_conv1, b_conv2
    ## Number of filters in each convolution layer
    global depth_conv1,depth_conv2
    ## kernal size
    global size_conv1,size_conv2
    ## Fully connected layer size
    global size_fc1,size_fc2
    global W_fc1, W_fc2, b_fc1, b_fc2

    global W_out,b_out

    ## First convolution Layer
    W_conv1 = tf.Variable(tf.random_normal([size_conv1,size_conv1,depth,depth_conv1]),name="W_conv1")
    b_conv1 =  tf.Variable(tf.random_normal([depth_conv1]),name="b_conv1")
    ## Convolution layer
    conv1 = tf.nn.conv2d(input,W_conv1,[1,1,1,1],padding='SAME',name="conv1") + b_conv1
    ## Max pooling
    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')
    ## Normalise
    norm1 = tf.contrib.layers.batch_norm(pool1,center=True, scale=True,is_training=phase)
    relu1 = tf.nn.relu(norm1,name="relu1")

    ## Second Convolution Layer
    W_conv2 = tf.Variable(tf.random_normal([size_conv2, size_conv2, depth_conv1, depth_conv2]), name="W_conv2")
    b_conv2 = tf.Variable(tf.random_normal([depth_conv2]), name="b_conv2")
    conv2 = tf.nn.conv2d(relu1,W_conv2,[1,1,1,1],padding='SAME',name="conv2") + b_conv2
    ## Maxpooling
    pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')
    ## Normalisation
    norm2  = tf.contrib.layers.batch_norm(pool2,center=True,scale=True,is_training=phase)
    relu2 = tf.nn.relu(norm2,name='relu2')

    ## Divide by 16 because two convolution layers each convolution layer both height and width has a stride of 2
    sizeAfterConv = int(height * width * depth_conv2 / 16)
    print(sizeAfterConv)

    inpFc = tf.reshape(relu2,[-1,sizeAfterConv])
    ## Weights for fully connected Layer
    W_fc1 = tf.Variable(tf.random_normal([sizeAfterConv,size_fc1]),name="W_fc1")
    b_fc1 = tf.Variable(tf.random_normal([size_fc1]),name="b_fc1")
    fc1 = tf.matmul(inpFc,W_fc1) + b_fc1
    norm_fc1 = tf.contrib.layers.batch_norm(fc1,center=True,scale=True,is_training=phase)
    relu_fc1 = tf.nn.relu(norm_fc1,name="relu_fc1")

    W_fc2 = tf.Variable(tf.random_normal([size_fc1,size_fc2]),name="W_fc2")
    b_fc2 = tf.Variable(tf.random_normal([size_fc2]),name="b_fc2")
    fc2 = tf.matmul(relu_fc1,W_fc2) + b_fc2
    norm_fc2 = tf.contrib.layers.batch_norm(fc2,center=True,scale=True,is_training=phase)
    relu_fc2 = tf.nn.relu(norm_fc2,name="relu_fc2")

    ## Final Layer
    W_out = tf.Variable(tf.random_normal([size_fc2,numClasses]),name="W_out")
    b_out = tf.Variable(tf.random_normal([numClasses]),name="b_out")
    out = tf.matmul(relu_fc2,W_out) + b_out

    return out

def oneHotVector(inpArray):
    inpArray = np.array(inpArray)
    inpLenth = inpArray.size
    oneHotVec = np.zeros((inpLenth, numClasses))
    oneHotVec[np.arange(inpLenth), inpArray] = 1
    return oneHotVec

def splitData():
    print("Splitting Data started")
    data_batch_1 = "/Users/sainikhilmaram/OneDrive/UCSB courses/Winter 2018/Advance Topics in Computer Vision/HW2/cifar-10-batches-py/data_batch_1"
    data_batch_2 = "/Users/sainikhilmaram/OneDrive/UCSB courses/Winter 2018/Advance Topics in Computer Vision/HW2/cifar-10-batches-py/data_batch_2"
    data_batch_3 = "/Users/sainikhilmaram/OneDrive/UCSB courses/Winter 2018/Advance Topics in Computer Vision/HW2/cifar-10-batches-py/data_batch_3"
    data_batch_4 = "/Users/sainikhilmaram/OneDrive/UCSB courses/Winter 2018/Advance Topics in Computer Vision/HW2/cifar-10-batches-py/data_batch_4"
    data_batch_5 = "/Users/sainikhilmaram/OneDrive/UCSB courses/Winter 2018/Advance Topics in Computer Vision/HW2/cifar-10-batches-py/data_batch_5"

    test_batch  =  "/Users/sainikhilmaram/OneDrive/UCSB courses/Winter 2018/Advance Topics in Computer Vision/HW2/cifar-10-batches-py/test_batch"

    data = []
    labels = []
    unPickledBatch = unpickle(data_batch_1)
    data.extend(unPickledBatch[b'data'])
    labels.extend(unPickledBatch[b'labels'])

    unPickledBatch = unpickle(data_batch_2)
    data.extend(unPickledBatch[b'data'])
    labels.extend(unPickledBatch[b'labels'])

    unPickledBatch = unpickle(data_batch_3)
    data.extend(unPickledBatch[b'data'])
    labels.extend(unPickledBatch[b'labels'])

    unPickledBatch = unpickle(data_batch_4)
    data.extend(unPickledBatch[b'data'])
    labels.extend(unPickledBatch[b'labels'])

    unPickledBatch = unpickle(data_batch_5)
    data.extend(unPickledBatch[b'data'])
    labels.extend(unPickledBatch[b'labels'])

    data = np.array(data)
    labels = np.array(labels)

    ## Convert the labels into one hot vectors
    oneHotLabels = oneHotVector(labels)
    test_data = []
    test_labels = []

    unPickledBatch = unpickle(test_batch)
    test_data = np.array(test_data.extend(unPickledBatch[b'data']))
    test_labels = np.array(test_labels.extend(unPickledBatch[b'labels']))

    training_data, validation_data, train_labels, validation_labels = train_test_split(data, oneHotLabels, test_size=0.1,random_state=55)
    print("Splitting Data Ended")
    return training_data, validation_data,test_data, train_labels, validation_labels,test_labels


def training():
    print("Started Training")

    training_data, validation_data, test_data, training_labels, validation_labels, test_labels = splitData()
    len_training_data = len(training_data)
    len_validation_data = len(validation_data)

    inputBytes = height * width * depth

    ## Place holders for input pixels and labels
    inputPixels = tf.placeholder(tf.float32,[None,inputBytes])
    inputLabels = tf.placeholder(tf.float32,[None,numClasses])

    ## input data is reshaped before feeding into convolution layer
    inputData = reshapeData(inputPixels)
    ## Correct predictions are obtained
    prediction = convLayer(inputData)

    numberPredictions = tf.equal(tf.argmax(prediction,1),tf.argmax(inputLabels,1))
    accuracy = tf.reduce_mean(tf.cast(numberPredictions,tf.float32))

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=inputLabels)
    optimizer = tf.train.AdamOptimizer(initialLearningRate).minimize(loss)

    prevAccuracy = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            print("In epoch" , i)
            ## Shuffle the training data for each epoch
            training_data, training_labels = shuffle(training_data,training_labels)
            index = 0
            while(index + batchSize < len_training_data):
                sess.run(optimizer,feed_dict={inputPixels:training_data[index:index+batchSize],inputLabels:training_labels[index:index+batchSize]})

                if(index in [10000,20000,30000,40000]):
                    print(sess.run(accuracy,feed_dict={inputPixels:training_data[index:index+10],inputLabels:training_labels[index:index+10]}))
                index = index + batchSize

            ## Checking the Validation accuracy
            index_valid = 0
            currAccuracy = 0
            while(index_valid + batchSize < len_validation_data):
                currAccuracy += sess.run(accuracy,feed_dict={inputPixels:validation_data[index_valid:index_valid+batchSize],
                                                             inputLabels:validation_labels[index_valid:index_valid+batchSize]})

            print(currAccuracy)






if __name__ == "__main__":

    data_batch_1 = "/Users/sainikhilmaram/OneDrive/UCSB courses/Winter 2018/Advance Topics in Computer Vision/HW2/cifar-10-batches-py/data_batch_1"
    training()
    # print(oneHotVector([2,3,4,5]))
    # data = unpickle(data_batch_1)
    # print(data.keys())
    ##print((data[b'data'][1][1024]))
    #a = np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]])
    #inputData = reshapeData(a)

    #print(data[b'labels'])
    # inputBytes = height * width * depth
    # input = tf.placeholder(tf.float32,[None,inputBytes])
    # inputData = reshapeData(input)
    # output = convLayer(inputData)
    # #with tf.Session() as sess:
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # sess.run(output,feed_dict={input:data[b'data'][:15]})

