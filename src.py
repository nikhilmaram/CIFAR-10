import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import  shuffle
import time

## Getting the data from CIFAR data set
height = 32
width = 32
depth = 3
numClasses = 10


W_conv1 = 0.0
W_conv2 = 0.0

b_conv1 = 0.0
b_conv2 = 0.0

## Number of filters in each convolution layer
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

epochs = 100
initialLearningRate = 0.1
batchSize = 128

model_path = "/tmp/model.ckpt"

def backupNetwork() :
    global W_fc1,   W_fc2,    b_fc1,   b_fc2
    global WB_fc1,  WB_fc2,  WB_out2, bB_fc1,  bB_fc2,  bB_out2
    global W_conv1,  W_conv2,  b_conv1,  b_conv2
    global WB_conv1, WB_conv2, bB_conv1, bB_conv2
    global W_out,b_out
    global WB_out,bB_out

    WB_fc1      = W_fc1
    WB_fc2      = W_fc2

    WB_conv1    = W_conv1
    WB_conv2    = W_conv2
    bB_fc1      = b_fc1
    bB_fc2      = b_fc2

    bB_conv1    = b_conv1
    bB_conv2    = b_conv2

    WB_out = W_out
    bB_out = b_out

def restoreNetwork() :
    global W_fc1,   W_fc2,    b_fc1,   b_fc2
    global WB_fc1,  WB_fc2,  WB_out2, bB_fc1,  bB_fc2,  bB_out2
    global W_conv1,  W_conv2,  b_conv1,  b_conv2
    global WB_conv1, WB_conv2, bB_conv1, bB_conv2
    global W_out,b_out
    global WB_out,bB_out

    W_fc1      = WB_fc1
    W_fc2      = WB_fc2

    W_conv1    = WB_conv1
    W_conv2    = WB_conv2
    b_fc1      = bB_fc1
    b_fc2      = bB_fc2

    b_conv1    = bB_conv1
    b_conv2    = bB_conv2

    W_out = W_out
    b_out = b_out

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

    ## Fully connected layer size
    global W_fc1, W_fc2, b_fc1, b_fc2

    global W_out,b_out

    ## First convolution Layer
    ##W_conv1 = tf.Variable(tf.random_normal([size_conv1,size_conv1,depth,depth_conv1]),name="W_conv1")
    W_conv1 = tf.get_variable("W_conv1",shape = [size_conv1,size_conv1,depth,depth_conv1],initializer=tf.contrib.layers.variance_scaling_initializer())

    b_conv1 =  tf.Variable(tf.random_normal([depth_conv1]),name="b_conv1")
    ## Convolution layer
    conv1 = tf.nn.conv2d(input,W_conv1,[1,1,1,1],padding='SAME',name="conv1") + b_conv1
    relu1 = tf.nn.relu(conv1, name="relu1")
    ## Max pooling
    pool1 = tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')

    batch_mean1,batch_var1 = tf.nn.moments(pool1,[0])
    norm1 = tf.nn.batch_normalization(pool1, batch_mean1, batch_var1, variance_epsilon=1e-8,scale=1,offset=1e-8)


    ## Second Convolution Layer
    ##W_conv2 = tf.Variable(tf.random_normal([size_conv2, size_conv2, depth_conv1, depth_conv2]), name="W_conv2")
    W_conv2 = tf.get_variable("W_conv2", shape=[size_conv2, size_conv2, depth_conv1, depth_conv2],initializer=tf.contrib.layers.variance_scaling_initializer())
    b_conv2 = tf.Variable(tf.random_normal([depth_conv2]), name="b_conv2")
    conv2 = tf.nn.conv2d(norm1,W_conv2,[1,1,1,1],padding='SAME',name="conv2") + b_conv2
    relu2 = tf.nn.relu(conv2, name='relu2')
    ## Maxpooling
    pool2 = tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')

    batch_mean2, batch_var2 = tf.nn.moments(pool2, [0])
    norm2 = tf.nn.batch_normalization(pool2, batch_mean2, batch_var2, variance_epsilon=1e-8,scale=1,offset=1e-8)


    ## Divide by 16 because two convolution layers each convolution layer both height and width has a stride of 2
    sizeAfterConv = int(height * width * depth_conv2 / 16)
    print(sizeAfterConv)

    inpFc = tf.reshape(norm2,[-1,sizeAfterConv])
    ## Weights for fully connected Layer
    ##W_fc1 = tf.Variable(tf.random_normal([sizeAfterConv,size_fc1]),name="W_fc1")
    W_fc1 = tf.get_variable("W_fc1",shape=[sizeAfterConv,size_fc1],initializer=tf.contrib.layers.variance_scaling_initializer())
    b_fc1 = tf.Variable(tf.random_normal([size_fc1]),name="b_fc1")
    fc1 = tf.matmul(inpFc,W_fc1) + b_fc1
    relu_fc1 = tf.nn.relu(fc1, name="relu_fc1")
    #norm_fc1 = tf.contrib.layers.batch_norm(relu_fc1,center=True,scale=True,is_training=phase)

    batch_mean_fc1, batch_var_fc1 = tf.nn.moments(relu_fc1, [0])
    norm_fc1 = tf.nn.batch_normalization(relu_fc1, batch_mean_fc1, batch_var_fc1, variance_epsilon=1e-8,scale=1,offset=1e-8)

    ## Added Dropout
    norm_fc1 = tf.nn.dropout(norm_fc1,keep_prob=0.9)

    ##W_fc2 = tf.Variable(tf.random_normal([size_fc1,size_fc2]),name="W_fc2")
    W_fc2 = tf.get_variable("W_fc2",shape= [size_fc1,size_fc2],initializer=tf.contrib.layers.variance_scaling_initializer())
    b_fc2 = tf.Variable(tf.random_normal([size_fc2]),name="b_fc2")
    fc2 = tf.matmul(norm_fc1,W_fc2) + b_fc2
    relu_fc2 = tf.nn.relu(fc2, name="relu_fc2")

    #norm_fc2 = tf.contrib.layers.batch_norm(relu_fc2,center=True,scale=True,is_training=phase)
    batch_mean_fc2, batch_var_fc2 = tf.nn.moments(relu_fc2, [0])
    norm_fc2 = tf.nn.batch_normalization(relu_fc2, batch_mean_fc2, batch_var_fc2, variance_epsilon=1e-8,scale=1,offset=1e-8)

    norm_fc2 = tf.nn.dropout(norm_fc2, keep_prob=0.9)
    ## Final Layer
    W_out = tf.Variable(tf.random_normal([size_fc2,numClasses]),name="W_out")
    b_out = tf.Variable(tf.random_normal([numClasses]),name="b_out")
    out = tf.matmul(norm_fc2,W_out) + b_out

    return out

def oneHotVector(inpArray):
    inpArray = np.array(inpArray)
    inpLenth = inpArray.size
    oneHotVec = np.zeros((inpLenth, numClasses))
    oneHotVec[np.arange(inpLenth), inpArray] = 1
    return oneHotVec

def splitData():
    print("Splitting Data started")
    data_batch_1 = "../cifar-10-batches-py/data_batch_1"
    data_batch_2 = "../cifar-10-batches-py/data_batch_2"
    data_batch_3 = "../cifar-10-batches-py/data_batch_3"
    data_batch_4 = "../cifar-10-batches-py/data_batch_4"
    data_batch_5 = "../cifar-10-batches-py/data_batch_5"

    test_batch  =  "../cifar-10-batches-py/test_batch"

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
    test_data.extend(unPickledBatch[b'data'])
    test_labels.extend(unPickledBatch[b'labels'])

    test_data = np.array(test_data)
    test_labels = oneHotVector(np.array(test_labels))

    training_data, validation_data, train_labels, validation_labels = train_test_split(data, oneHotLabels, test_size=0.1,random_state=55)
    print("Splitting Data Ended")
    return training_data, validation_data,test_data, train_labels, validation_labels,test_labels

def saveFile(accuracyList):
    f = open("./accuracy.txt","w")
    for elem in accuracyList:
        f.write(str(elem)+ '\n')


def training():
    print("Started Training")

    try:
        training_data, validation_data, test_data, training_labels, validation_labels, test_labels = splitData()
    except:
        print("Please have the data in CIFAR-10 directory")
        return
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
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #for i in range(epochs):
        epoch = 0
        accuracyList = []
        ## Run till the max Epochs is reached or reached saturation
        while(1):
            print("In epoch" , epoch)
            start_time = time.time()
            ## Shuffle the training data for each epoch
            training_data, training_labels = shuffle(training_data,training_labels)
            index = 0
            while(index + batchSize < len_training_data):
                sess.run(optimizer,feed_dict={inputPixels:training_data[index:index+batchSize],inputLabels:training_labels[index:index+batchSize]})

                #if(index in [10000,20000,30000,40000]):
                #    print(sess.run(accuracy,feed_dict={inputPixels:training_data[index:index+10],inputLabels:training_labels[index:index+10]}))
                index = index + batchSize

            ## Checking the Validation accuracy
            index_valid = 0
            currAccuracyList = []

            ## Running till the epochSize
            while(index_valid + batchSize < len_validation_data):
                currAccuracyList.append(sess.run(accuracy,feed_dict={inputPixels:validation_data[index_valid:index_valid+batchSize],
                                                             inputLabels:validation_labels[index_valid:index_valid+batchSize]}))
                index_valid = index_valid + batchSize
            currAccuracy = np.mean(currAccuracyList)
            accuracyList.append(currAccuracy)
            print("epoch time : ",time.time()-start_time)
            print(currAccuracy)
            ## if overfitting is achieved
            if(currAccuracy + 0.05 <  prevAccuracy):
                print("Saturation has been achieved")
                print("Epoch : ",epoch+1)
                restoreNetwork()
                saver.save(sess,model_path)
                ## Testing the test accuracy
                print(sess.run(accuracy, feed_dict={inputPixels: test_data,
                                                    inputLabels: test_labels}))
                saveFile(accuracyList)
                break
            prevAccuracy = currAccuracy

            epoch = epoch + 1
            ## So not to go infinite loop
            if(epoch == epochs):
                print("Max epochs crossed")
                print("Test Accuracy")
                saver.save(sess, model_path)
                ## Testing the test accuracy
                print(sess.run(accuracy,feed_dict={inputPixels:test_data,
                                                    inputLabels:test_labels}))
                saver.save(sess, model_path)
                saveFile(accuracyList)
                break
            backupNetwork()

if __name__ == "__main__":
    training()
