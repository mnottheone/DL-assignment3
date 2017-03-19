'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Aman Kumar Singh
Roll No.: 13EC10064

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf

n_classes = 10
batch_size = 50

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



c1 = tf.Variable(tf.random_normal([5,5,1,32])),
c2 = tf.Variable(tf.random_normal([5,5,32,16])),
fcc = tf.Variable(tf.random_normal([7*7*16,1024])),
out = tf.Variable(tf.random_normal([1024, n_classes]))

b1 = tf.Variable(tf.random_normal([32])),
b2 = tf.Variable(tf.random_normal([16])),
b3 = tf.Variable(tf.random_normal([1024])),
b4 = tf.Variable(tf.random_normal([n_classes]))

x_im = tf.reshape(x, shape=[-1, 28, 28, 1])

conv1 = tf.nn.relu(conv2d(x_im, c1[0]) + b1)
conv1 = maxpool2d(conv1)
    
conv2 = tf.nn.relu(conv2d(conv1, c2[0]) + b2)
conv2 = maxpool2d(conv2)

fc = tf.reshape(conv2,[-1,7*7*16])
fc = tf.nn.relu(tf.matmul(fc,fcc[0])+b3)
fc = tf.nn.dropout(fc, keep_rate)

prediction = tf.matmul(fc,out)+b4


def train(trainX, trainY):

    trainX = trainX.reshape(len(trainX),-1)
    trainX = trainX/trainX.max()
    onehot = np.zeros((trainY.shape[0], 10))
    onehot[np.arange(trainY.shape[0]), trainY] = 1
    trainY = onehot

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
    saver = tf.train.Saver() 

    

    epochs = 6
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(epochs):
			epoch_loss = 0
			for i in range(int(trainX.shape[0]/batch_size)):
				n = min(batch_size, trainX.shape[0]-i*batch_size)
				epoch_x = trainX[i*batch_size:i*batch_size+n]
				epoch_y = trainY[i*batch_size:i*batch_size+n]
				_, c , pred = sess.run([optimizer, cost, prediction ], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
				acc = np.mean(np.argmax(pred,axis=1) == np.argmax(trainY[i*batch_size:i*batch_size+n], axis=1) )
				if i%100==0 :
					print (i,c,acc)
			print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)

        saver.save(sess, './cnn-model.cpkt')


def test(testX):
	testX = testX.reshape(len(testX),-1)
	testX = testX/testX.max()
	sess = tf.Session()
	new_saver = tf.train.Saver()
	new_saver.restore(sess, './cnn-model.cpkt')
	sess.run(tf.all_variables())
	
	
	pred = []
	for i in range(int(testX.shape[0]/batch_size)):
		n = min(batch_size, testX.shape[0]-i*batch_size)
		batch_xs = testX[i*batch_size:i*batch_size+n] 
		batch_ys = np.zeros((batch_size,10))
		[p] = sess.run([prediction], feed_dict={x: batch_xs})
		pred.extend(np.argmax(p,axis=1))

	return pred
