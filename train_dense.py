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


input_size = 784
hidden_size1 = 256
hidden_size2 = 64
output_size = 10
batch_size = 50
n_iter = 12


# Create the model
x = tf.placeholder(tf.float32, [None, input_size])
W1 = tf.Variable(tf.random_normal([input_size, hidden_size1]),name='w1')
b1 = tf.Variable(tf.random_normal([hidden_size1]), name ='b1')
h1 = tf.nn.sigmoid( tf.matmul(x,W1)+b1 )

W2 = tf.Variable(tf.random_normal([hidden_size1, hidden_size2]),name='w2')
b2 = tf.Variable(tf.random_normal([hidden_size2]), name ='b2')
h2 = tf.nn.sigmoid( tf.matmul(h1,W2)+b2 )

W3 = tf.Variable(tf.random_normal([hidden_size2, output_size]),name='w3')
b3 = tf.Variable(tf.random_normal([output_size]), name ='b3')

y = tf.matmul(h2, W3) +b3
y_ = tf.placeholder(tf.float32, [None,output_size])

saver = tf.train.Saver()

def train(trainX, trainY):
	
	onehot = np.zeros((trainY.shape[0], output_size))
	onehot[np.arange(trainY.shape[0]), trainY] = 1
	trainy = onehot

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.AdamOptimizer().minimize(cost)
	saver = tf.train.Saver()

	#sess = tf.Session()
	#sess.run(tf.initialize_all_variables())
	
	# Train
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for current_iter in range(n_iter):
			epoch_loss = 0
			for i in range(int(trainX.shape[0] / batch_size)):
				n = min(batch_size, trainX.shape[0]-i*batch_size)
				batch_xs = trainX[i*batch_size:i*batch_size+n].reshape(n,input_size) 
				batch_ys = trainy[i*batch_size:i*batch_size+n]
			
				_,loss,pred = sess.run([train_step, cost,y], feed_dict={x: batch_xs, y_: batch_ys})
				acc = np.mean(np.argmax(pred,axis=1) == trainY[i*batch_size:i*batch_size+n] )
				epoch_loss+= loss 
				if i%200==0:
					print(i,loss,acc)
		
			print('epoch ' ,current_iter,'loss : ', epoch_loss)
		saver.save(sess, 'dense/model.ckpt')

def test(testX):
	
	sess = tf.Session()
	saver = tf.train.Saver()


	'''
	print("loading a session")
	ckpt = tf.train.get_checkpoint_state('')
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, 'model.ckpt')
	else:
		raise Exception("no checkpoint found")
	'''
	saver.restore(sess, tf.train.latest_checkpoint('dense/'))
	sess.run(tf.all_variables())

	pred = []
	for i in range(int(testX.shape[0] / batch_size)):
		n = min(batch_size, testX.shape[0]-i*batch_size)
		batch_xs = testX[i*batch_size:i*batch_size+n].reshape(n,input_size) 
		batch_ys = np.zeros((batch_size,output_size))
	
		p = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})

		pred.extend(np.argmax(p,axis=1))

	return pred
