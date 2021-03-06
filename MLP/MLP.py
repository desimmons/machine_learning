import numpy as np
import scipy.io
import os
import pickle
import tensorflow as tf
import matplotlib.pylab as plt
import pylab
import random
from get_data import *
"""This script uses tensorflow to build a multilayer perceptron.
It then predicts whether a signal transmission is between two devices
on the same body/user or two devices located on two seperate bodies
/users. The signals are scraped from the folder specified in get_data.
"""

def one_hot(Y):
	import numpy as np
	one_hot = []
	for i in range(0,len(Y)):
		if Y[i] == 1: #on body
			one_hot.append([1,0])
		else:	#off body
			one_hot.append([0,1])
	return np.array(one_hot)

#initialize weight matrices
def init_weights(shape, vname):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(weights, name=vname)

#initialize bias terms
def init_bias(shape, vname):
    """ Weight initialization """
    weights = tf.random_normal([shape], stddev=0.01)

    return tf.Variable(weights, name=vname)

#define the forward propagation model
def forwardprop(X, w1, w2, w3, b1, b2, b3, p_keep_input, p_keep_hidden):
	"""
	Forward-propagation.
	IMPORTANT: outlayer is not softmax since TensorFlow's 
	softmax_cross_entropy_with_logits() does that internally.
	"""
	with tf.name_scope("Layer_one"):
		X = tf.nn.dropout(X, p_keep_input)
		h1 = tf.nn.relu(tf.add(tf.matmul(X, w1), b1))  # The \sigma function	
	with tf.name_scope("Layer_two"): 
	    h1 = tf.nn.dropout(h1, p_keep_hidden)
	    h2 = tf.nn.relu(tf.add(tf.matmul(h1, w2), b2))
	with tf.name_scope("Layer_three"):  
		h2 = tf.nn.dropout(h2, p_keep_hidden)   
		out_layer = tf.matmul(h2, w3)+b3
	    # yhat = tf.nn.softmax(out_layer)  # The \varphi function
	return out_layer

stats = None
fft = None
order = None
X,Y = scraper() #scraper scrapes data from specified folders
while stats != "y" and stats != "n":
	stats=raw_input("Would you like to use statistical features (var, skew, percentiles, etc) (y/n): ")
	if stats != "y" and stats != "n":
		print "Obey the instructions!\n"
while fft != "y" and fft != "n":
	fft=raw_input("Would you like to use frequency sample features (y/n): ")
	if stats != "y" and stats != "n":
		print "Obey the instructions!\n"
if fft == "y":
	while type(order) != int:
		order=int(raw_input("Enter an integer FFT order (e.g. 100): "))
		if type(order) != int:
			print "Obey the instructions!\n"
	
X = feature(X,stats,fft,order) 
Y = one_hot(Y)


shuffle_range = np.random.permutation(len(X)) #shuffle the data
percentage = 0.66 #percentage of data used for training
X_train, Y_train = X[shuffle_range[:int(np.floor(percentage*len(X)))]], Y[shuffle_range[:int(np.floor(percentage*len(X)))]]
X_test, Y_test = X[shuffle_range[int(np.ceil(percentage*len(X))):]], Y[shuffle_range[int(np.ceil(percentage*len(X))):]]

#non tf variables
hsize1 = 100 #size of first hidden layer
hsize2 = 35	#size of second hidden layer
learning_rate = 0.001 #learning rate for Adam optimizer
#tf variables
W1 = init_weights((X.shape[1], hsize1), "W1") 
b1 = init_bias((hsize1), "b1")
W2 = init_weights((hsize1, hsize2), "W2") 
b2 = init_bias((hsize2), "b2")
W3 = init_weights((hsize2, Y.shape[1]), "W3") 
b3 = init_bias((Y.shape[1]), "b3")

#tf placeholders
x = tf.placeholder(tf.float32, [None, X.shape[1]], name="X")
y_ = tf.placeholder(tf.float32, [None,  Y.shape[1]], name = "labels")
p_keep_hidden = tf.placeholder(tf.float32)
p_keep_input =  tf.placeholder(tf.float32)

#setup neural network output
y = forwardprop(x, W1, W2, W3, b1, b2, b3, p_keep_input, p_keep_hidden)
#evaluate/optimize output performance
with tf.name_scope("xent"):
	""""
	IMPORTANT: forwardprop() does not apply softmax because softmax is 
	applied in tf.nn.softmax_cross_entropy_with_logits()
	"""
	xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))
with tf.name_scope("train"):
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(xent)
with tf.name_scope("Accuracy"):
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#setup tensorboard summaries	
tf.summary.histogram("weights 1",W1)
tf.summary.histogram("biases 1",b1)
tf.summary.histogram("weights 2",W2)
tf.summary.histogram("biases 2",b2)
tf.summary.scalar("cross_entropy",xent)
tf.summary.scalar("Accuracy",accuracy) 
writer = tf.summary.FileWriter("../tens_bor_dump/MLP")
merged_summary = tf.summary.merge_all()

sess = tf.InteractiveSession()
writer.add_graph(sess.graph)
tf.global_variables_initializer().run()

#setup batch based SGD
batch_size = 200
for i in range(10000):
	batch = random.sample(range(len(X_train)),batch_size)
	batch_xs, batch_ys = X_train[batch], Y_train[batch]
	if i%100 == 0:
		s = sess.run(merged_summary,feed_dict={x: X_test, y_: Y_test, p_keep_input: 1, p_keep_hidden:1})
		writer.add_summary(s,i)
	sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys, p_keep_input: 0.98, p_keep_hidden:0.98})

prediction = tf.argmax(y,1)
print(sess.run(prediction, feed_dict={x: X_test, y_: Y_test, p_keep_input: 1, p_keep_hidden:1}))
print "Test data accuracy: " , sess.run(accuracy, feed_dict={x: X_test, y_: Y_test, p_keep_input: 1, p_keep_hidden:1})