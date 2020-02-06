# CarND-LeNet-Lab
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Discussion
In the original example, Test Accuracy = 0.989. I'm trying to change some parameters.

* To make the model have less cost and more accuracy, i change its weight to see if the result improves. When I change the sigma value，test accuracy =0.988. Unfortunately, the result didn't get any better. The second time the value of sigma was changed to 0.05, and the Test Accuracy was 0.988, which was also not improved.

* padding: ‘SAME’ or ‘VALID’
 from tensorflow.contrib.layers import flatten

 
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

 
    conv1 = tf.nn.relu(conv1)

   
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

   
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
    
  
    conv2 = tf.nn.relu(conv2)

  
 
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  
    fc0   = flatten(conv2)
   
   
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1024,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0,fc1_W) + fc1_b
    
   
    fc1    = tf.nn.relu(fc1)
  

 
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1,fc2_W) + fc2_b
    
 
    fc2    = tf.nn.relu(fc2)


  
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2,fc3_W) + fc3_b
    
    return logits
 
 
 
 
 
 
 
 * max or average pooling
