# CarND-LeNet-Lab
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Discussion
In the original example, Test Accuracy = 0.989. I'm trying to change some parameters.

* ` To make the model have less cost and more accuracy, i change its weight to see if the result improves.` When I change the sigma value，  test accuracy =0.988. Unfortunately, the result didn't get any better. The second time the value of sigma was changed to 0.05, and the Test Accuracy was 0.988, which was also not improved.

* `padding: ‘SAME’ or ‘VALID’`
 When padding='SAME', the result becomes 0.988.
 
 * `max or average pooling`
  When "conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')", the result becomes 0.988.

To sum up, the accuracy has not changed much. I think there may be other ways to improve accuracy.
