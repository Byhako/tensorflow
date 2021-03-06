import tensorflow as tf

# Model parameters
w = tf.Variable([0.3], dtype = tf.float32)
b = tf.Variable([-0.3], dtype = tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
lineal = w*x + b
y = tf.placeholder(tf.float32)

# loss
squares = tf.square(lineal - y)
loss = tf.reduce_sum(squares)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# taining loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
	sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_w, curr_b, curr_loss = sess.run([w, b, loss], {x: x_train, y: y_train})

print('\nw: %s, b: %s, loss: %s'%(curr_w, curr_b, curr_loss))
