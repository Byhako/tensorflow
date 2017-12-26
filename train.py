import tensorflow as tf

sess = tf.Session()

w = tf.Variable([0.3], dtype = tf.float32)
b = tf.Variable([-0.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)

lineal = w*x + b

init = tf.global_variables_initializer()
sess.run(init)

y = tf.placeholder(tf.float32)
squares = tf.square(lineal - y)
loss = tf.reduce_sum(squares)


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
	sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([w, b]))

