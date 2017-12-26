from __future__ import print_function
import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # tf.float32 implicity

# evaluate the nodes. Creates a Session object
sess = tf.Session()

print(node1, node2, '\n')
print(sess.run([node1, node2]), '\n')

# add two nodes

node3 = tf.add(node1, node2)
print('node3: ', node3, '\n')
print('sess.run(node3): ', sess.run(node3), '\n')

# external inputs. Placeholders, promise to provide a value later

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print('Add two nodes. \n')
print(sess.run(adder_node, {a:3, b:4.3}), '\n')
print(sess.run(adder_node, {a:[1, 3], b:[2,5]}), '\n')

# other more complex

add_and_triple = adder_node*3
print('add and triple: ', sess.run(add_and_triple, {a:3, b:5}))

#------------------------------------------------------------------
#    LINEAL MODEL

w = tf.Variable([0.3], dtype = tf.float32)
b = tf.Variable([-0.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)

lineal = w*x + b

# initialize variables

init = tf.global_variables_initializer()
sess.run(init)

print('\nLinear model: ',sess.run(lineal, {x: [1, 2, 3, 4]}))

# loss function
y = tf.placeholder(tf.float32)
squares = tf.square(lineal - y)
loss = tf.reduce_sum(squares)

print('\nloss function: ',sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# we can change the value of a already inicialized variable with the operator tf.assign()

w = tf.assign(w, [-1.0])
new_b = tf.assign(b, [1.0])

sess.run([w, new_b])

print('\n=========================================')
print("\nlet's see that they are different objects.\n")
print('new_b: ',b,' b: ',new_b)
print('\nbut both have the same value.\n')
print('new_b: ',sess.run(b),' b: ',sess.run(new_b))
print('=========================================')
print('\nloss function 2: ',sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


