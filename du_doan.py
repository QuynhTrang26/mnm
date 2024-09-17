import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import pandas as pd
from numpy import array
tf.disable_v2_behavior()
# Generating random linear data
# There will be 50 data points ranging from 0 to 50
df=pd.read_csv('Student_Performance.csv')#,index_col=0,header = 0)
x = array(df.iloc[:90,0:5]).astype(np.float64)
y = array(df.iloc[:90,4:5]).astype(np.float64)
#y = list(y)
n =10
print(x)
'''
#Tạo tâp giá trị x và y
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)
 
# Cộng thêm nhiễu cho tập x và y để có tập dữ liệu ngẫu nhiên
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)
n = len(x) # Số lượng dữ liệu
# Plot of Training Data
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")
plt.show()'''
# Tạo model cho tập dữ liệu
X = tf.placeholder("float64",shape = [5,1])
Y = tf.placeholder("float64",shape = 1)

# khởi tạo biến w và b
W = tf.Variable(np.random.randn(5,1), name = "W")
b = tf.Variable(np.random.randn(1), name = "b" )
# thiết lập tốc độ học
learning_rate = 0.01
# số vòng lặp
training_epochs = 100
# Hàm tuyến tính y = w*x +b
#y_pred = tf.add(tf.multiply(X, W), b)
y1 = []
for i in range(0,5):
    y_pred1 = tf.multiply(X[i],W[i])
    y1.append(y_pred1)
tam = tf.add_n(y1)
print(tam)
y_pred = tf.add(tam,b)
print("aaaaaaaaaaaaaaaaaaaa")
# Mean Squared Error Cost Function
cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n)
 
# Tối ưu bằng Gradient Descent 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
 
# Thiết lập Global Variables 
init = tf.global_variables_initializer()
# Starting the Tensorflow Session
with tf.Session() as sess:
     
    # Initializing the Variables
    sess.run(init)
     
    # Iterating through all the epochs
    for epoch in range(training_epochs):
         
        # Feeding each data point into the optimizer using Feed Dictionary
        for (_x, _y) in zip(x, y):
           _x = np.reshape(_x,(5,1))
           sess.run(optimizer, feed_dict = {X : _x, Y : _y})
         
        # Displaying the result after every 50 epochs
        if (epoch + 1) % 10 == 0:
            # Calculating the cost a every epoch
            c = sess.run(cost, feed_dict = {X : x, Y : y})
            print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b))
     
    # Storing necessary values to be used outside the Session
    training_cost = sess.run(cost, feed_dict ={X: x, Y: y})
    weight = sess.run(W)
    bias = sess.run(b)
# Calculating the predictions
predictions = weight * x + bias
print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n')
# Plotting the Results
plt.plot(x, y, 'ro', label ='Original data')
plt.plot(x, predictions, label ='Fitted line')
plt.title('Linear Regression Result')
plt.legend()
plt.show()
