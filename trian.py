import tensorflow as tf
import model
import input_data
import numpy as np
w=100
h=100
c=3
n_epoch=100
batch_size=64

#模型保存地址
model_path='E:\\tensorflow\\cnn_classifier\\logs\\model.ckpt'

x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x') #tensor
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = model.inference(x,regularizer)

#(小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval')

loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



#保存精度最高的三代
saver=tf.train.Saver(max_to_keep=3)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
x_train,y_train,x_val,y_val = input_data.get_batch()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()  # 开启多线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 这里指代的是读取数据的线程，如果不加的话队列一直挂起

    f = open('logs/acc.txt', 'w')
    max_acc = 0
    for epoch in range(n_epoch):
        #training
        train_loss, train_acc, n1_batch = 0, 0, 0
        # 相当于 for i in range(batch_size) batch_size = 64 次的迭代
        for x_train_a, y_train_a in input_data.minibatches(x_train, y_train, batch_size, shuffle=True):
            _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err; train_acc += ac; n1_batch += 1

        #validation
        val_loss, val_acc, n2_batch = 0, 0, 0
        for x_val_a, y_val_a in input_data.minibatches(x_val, y_val, batch_size, shuffle=False):
            err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err; val_acc += ac; n2_batch += 1
        print("Iter:%d,train acc: %.3f%%, val acc: %.3f%%" % (epoch,
                                    (np.sum(train_acc) / n1_batch) * 100, np.sum(val_acc) / n2_batch * 100))

        f.write(str(epoch*batch_size+n2_batch) + ', val_acc: ' + str(np.sum(val_acc) / n2_batch * 100)+ '\n')

        if val_acc > max_acc:
            max_acc = val_acc
            saver.save(sess,model_path,global_step=epoch*batch_size+n2_batch)
        coord.request_stop()  # 多线程关闭
        coord.join(threads)
sess.close()
