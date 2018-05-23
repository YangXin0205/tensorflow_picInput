import tensorflow as tf
import input_data
import numpy as np
import model
w = 64
h = 64
c = 3
n_epoch = 100
batch_size = 32

# 模型保存地址
model_path=r'D:\PycharmProjects\Test\faceRecognition\model_save\model.ckpt'

x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')  # tensor
y_ = tf.placeholder(tf.int32, shape=[None,], name='y_')
keep_prob = tf.placeholder(tf.float32, name='kp')

regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = model.inference(x, regularizer, keep_prob)

# (小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1, dtype=tf.float32)
logits_eval = tf.multiply(logits, b, name='logits_eval')

cross_entropy = model.loss(logits, y_)
train_op = model.training(cross_entropy, 0.001)
acc = model.accuracy(logits, y_)


merged = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=1)

x_train, y_train, x_val, y_val = input_data.get_batch()
iter_num = int(len(x_train)/batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()  # 开启多线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 这里指代的是读取数据的线程，如果不加的话队列一直挂起

    train_writer = tf.summary.FileWriter('log' + '/train', sess.graph)
    test_writer = tf.summary.FileWriter('log' + '/val')

    max_acc = 0
    for epoch in range(n_epoch):
        # training
        train_loss, train_acc, n1_batch = 0, 0, 0
        # 相当于 for i in range(batch_size) batch_size = 64 次的迭代
        for x_train_a, y_train_a in input_data.minibatches(x_train, y_train, batch_size, shuffle=True):
            summary,_, err, ac = sess.run([merged,train_op, cross_entropy, acc], feed_dict={x: x_train_a, y_: y_train_a, keep_prob: 0.6})
            train_loss += err
            train_acc += ac
            n1_batch += 1
            print(epoch*iter_num + n1_batch)
            train_writer.add_summary(summary, epoch*iter_num + n1_batch)

        # validation
        val_loss, val_acc, n2_batch = 0, 0, 0
        for x_val_a, y_val_a in input_data.minibatches(x_val, y_val, batch_size, shuffle=False):
            summary,err, ac = sess.run([merged,cross_entropy, acc], feed_dict={x: x_val_a, y_: y_val_a, keep_prob: 1})
            val_loss += err
            val_acc += ac
            n2_batch += 1

            test_writer.add_summary(summary, epoch*iter_num + n2_batch)

        print("Iter:%d,train acc: %.3f%%, val acc: %.3f%%" % (epoch,
                                                              (np.sum(train_acc) / n1_batch) * 100,
                                                              np.sum(val_acc) / n2_batch * 100))

        if val_acc > max_acc:
            max_acc = val_acc
            with open('log\\acc.txt', 'w') as f:
                f.write('iter:' + str(epoch * batch_size + n2_batch) + ', val_acc: ' + str(
                np.sum(val_acc) / n2_batch * 100) + '%' + '\n')
            saver.save(sess, model_path, global_step=epoch * batch_size + n2_batch)
    coord.request_stop()  # 多线程关闭
    coord.join(threads)
