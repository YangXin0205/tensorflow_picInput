import tensorflow as tf

# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图

def inference(input_tensor,regularizer,keep_pro):
    with tf.variable_scope('layer1-conv1'):
        with tf.name_scope('weights'):
            conv1_weights = tf.get_variable("weight",[5,5,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
            variable_summaries(conv1_weights)
        with tf.name_scope('biases'):
            conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
            variable_summaries(conv1_biases)
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight",[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        variable_summaries(conv2_weights)
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        variable_summaries(conv2_biases)
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.variable_scope("layer6-conv4"):
        conv4_weights = tf.get_variable("weight",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [256], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.variable_scope("layer7-conv5"):
        conv5_weights = tf.get_variable("weight",[3,3,256,512],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(relu4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))
        drop5 = tf.nn.dropout(relu5, keep_pro)

    with tf.name_scope("layer8-pool3"):
        pool4 = tf.nn.max_pool(drop5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = 8*8*512
        reshaped = tf.reshape(pool4,[-1,nodes])

    with tf.variable_scope('layer10-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        fc1 = tf.nn.dropout(fc1, keep_pro)

    with tf.variable_scope('layer11-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        fc2 = tf.nn.dropout(fc2, keep_pro)

    with tf.variable_scope('layer12-fc3'):
        fc3_weights = tf.get_variable("weight", [1024, 2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit



def loss(logits,label):
    with tf.name_scope('cross_entropy'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(loss)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    return cross_entropy_mean

def training(loss,lr):
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    return train_op

def accuracy(logits,label):
    with tf.name_scope('accuracy'):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), label)
        with tf.name_scope("accuracy"):
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', acc)
    return acc
