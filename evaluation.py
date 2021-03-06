from skimage import io,transform
import tensorflow as tf
import numpy as np
import os
import glob

pic_size = 64

'''由训练集目录生成分类dict'''
flower_dict = {}
trian_pic = r"D:\PycharmProjects\retrain\test"
for idx,dir in enumerate(os.listdir(trian_pic)):
    flower_dict[idx] = dir


def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(pic_size,pic_size))
    return np.asarray(img)

with tf.Session() as sess:
    data = []
    pic_name = []
    test_pic = r"D:\PycharmProjects\retrain\test1"
    for i in os.listdir(test_pic):
        if os.path.isfile(os.path.join(test_pic,i)):
            pic_name.append(i)
            data.append(read_one_image(os.path.join(test_pic,i)))

    saver = tf.train.import_meta_graph(glob.glob('D:\\PycharmProjects\\Test\\faceRecognition\\model_save\\model.ckpt-*.meta')[0])
    saver.restore(sess,tf.train.latest_checkpoint(r'D:\PycharmProjects\Test\faceRecognition\model_save'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    kp= graph.get_tensor_by_name("kp:0")

    feed_dict = {x:data,kp:1.0}

    logits = graph.get_tensor_by_name("logits_eval:0")


    #classification_result = sess.run(logits,feed_dict)
    classification_result = sess.run(tf.nn.softmax(logits), feed_dict)
    #打印出概率预测矩阵
    print(classification_result)
    #打印出预测矩阵每一行最大值的索引
    #print(tf.argmax(classification_result,1).eval())
    #根据索引通过字典对应花的分类
    output = tf.argmax(classification_result,1).eval()


    for i in range(len(output)):
        print("pic",i+1,"class: "+flower_dict[output[i]])
        classes = flower_dict[output[i]]
        #测试图片abs路径
        path1 = os.path.join(test_pic,classes)

        if not os.path.exists(path1):
            os.mkdir(path1)
            os.renames(os.path.join(test_pic, pic_name[i]), os.path.join(path1, pic_name[i]))
        else:
            os.renames(os.path.join(test_pic, pic_name[i]), os.path.join(path1, pic_name[i]))
