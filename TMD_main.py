import os
import numpy as np
import data_one
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import model
lambda1=0.01
learning_rate=0.00001
do_trans_data = False
BatchSize = 64
EPOCHS = 100
traindata_size=36000
testdata_size=40000-traindata_size
tfrecords_file = 'train.tfrecords'
if do_trans_data:
    data_one.transData()
def lrschedule(i):
    if i<=40:
        return 0.1
    elif i<=80:
        return 0.01
    else:
        return 0.001
def train_nn(istrain=True,create_csv=True):
    x = tf.placeholder("float", shape=[None, 256, 256, 1])
    y = tf.placeholder("float", shape=[None, 100])
    keep_prob = tf.placeholder(tf.float32)
    batch_num = traindata_size/ BatchSize
    dataset = data_one.DataSet()
    prediction = model.network1(x, keep_prob, lambda1)
    mse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    tf.add_to_collection('losses', mse_loss)
    loss = tf.add_n(tf.get_collection('losses'))
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float16"))
    sess = tf.Session()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    if istrain:
        saver = tf.train.Saver(max_to_keep=1)
        history=np.zeros([0,2],'float32')
        last_accuracy=0
        for i in range(EPOCHS):
            print(i, '/', EPOCHS)
            # train_op = tf.train.GradientDescentOptimizer(learning_rate=lrschedule(i)).minimize(loss)
            for batch_step in range(batch_num.__int__()):
                batch_index= np.arange(batch_step*BatchSize, (batch_step+1)*BatchSize, 1, dtype=np.int)
                train_data, train_label = dataset.__getitem__(batch_index)
                current=sess.run([train_op, loss, accuracy], feed_dict={x: train_data, y: train_label, keep_prob: 0.8})

                print('batch index:%d/%d, loss:%f, accuracy:%f'%(batch_step,batch_num.__int__(), current[1],current[2]))
                history=np.append(history,np.array(current[1:]).reshape([1,2]),axis=0)
            # last_accuracy,stoptraining_flag=validation_check(sess, saver, dataset, prediction, loss,
            #                                     accuracy, x, y, keep_prob, last_accuracy)
            # if stoptraining_flag:
            #     break
        saver.save(sess, "./my-model.ckpt", global_step=i)
        np.save('history.npy', history)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)

    if create_csv:
        validation_check(sess, saver, dataset, prediction, loss,accuracy, x, y, keep_prob)
        model_file = tf.train.latest_checkpoint('./')
        saver.restore(sess, model_file)
        Testset=data_one.TestSet(True)
        testBatchSize=100
        predicted_label=np.array([],dtype='float')
        predicted_label.shape=[-1,100]
        batch_num=Testset.__len__()/testBatchSize
        for batch_step in range(batch_num.__int__()):
            batch_index = np.arange(batch_step * testBatchSize, (batch_step + 1) * testBatchSize, 1, dtype=np.int)
            test_data = Testset.__getitem__(batch_index)
            predicted_label=np.append(predicted_label,sess.run(prediction, feed_dict={x: test_data, keep_prob: 1.0}),axis=0)
        Testset.label_data(predicted_label)

def validation_check(sess,saver,dataset,prediction,loss,accuracy,x,y,keep_prob,last_accuracy=0,flag_count=0):
    batch_num = testdata_size / BatchSize
    model_file = tf.train.latest_checkpoint('./')
    saver.restore(sess, model_file)
    val_loss = 0
    val_acc = 0
    predicted_label = []
    for batch_step in range(batch_num.__int__()):
        batch_index = np.arange(traindata_size + batch_step * BatchSize,
                                traindata_size + (batch_step + 1) * BatchSize, 1, dtype=np.int)
        test_data, test_label = dataset.__getitem__(batch_index)
        val_loss, val_acc = np.add([val_loss, val_acc],sess.run([loss, accuracy], feed_dict={x: test_data, y: test_label, keep_prob: 1.0}))
        predicted_label=np.append(sess.run([tf.argmax(prediction)], feed_dict={x: test_data, keep_prob: 1.0}),predicted_label)
    val_loss=val_loss/batch_num
    val_acc=val_acc/batch_num
    print('val_loss:%f, val_acc:%f'%(val_loss,val_acc))
    if val_acc/last_accuracy>=0.99:
        flag_count
    else:
        flag_count=0;
        stoptraining_flag=False
    return val_acc, stoptraining_flag,flag_count
def draw_loss_acc(data_size,history):
    x_trick = [x + 1 for x in range(EPOCHS-1)]
    loss = history[:,0]
    acc = history[:,1]
    batch_num = data_size / BatchSize
    loss_plot=[]
    acc_plot = []
    for i in range((loss.__len__()/batch_num).__int__()):
        index= np.arange(i*batch_num, (i+1)*batch_num-1, 1, dtype=np.int)
        loss_plot=np.append(loss_plot,np.mean(loss[index]))
        acc_plot=np.append(acc_plot,np.mean(acc[index]))
    plt.style.use('ggplot')

    plt.figure(figsize=(10, 6))
    plt.title('model = %s, batch_size = %s' % ('losses', BatchSize))
    plt.plot(x_trick, loss_plot, 'g-', label='loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.savefig( 'loss.png', format='png', dpi=300)
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.title('learninngRate = %f, batch_size = %s' % (learning_rate, BatchSize))
    plt.plot(x_trick, acc_plot, 'b-', label='acc')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('acc')

    plt.savefig('acc.png', format='png', dpi=300)
    plt.show()
# train_nn(True)
history = np.load('history.npy')
draw_loss_acc(traindata_size,history)
train_nn(False,True)

