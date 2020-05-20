import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average
import numpy as np
import os, pdb
import cv2
import numpy as np
import random as rn
import tensorflow as tf
import threading
import time
from sklearn import metrics
import utils
global n_classes


n_classes = 2


def activation(x,name="activation"):
    return tf.nn.swish(x)
    
def conv2d(name, l_input, w, b, s, p):
    l_input = tf.nn.conv2d(l_input, w, strides=[1,s,s,1], padding=p, name=name)
    l_input = l_input+b

    return l_input

def batchnorm(conv, isTraining, name='bn'):
    return tf.layers.batch_normalization(conv, training=isTraining, name="bn"+name)

def initializer(in_filters, out_filters, name):
    w1 = tf.get_variable(name+"W", [3, 3, in_filters, out_filters], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable(name+"B", [out_filters], initializer=tf.truncated_normal_initializer())
    return w1, b1
  
def residual_block(in_x, in_filters, out_filters, stride, isDownSampled, name, isTraining):
    global ema_gp
    # first convolution layer
    if isDownSampled:
      in_x = tf.nn.avg_pool(in_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
      
    x = batchnorm(in_x, isTraining, name=name+'FirstBn')
    x = activation(x)
    w1, b1 = initializer(in_filters, in_filters, name+"first_res")
    x = conv2d(name+'r1', x, w1, b1, 1, "SAME")

    # second convolution layer
    x = batchnorm(x, isTraining, name=name+'SecondBn')
    x = activation(x)
    w2, b2 = initializer(in_filters, out_filters, name+"Second_res")
    x = conv2d(name+'r2', x, w2, b2, 1, "SAME")
    
    if in_filters != out_filters:
        difference = out_filters - in_filters
        left_pad = difference // 2
        right_pad = difference - left_pad
        identity = tf.pad(in_x, [[0, 0], [0, 0], [0, 0], [left_pad, right_pad]])
        return x + identity
    else:
        return in_x + x

      
def ResNet(_X, isTraining):
    global n_classes
    w1 = tf.get_variable("initW", [7, 7, 3, 96], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable("initB", [96], initializer=tf.truncated_normal_initializer())
    x = conv2d('conv1', _X, w1, b1, 4, "VALID")
    
    filters_num = [96,128,256]
    block_num = [2,2,2]
    l_cnt = 1
    for i in range(len(filters_num)):
      for j in range(block_num[i]):
          
          if ((j==block_num[i]-1) & (i<len(filters_num)-1)):
            x = residual_block(x, filters_num[i], filters_num[i+1], 2, True, 'ResidualBlock%d'%(l_cnt), isTraining)
            print('[L-%d] Build %dth connection layer %d from %d to %d channels' % (l_cnt, i, j, filters_num[i], filters_num[i+1]))
          else:
            x = residual_block(x, filters_num[i], filters_num[i], 1, False, 'ResidualBlock%d'%(l_cnt), isTraining)
            print('[L-%d] Build %dth residual block %d with %d channels' % (l_cnt,i, j, filters_num[i]))
          l_cnt +=1

    saliency = x
    x_shape = x.get_shape().as_list()
    dense1 = x_shape[1]*x_shape[2]*x_shape[3]
    W = tf.get_variable("featW", [dense1, 128], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable("featB", [128], initializer=tf.truncated_normal_initializer())
    dense1 = tf.reshape(x, [-1, dense1])
    feat = tf.nn.softmax(tf.matmul(dense1, W) + b)
    
    with tf.variable_scope('Final'):
        x = batchnorm(x, isTraining, name='FinalBn')
        x = activation(x)
        wo, bo=initializer(filters_num[-1], n_classes, "FinalOutput")
        x = conv2d('final', x, wo, bo, 1, "SAME")


        x=tf.reduce_mean(x, [1, 2])

        W = tf.get_variable("FinalW", [n_classes, n_classes], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable("FinalB", [n_classes], initializer=tf.truncated_normal_initializer())

        out = tf.matmul(x, W) + b
                            

    return out, feat, saliency


#==========================================================================
#=============Reading data in multithreading manner========================
#==========================================================================
def read_labeled_image_list(image_list_file, training_img_dir):
    f = open(image_list_file, 'r')
    filenames = []
    labels = []

    for line in f:
        filename, label = line[:-1].split(' ')
        filename = training_img_dir+filename
        filenames.append(filename)
        labels.append(int(label))
        
    return filenames, labels
    
    
def read_images_from_disk(input_queue, size1=64):
    label = input_queue[1]
    fn=input_queue[0]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    
    #example = tf.image.decode_png(file_contents, channels=3, name="dataset_image") # png fo rlfw
    example=tf.image.resize_images(example, [size1,size1])
    return example, label, fn
    
def setup_inputs(sess, filenames, training_img_dir, image_size=64, crop_size=64, isTest=False, batch_size=128):
    
    # Read each image file
    image_list, label_list = read_labeled_image_list(filenames, training_img_dir)

    images = tf.cast(image_list, tf.string)
    labels = tf.cast(label_list, tf.int64)
     # Makes an input queue
    if isTest is False:
        isShuffle = True
        numThr = 4
    else:
        isShuffle = False
        numThr = 1
        
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=isShuffle)
    image, y,fn = read_images_from_disk(input_queue)

    channels = 3
    image.set_shape([None, None, channels])
        
    # Crop and other random augmentations
    if isTest is False:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, .95, 1.05)
        image = tf.image.random_brightness(image, .05)
        image = tf.image.random_contrast(image, .95, 1.05)
        
    image = tf.cast(image, tf.float32)/255.0
    
    image, y,fn = tf.train.batch([image, y, fn], batch_size=batch_size, capacity=batch_size*3, num_threads=numThr, name='labels_and_images')

    tf.train.start_queue_runners(sess=sess)

    return image, y, fn, len(label_list)

#==========================================================================
#=============Reading data in multithreading manner========================
#==========================================================================
def read_labeled_image_list2(image_list_file, training_img_dir):
    f = open(image_list_file, 'r')
    filenames = []
    filenames2 = []
    labels = []

    for line in f:
        filename, fn2, label = line[:-1].split(' ')
        filename = training_img_dir+filename
        filenames.append(filename)
        filename = training_img_dir+fn2
        filenames2.append(filename)
        labels.append(int(label))
        
    return filenames, filenames2, labels
    
    
def read_images_from_disk2(input_queue, size1=64):
    label = input_queue[2]
    fn=input_queue[0]
    file_contents = tf.read_file(input_queue[0])
    file_contents2 = tf.read_file(input_queue[1])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    example2 = tf.image.decode_jpeg(file_contents2, channels=3)
    
    #example = tf.image.decode_png(file_contents, channels=3, name="dataset_image") # png fo rlfw
    example=tf.image.resize_images(example, [size1,size1])
    example2=tf.image.resize_images(example2, [size1,size1])
    return example, example2, label, fn
    
def setup_inputs2(sess, filenames, training_img_dir, image_size=64, crop_size=64, isTest=False, batch_size=128):
    
    # Read each image file
    image_list, image2_list, label_list = read_labeled_image_list2(filenames, training_img_dir)

    images = tf.cast(image_list, tf.string)
    images2 = tf.cast(image2_list, tf.string)
    labels = tf.cast(label_list, tf.int64)
     # Makes an input queue
    if isTest is False:
        isShuffle = True
        numThr = 4
    else:
        isShuffle = False
        numThr = 1
        
    input_queue = tf.train.slice_input_producer([images, images2, labels], shuffle=isShuffle)
    image, image2, y,fn = read_images_from_disk2(input_queue)

    channels = 3
    image.set_shape([None, None, channels])
    image2.set_shape([None, None, channels])
        
    # Crop and other random augmentations
    if isTest is False:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, .95, 1.05)
        image = tf.image.random_brightness(image, .05)
        image = tf.image.random_contrast(image, .95, 1.05)
        image2 = tf.image.random_flip_left_right(image2)
        image2 = tf.image.random_saturation(image2, .95, 1.05)
        image2 = tf.image.random_brightness(image2, .05)
        image2 = tf.image.random_contrast(image2, .95, 1.05)
        

    image = tf.cast(image, tf.float32)/255.0
    image2 = tf.cast(image2, tf.float32)/255.0
    
    image, image2, y,fn = tf.train.batch([image, image2, y, fn], batch_size=batch_size, capacity=batch_size*3, num_threads=numThr, name='labels_and_images')

    tf.train.start_queue_runners(sess=sess)

    return image, image2, y, fn, len(label_list)
'''
Main Program:
Fake image detection based on siamese network (with constrastive loss)
'''
batch_size = 32
display_step = 80
learning_rate = tf.placeholder(tf.float32)      # Learning rate to be fed
lr = 1e-3                         # Learning rate start
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
print('GO!!')


# In[ ]:


# Setup the tensorflow...
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

print("Preparing the training & validation data...")
# pairwise_progressGAN.txt ==> PGGAN is excluded in the training set
# pairwise_wgan.txt        ==> WGAN is excluded in the training set

train_file = "/home/ubuntu/prep_data/cffn_classifier_train.txt"
val_file = "/home/ubuntu/prep_data/cffn_classifier_val.txt"
pairs_file = "/home/ubuntu/prep_data/cffn_pairs.txt"
train_img_dir = "/home/ubuntu/prep_data/"

sia_data, sia_data2, sia_labels, sialist1, slen1 = setup_inputs2(sess, pairs_file, train_img_dir, batch_size=batch_size)
# For Level-2 learning : Classifier learning
train_data, train_labels, filelist1, glen1 = setup_inputs(sess, train_file, train_img_dir, batch_size=batch_size)
# For Level-2 test: valprogressGAN.txt means that the PGGAN is used to verify the system performance.
val_data, val_labels, filelist2, tlen1 = setup_inputs(sess, val_file, train_img_dir, batch_size=10, isTest=True)
print("Found %d pairs, %d training images, and %d validation images..." % (slen1, glen1, tlen1))

max_iter = glen1*4  # How many epochs we want to used to train...
print("Preparing the training model with learning rate = %.5f..." % (lr))

# Make a model
with tf.variable_scope("ResNet") as scope:
    #=================Level-1 training based on constrative learning===================
    _, feat1,_ = ResNet(sia_data, True)
    scope.reuse_variables()
    _, feat2,_ = ResNet(sia_data2, True)
    scope.reuse_variables()
    #==================================================================================
    #=================Level-2 training based on corss-entropy==========================
    pred, _,_ = ResNet(train_data, True)
    scope.reuse_variables()
    valpred, _, saliency = ResNet(val_data, False)
    #==================================================================================


#==================Set up the constrative loss===============================
with tf.name_scope('ContrastiveLoss'):
  margin = 0.5
  labels_t = tf.cast(train_labels, tf.float32)
  labels_f = tf.cast(1-train_labels, tf.float32)         # labels_ = !labels;
  eucd2 = tf.pow(feat1- feat2, 2.0)
  eucd2 = tf.reduce_sum(eucd2, [1])
  eucd = tf.sqrt(eucd2+1e-10, name="eucd")
  C = tf.constant(margin, name="C")
  pos = labels_t * eucd2
  neg = labels_f *tf.pow(tf.maximum(C- eucd, 0), 2)
  losses = pos + neg
  sialoss = tf.reduce_mean(losses, name="Contrastive_loss")
#=============================================================================
#==================Set up the optimizer=======================================
with tf.name_scope('Loss_and_Accuracy'):
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    t_vars=tf.trainable_variables() 
    #t_vars=[var for var in t_vars if 'Final']
    cost = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=t_vars)
    sia_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(sialoss)
  
  #================Set up the accuracy measurement============================
  correct_prediction = tf.equal(tf.argmax(pred, 1), train_labels)   # Training acc.
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  correct_prediction2 = tf.equal(tf.argmax(valpred, 1), val_labels) # Validation acc
  accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))


#==========Summary============================
tf.summary.scalar("Contrastive_loss", sialoss)
tf.summary.scalar('Loss', cost)
tf.summary.scalar('Training_Accuracy', accuracy)
tf.summary.scalar('Validation_Accuracy', accuracy2)

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)
step = 0
writer = tf.summary.FileWriter("logs/sia/", sess.graph)
summaries = tf.summary.merge_all()

print("We are going to pretrain model susing ResNet based on contrastive loss!!!")
start_lr = lr
while (step * batch_size) < max_iter:
    epoch1=np.floor((step*batch_size)/glen1)
    # Learning rate decay
    if (((step*batch_size)%glen1 < batch_size) & (lr==1e-3) & (epoch1 >2)):
        lr /= 10
    
    if epoch1 <=1:
        sess.run([sia_optimizer],  feed_dict={learning_rate: lr})
    else:
        # Learning rate decay at level-2 training
#         if start_lr == lr:
#             lr = lr /10
        sess.run([optimizer],  feed_dict={learning_rate: lr})
        
    if (step % 15000==1) & (step>15000):
        save_path = saver.save(sess, "checkpoints/tf_deepUD_model_iter" + str(step) + ".ckpt")
        print("Model saved in file at iteration %d: %s" % (step*batch_size,save_path))

    if step>0 and step % display_step == 0:
        # calculate the loss
        loss, acc, summaries_string, sia_val = sess.run([cost, accuracy, summaries, sialoss])
        print("Iter=%d/epoch=%d, Loss=%.6f, Contrastive loss=%.6f, Training Accuracy=%.6f, lr=%f" % (step*batch_size, epoch1 ,loss, sia_val, acc, lr))
        writer.add_summary(summaries_string, step)
    
    if step>0 and (step % (display_step*20) == 0):
        rounds = tlen1 // 1000
        valacc=[]
        vis=[]
        tis=[]
        for k in range(rounds):
          a2, vi, ti = sess.run([accuracy2, tf.argmax(valpred, 1), val_labels])
          valacc.append(a2)
          vis.append(vi)
          tis.append(ti)
        tis = np.reshape(np.asarray(tis), [-1])
        vis = np.reshape(np.asarray(vis), [-1])
        precision=metrics.precision_score(tis, vis) 
        recall=metrics.recall_score(tis, vis)
        
        sal, valimg = sess.run([saliency, val_data])
        #utils.batchimwrite2(sal, 'saliency_img/sal')
        #utils.batchimwrite2(valimg, 'saliency_img/img')

        print("Iter=%d/epoch=%d, Validation Accuracy=%.6f, Precision=%.6f, Recall=%.6f" % (step*batch_size, epoch1 , np.mean(valacc), precision, recall))

  
    step += 1
print("Optimization Finished!")
save_path = saver.save(sess, "checkpoints/tf_deepUD_model.ckpt")
print("Model saved in file: %s" % save_path)
