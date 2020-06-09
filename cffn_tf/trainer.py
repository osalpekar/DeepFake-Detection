import tensorflow as tf
import numpy as np
import os, pdb
import cv2
import numpy as np
import random as rn
import threading
import time
from sklearn import metrics
import utils
global n_classes
import triplet_loss as tri
import os.path
from data_reader import setup_inputs
import time
import seaborn as sns
import matplotlib.pyplot as plt
from metrics import compute_precision, compute_recall, compute_f1


'''
#===========================================================================
Parameters:
        n_classes: Number of classes (2 for now. one for fake and one for real)
        data_dir: The path to the file list directory
        image_dir: The path to the images directory (if the image list is stored in absoluate path, set this to './')
        margin: Marginal value in triplet loss function
        learning_rate: starting learning rate used by optimizers
        num_cffn_epochs: number of epochs trained by the siamese network
        num_classifier_epochs: number of epochs training by classification network
        regularization lambda: lambda constant used in regularization term in loss calculation

Data Preparation:
        Data should be in /home/ubuntu/prep_data/cffn_classification_{train/val}.txt
        formatted as follows with 1 being real and 0 fake:
        image_path1 0
        image_path2 1
        image_path3 1
        image_path4 0
#===========================================================================
'''

# Some Basic Setup And Hyperparameters
n_classes = 2
batch_size = 128
print_interval = 80
validation_interval = print_interval * 2
learning_rate = tf.placeholder(tf.float32)
lr = 1e-4     
regularization_lambda = 0.001
margin = 0.8
num_cffn_epochs = 5
num_classifier_epochs = 15
total_epochs = num_cffn_epochs + num_classifier_epochs
# threshold for early stopping
val_threshold = 0.94
image_dir = "/home/ubuntu/big_data/"
data_file = "/home/ubuntu/big_data/labels_new.json"

#========================Mode basic components============================
def activation(x,name="activation"):
    return tf.nn.swish(x)
    
def conv2d(name, l_input, w, b, s, p):
    l_input = tf.nn.conv2d(l_input, w, strides=[1,s,s,1], padding=p, name=name)
    l_input = l_input+b

    return l_input

def batchnorm(conv, isTraining, name='bn'):
    return tf.layers.batch_normalization(conv, training=isTraining, name="bn"+name)

def initializer(in_filters, out_filters, name, k_size=3):
    w1 = tf.get_variable(name+"W", [k_size, k_size, in_filters, out_filters], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable(name+"B", [out_filters], initializer=tf.truncated_normal_initializer())
    return w1, b1
  
def residual_block(in_x, in_filters, out_filters, stride, isDownSampled, name, isTraining, k_size=3):
    global ema_gp
    # first convolution layer
    if isDownSampled:
      in_x = tf.nn.avg_pool(in_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
      
    x = batchnorm(in_x, isTraining, name=name+'FirstBn')
    x = activation(x)
    w1, b1 = initializer(in_filters, in_filters, name+"first_res", k_size=k_size)
    x = conv2d(name+'r1', x, w1, b1, 1, "SAME")

    # second convolution layer
    x = batchnorm(x, isTraining, name=name+'SecondBn')
    x = activation(x)
    w2, b2 = initializer(in_filters, out_filters, name+"Second_res",k_size=k_size)
    x = conv2d(name+'r2', x, w2, b2, 1, "SAME")
    
    if in_filters != out_filters:
        difference = out_filters - in_filters
        left_pad = difference // 2
        right_pad = difference - left_pad
        identity = tf.pad(in_x, [[0, 0], [0, 0], [0, 0], [left_pad, right_pad]])
        return x + identity
    else:
        return in_x + x


'''
#===========================================================================
Network architecture based on ResNet
#===========================================================================
'''      
def ResNet(_X, isTraining):
    global n_classes
    w1 = tf.get_variable("initWeight", [7, 7, 3, 64], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable("initBias", [64], initializer=tf.truncated_normal_initializer())
    initx = conv2d('conv1', _X, w1, b1, 4, "VALID")
    
    filters_num = [64,96,128, 256]
    block_num = [2,4,3, 6]
    l_cnt = 1
    x = initx
    
    # ============Feature extraction network with kernel size 3x3============
    
    for i in range(len(filters_num)):
        for j in range(block_num[i]):
          
            if ((j==block_num[i]-1) & (i<len(filters_num)-1)):
                x = residual_block(x, filters_num[i], filters_num[i+1], 2, True, 'ResidualBlock%d'%(l_cnt), isTraining)
                print('[L-%d] Build %dth connection layer %d from %d to %d channels' % (l_cnt, i, j, filters_num[i], filters_num[i+1]))
            else:
                x = residual_block(x, filters_num[i], filters_num[i], 1, False, 'ResidualBlock%d'%(l_cnt), isTraining)
                print('[L-%d] Build %dth residual block %d with %d channels' % (l_cnt,i, j, filters_num[i]))
            l_cnt +=1
    
    layer_33 = x
    x = initx
    
    # ============Feature extraction network with kernel size 5x5============
    for i in range(len(filters_num)):
        for j in range(block_num[i]):
          
            if ((j==block_num[i]-1) & (i<len(filters_num)-1)):
                x = residual_block(x, filters_num[i], filters_num[i+1], 2, True, 'Residual5Block%d'%(l_cnt), isTraining, k_size=5)
                print('[L-%d] Build %dth connection layer %d from %d to %d channels' % (l_cnt, i, j, filters_num[i], filters_num[i+1]))
            else:
                x = residual_block(x, filters_num[i], filters_num[i], 1, False, 'Residual5Block%d'%(l_cnt), isTraining, k_size=5)
                print('[L-%d] Build %dth residual block %d with %d channels' % (l_cnt,i, j, filters_num[i]))
            l_cnt +=1
    layer_55 = x
    print("Layer33's shape", layer_33.get_shape().as_list())
    print("Layer55's shape", layer_55.get_shape().as_list())

    x = tf.concat([layer_33, layer_55], 3)
    
    # ============ Classifier Learning============
    
    classification_weights = []
    x_shape = x.get_shape().as_list()
    dense1 = x_shape[1]*x_shape[2]*x_shape[3]
    W = tf.get_variable("featW", [dense1, 128], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable("featB", [128], initializer=tf.truncated_normal_initializer())
    dense1 = tf.reshape(x, [-1, dense1])
    feat = tf.nn.softmax(tf.matmul(dense1, W) + b)
    classification_weights.append(W)
    
    with tf.variable_scope('Final'):
        x = batchnorm(x, isTraining, name='FinalBn')
        x = activation(x)
        wo, bo=initializer(filters_num[-1]*2, n_classes, "FinalOutput")
        x = conv2d('final', x, wo, bo, 1, "SAME")
        saliency = tf.argmax(x, 3)
        x=tf.reduce_mean(x, [1, 2])

        W = tf.get_variable("FinalW", [n_classes, n_classes], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable("FinalB", [n_classes], initializer=tf.truncated_normal_initializer())

        out = tf.matmul(x, W) + b
        classification_weights.append(W)
                            

    return out, feat, saliency, classification_weights


if not os.path.isdir('logs'):
    os.mkdir('logs')

tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# Read Training and Validation Data
train_data, train_labels, num_train_samples, val_data, val_labels, num_val_samples = setup_inputs(sess, data_file, image_dir, batch_size=batch_size)
print("Found %d training images, and %d validation images..." % (num_train_samples, num_val_samples))

max_iter = num_train_samples * total_epochs

def is_first_batch_in_epoch(step):
    return (step*batch_size) % num_train_samples < batch_size

# Initialize the model for training set and validation sets
with tf.variable_scope("ResNet") as scope:
    pred, feat, _, classification_weights = ResNet(train_data, True)
    scope.reuse_variables()
    valpred, _, saliency, _ = ResNet(val_data, False)


# Forming the triplet loss by hard-triplet sampler  
with tf.name_scope('Triplet_loss'):
    sialoss = tri.batch_hard_triplet_loss(train_labels, feat, margin, squared=False)

def compute_regularization(classification_weights):
    regularization_loss = 0
    for weights in classification_weights:
        regularization_loss += tf.nn.l2_loss(weights) 

    return regularization_loss

# Forming the cross-entropy loss and accuracy for classifier learning
with tf.name_scope('Loss_and_Accuracy'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        t_vars = tf.trainable_variables() 
        cost = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=pred) + \
               regularization_lambda * compute_regularization(classification_weights) 
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=t_vars)
        sia_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(sialoss)

    training_predictions = tf.equal(tf.argmax(pred, 1), train_labels)
    accuracy = tf.reduce_mean(tf.cast(training_predictions, tf.float32))
    validation_predictions = tf.equal(tf.argmax(valpred, 1), val_labels)
    validation_accuracy = tf.reduce_mean(tf.cast(validation_predictions, tf.float32))

  
tf.summary.scalar("Triplet_loss", sialoss)
tf.summary.scalar('Loss', cost)
tf.summary.scalar('Training_Accuracy', accuracy)
tf.summary.scalar('Validation_Accuracy', validation_accuracy)

datestring = time.strftime("%Y_%m_%d-%H_%M_%S")

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)
step = 0

writer = tf.summary.FileWriter("logs/" + datestring + "/", sess.graph)
summaries = tf.summary.merge_all()

print("We are going to train fake detector using ResNet based on triplet loss!!!")
print("num_train_samples " + str(num_train_samples))
print("num_val_samples " + str(num_val_samples))
start_lr = lr
valaccs = []
precisions = []
recalls = []

while (step * batch_size) < max_iter:
    epoch1=np.floor((step*batch_size)/num_train_samples)

    # trigger learning rate scheduling
    if (is_first_batch_in_epoch(step) & (lr == 1e-4) & (epoch1 >= num_cffn_epochs)):
        lr /= 10
    
    if (is_first_batch_in_epoch(step) & (lr == 1e-5) & (epoch1 >= num_cffn_epochs + 16)):
        lr /= 10
    
    if epoch1 <= num_cffn_epochs:
        sess.run([sia_optimizer],  feed_dict={learning_rate: lr})
    else:
        sess.run([optimizer],  feed_dict={learning_rate: lr})
        
    if (step % 15000 == 1) & (step > 15000):
        save_path = saver.save(sess, "checkpoints/tf_deepUD_tri_model_iter_%d.ckpt" % (step))
        print("Model saved in file at iteration %d: %s" % (step*batch_size,save_path))

    if step > 0 and step % print_interval == 0:
        # calculate the loss
        loss, train_accuracy, summaries_string, triplet_loss = sess.run([cost, accuracy, summaries, sialoss])
        print("Iter=%d/epoch=%d, Loss=%.6f, Triplet loss=%.6f, Training Accuracy=%.6f, lr=%f" % (step*batch_size, epoch1, loss, triplet_loss, train_accuracy, lr))
        writer.add_summary(summaries_string, step)
    
    if step > 0 and step % validation_interval == 0:
        valacc=[]
        a2, preds, vlbls = sess.run([validation_accuracy, tf.argmax(valpred, 1), val_labels])
        valacc.append(a2)
        conf_mat = tf.math.confusion_matrix(vlbls, preds)
        conf_mat = conf_mat.eval(session=sess)
        precision = compute_precision(conf_mat)
        recall = compute_recall(conf_mat)

        print("Iter=%d/epoch=%d, Validation Accuracy=%.6f" % (step*batch_size, epoch1 , np.mean(valacc)))
        valaccs.append(np.mean(valacc))
        precisions.append(precision)
        recalls.append(recall)

        # Implement early stopping
        if np.mean(valacc) >= val_threshold and epoch1 >= 15:
            break;

        if np.mean(valacc) >= 0.89 and lr == 1e-5:
            lr /= 10

  
    step += 1
print("Optimization Finished!")
save_path = saver.save(sess, "checkpoints/tf_deepUD_tri_model.ckpt")
with open("valaccs/a" + datestring + ".txt", "w+") as f:
    f.write("Validation Accuracy:\n")
    f.write(str(valaccs))
    f.write("\nPrecision:\n")
    f.write(str(precisions))
    f.write("\nRecall:\n")
    f.write(str(recalls))
f.close()
print("Model saved in file: %s" % save_path)

preds, labels = sess.run([tf.argmax(valpred, 1), val_labels])
conf_mat = tf.math.confusion_matrix(labels, preds)
conf_mat = conf_mat.eval(session=sess)
# Normalize the Confusion Matrix to get percentages
cfsum = np.sum(conf_mat)
cf_norm = conf_mat/cfsum
plot_filename = "plots/conf_mat" + datestring + ".jpg"
print("Confusion Matrix - saved to " + plot_filename)
print(conf_mat)
sns.heatmap(cf_norm, annot=True, fmt='.2%', cmap='Blues')
plt.savefig(plot_filename)

# Compute F1 score, precision, and recall
precision = compute_precision(conf_mat)
recall = compute_recall(conf_mat)
f1 = compute_f1(conf_mat)
print("F1 Score: %0.6f, Precision: %0.6f, Recall: %0.6f " % (f1, precision, recall))
