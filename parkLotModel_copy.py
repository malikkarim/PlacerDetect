# -*- coding: utf-8 -*-
"""
@author: Karim
"""
# import required packages and libraries
import tensorflow as tf
#from tensorflow.python.keras import models
from keras.layers import Input
#from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam,RMSprop
import tensorflow.keras.backend as K
import keras
# from keras.models import *
# from keras.layers import *
# from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras import backend as keras
#from keras import backend as K
#import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers, losses
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
#from tensorflow.python.keras import backend as K
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import numpy as np 
import os
#import skimage.io as io
#import skimage.transform as trans
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from keras.utils import np_utils
import h5py
import timeit
from sklearn.preprocessing import MinMaxScaler
#===================================================
# GPU sttings .................
#K.clear_session()
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0}) 
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)

from tensorflow.python.client import device_lib
print(device_lib)

#------------------------------------------------
os.chdir(r"D:\Malik\skySat")# directory to your data
#-----------------------------------------
images = h5py.File(r'C:\Users\karim\Desktop\parkLot_proj\skySat\images.h5', 'r')
image = images.get('images')
print(image.shape)
#-------------------------------------------
labels = h5py.File(r'C:\Users\karim\Desktop\parkLot_Proj\skySat\labels.h5', 'r')
label  = labels.get('labels')
print(label.shape)

#===========================================
#Alternative functions for processing three images ...
# Loading training images and labels
def loadParkLot(): # function to load images
    path_2 = os.path.join(os.getcwd(),'images') 
    classes =  os.listdir(path_2)
    sampSize = np.size(classes)
    print("numb of images:", sampSize)
    print('[Progress] Reading into folder...')
    images = np.array([np.array(cv2.imread(path_2 +'\\'+ img)) for img in classes if img.endswith("tif")])   
    #images /= 255.0 #normalize data
    return images
images = loadParkLot() 
#---------------------------------------------------------------------------
def loadLabel(): 
    path1 = os.path.join(os.getcwd(),'labels')   
    classes =  os.listdir(path1)
    sampSize = np.size(classes)
    print("numb of images:", sampSize)
    print('[Progress] Reading into folder...')
    labels = np.array([np.array(cv2.imread(path1 +'\\'+ img)) for img in classes if img.endswith("tif")], "f")   
    return labels
labels = loadLabel() 
#=================================================================================

# call and display sample images
fig = plt.figure(figsize=(12,12))# define size of imagelets == 10*10
plt.title('Sample images-{}'.format('test maps'))
for i in range(16):
    ax = fig.add_subplot(4,4,i+1) # stuck maps 6*6 == 36 images
    #plt.imshow(images[i][:,:,2])
    plt.imshow(images[i])
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout() 
plt.show()
#----------------------------------
# call labels/mask
fig = plt.figure(figsize=(12,12))# define size of imagelets == 10*10
plt.title('Sample labels-{}'.format('test maps'))
for i in range(16):
    ax = fig.add_subplot(4,4,i+1) # stuck maps 6*6 == 36 images
    plt.imshow(label[i])
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout() 
plt.show()
#=========================================================
# manually split data into training and test set
Xtrain = image[:700]
Ytrain = label[:700]
print('Train mask size :', Ytrain.shape)
#Convert train mask labels into one-hot-encode
Ytrain = tf.keras.utils.to_categorical(Ytrain, num_classes=None, dtype='float32')
#Ytrain = Ytrain[:,:,:,1:]
print(Ytrain.shape)
plt.imshow(Xtrain[2,:,:,1])
plt.imshow(Ytrain[2,:,:,1])

#-----------------------------------
#test data
Xtest = image[702:]
print('test data', Xtest.shape)
Ytest = label[702:]
plt.imshow(Ytest[0,:,:])
print('Test mask size :', Ytest.shape)
#Convert test mask labels into one-hot-encode
Ytest = tf.keras.utils.to_categorical(Ytest, num_classes=None, dtype='float32')
print(Ytest.shape)
plt.imshow(Ytest[0,:,:,1])
plt.imshow(Xtest[0,:,:,1])

#=========================================================================
#Define loss functions and accuracy metrics
# Dice losses 
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def mce_dice_loss(y_true, y_pred):
    loss = losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

#==============================================================================
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05): 
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis = 0)

#------------------------------------------------------------------------------
# Define a custom weighted loss
# set weight for each class based on its abundance
class_weights = np.array([0.00000001,1, 0.8, 1, 1, 0.9,1, 1]) 
weights = K.variable(class_weights)
def weighted_categorical_crossentropy(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis = -1, keepdims = True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calculate loss and weight loss
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1) # + dice_loss(y_true, y_pred)
    return loss

#==============================================================================

# Other parameter values
batch = 12
epoch = 50
verbose = 1
num_class = 8
Valid_split = 0.25
input_shape = Xtrain[0].shape
#input_shape = (512,512,1)
#=====================================================================================================
# Define UNET model  function 
def unet(input_size = input_shape, n_filters = 14, filt_size = 3):
    inputs = Input(input_size)
    conv1 = Conv2D(n_filters*1, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(n_filters*1, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    bNorm1 = BatchNormalization()(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(bNorm1)
    conv2 = Conv2D(n_filters*2, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(n_filters*2, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    bNorm2 = BatchNormalization()(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(bNorm2)
    conv3 = Conv2D(n_filters*4, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(n_filters*4, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    bNorm3 = BatchNormalization()(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(bNorm3)
    conv4 = Conv2D(n_filters*8, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(n_filters*8, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv4)   
    bNorm4 = BatchNormalization()(conv4)
    #drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2)) (bNorm4)#(drop4)

    conv5 = Conv2D(n_filters*16, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(n_filters*16, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    bNorm5 = BatchNormalization()(conv5)
    #drop5 = Dropout(0.5)(conv5)
  #  Expansion/contraction block 
    up6 = Conv2DTranspose(n_filters*8, (filt_size, filt_size), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(bNorm5)#(drop5)
    #up6 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([bNorm4,up6], axis = 3)
    #merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(n_filters*8, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(n_filters*8, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    up7 = Conv2DTranspose(n_filters*4, (filt_size,filt_size), strides=(2, 2), padding='same', kernel_initializer = 'he_normal') (conv6)
    #up7 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([bNorm3,up7], axis = 3)
    conv7 = Conv2D(n_filters*4, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(n_filters*4, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(n_filters*2, (filt_size, filt_size), strides=(2, 2), padding='same', kernel_initializer = 'he_normal') (conv7)
    #up8 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([bNorm2,up8], axis = 3)
    conv8 = Conv2D(n_filters*2, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(n_filters*2, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    
    up9 = Conv2DTranspose(n_filters*1, (filt_size, filt_size), strides=(2, 2), padding='same', kernel_initializer = 'he_normal') (conv8)
    #up9 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([bNorm1,up9], axis = 3)
    conv9 = Conv2D(n_filters*1, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(n_filters*1, filt_size, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    output = Conv2D(num_class, 1, activation = 'sigmoid')(conv9)
    
    #model = models.Model(inputs=[inputs], outputs=[output])

    model = Model(inputs = inputs, outputs = output)
   # model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])
    #model.compile(optimizer = Adam(lr = 1e-4), loss= mce_dice_loss, metrics=[dice_loss])
    model.compile(optimizer = Adam(lr = 1e-3), loss= weighted_categorical_crossentropy, metrics=[dice_loss])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    model.summary()

   # if(pretrained_weights):
    	#model.load_weights(pretrained_weights)
    return model
#------------------------------------------------------------------------------
# Instantiate/initialize the UNET model instance for training    
UnetPlacer = unet()
#-----------------------------------------------------------------
# Training the model
start_time = timeit.default_timer()
save_model_path = 'C:/Users/karim/Desktop/parkLot_Proj/UNet_sky1.h5' # saving the best model to this path
cp = keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_loss', save_best_only=True, verbose=1)
es = EarlyStopping(mode='min', monitor='val_loss', patience=5, verbose=1)
history = UnetPlacer.fit(Xtrain, Ytrain, batch_size = batch, epochs= epoch,\
            validation_split = Valid_split, verbose = verbose, callbacks = [cp,es])
end_time = timeit.default_timer()
ExecTime = (end_time - start_time)/3600 
print('RunTime(hrs):', ExecTime)
#--------------------------------------------------------------------
# save model history to csv for ploting later
hist_df = pd.DataFrame(history.history) 
hist_file = ('history.csv %s' % end_time)
with open(hist_file, mode='w') as f:
    hist_df.to_csv(f)
#-------------------------------------------------------------------------
# Set labels list and predict mask ...
label_names = (["Bacground","Road", "Building", "GreenArea", "Cars", "Trees", "Other1", "Other2"])
name2id = {v:k for k,v in enumerate(label_names)}
id2name = {k:v for k,v in enumerate(label_names)}
#-------------------------------------------------------------------------------------
# Create color codes for model predictions
label_codes = ([(105,105,105), (0,0,139), (225,105,125), (50,240,50),
                (255,0,0), (65,100,210), (80,250,250), (106,156,245)])    
code2id = {v:k for k,v in enumerate(label_codes)}
id2code = {k:v for k,v in enumerate(label_codes)}
#----------------------------------------------------------
# import or load the trained model
model = load_model('C:/Users/karim/Desktop/UNet_sky5.h5',
    custom_objects ={'dice_loss':dice_loss,'weighted_categorical_crossentropy':weighted_categorical_crossentropy})

# Predict image mask
pred_all = model.predict(Xtest[:25])
#pred_all = prediction[:]
batch_mask = Ytest[:25]
batch_img = Xtest[:25]
#--------------------------------------------------
# Define a function to plot segmentation maps
def onehot_to_rgb(onehot, colormap = id2code):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)
#------------------------------------------------------------

# loop over and plot images and ground truth
for i in range(0,np.shape(pred_all)[0]):
    
    fig = plt.figure(figsize=(15,15))
    
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(batch_img[i,:,:,3])
    ax1.title.set_text('Actual frame')
    ax1.grid(b=None)
       
    ax2 = fig.add_subplot(1,3,2)    
    ax2.set_title('Ground truth labels')   
    ax2.imshow(onehot_to_rgb(batch_mask[i],id2code))
    ax2.grid(b=None)
    
    ax3 = fig.add_subplot(1,3,3) 
    ax3.set_title('Predicted labels')
    ax3.imshow(onehot_to_rgb(pred_all[i],id2code))
    ax3.grid(b=None)
    
    plt.show() 
#------------------------------------------------------------------------
# Visualizing training history
plt.style.use('ggplot')
dice = history.history['dice_loss']
val_dice = history.history['val_dice_loss']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(50)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, dice, label='Training - Dice Loss')
plt.plot(epochs_range, val_dice, label='Validation - Dice Loss')
plt.legend(loc='upper right')
#plt.title('Dice loss ==> Placers')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training - WCE Loss')
plt.plot(epochs_range, val_loss, label='Validation - WCE Loss')
plt.legend(loc='upper right')
#plt.title('Weighted cross-entropy loss ==> Placers')
plt.savefig("Model-Loss.jpg", dpi = 600, bbox_inches = "tight", pad_inches = 0)
plt.show()


#pred = UnetPlacer.predict(xTrain[0:5])
#------------------------------------------------------------------------
# Compute confusion matrix
#test_res = testing(UnetPlacer, Xtrain, Ytrain, Xtest, Ytest)

def to_class_no(y_hot_list):
    y_class_list = []   
    n = len(y_hot_list)   
    for i in range(n):        
        out = np.argmax(y_hot_list[i])     
        y_class_list.append(out)       
    return y_class_list

#-----------------------------------------------------------------
num_classes = 8

def conf_matrix(Y_gt, Y_pred, num_classes = num_classes):   
    total_pixels = 0
    kappa_sum = 0
    sudo_confusion_matrix = np.zeros((num_classes, num_classes))
   
#    if len(Y_pred.shape) == 3:
#        h,w,c = Y_pred.shape
#        Y_pred = np.reshape(Y_pred, (1,))
    n = len(Y_pred)    
    for i in range(n):
        y_pred = Y_pred[i]
        y_gt = Y_gt[i]      
        #y_pred_hotcode = hotcode(y_pred)
        #y_gt_hotcode = hotcode(y_gt)
        
        pred = np.reshape(y_pred, (y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2]))
        gt = np.reshape(y_gt, (y_gt.shape[0]*y_gt.shape[1], y_gt.shape[2]))
        
        pred = [i for i in pred]
        gt = [i for i in gt]
        
        pred = to_class_no(pred)
        gt = to_class_no(gt)        
#        pred.tolist()
#        gt.tolist()
        gt = np.asarray(gt, dtype = 'int32')
        pred = np.asarray(pred, dtype = 'int32')
        conf_matrix = confusion_matrix(gt, pred, labels=[0,1,2,3,4,5])      
        kappa = cohen_kappa_score(gt,pred, labels=[0,1,2,3,4,5])
        pixels = len(pred)
        total_pixels = total_pixels+pixels     
        sudo_confusion_matrix = sudo_confusion_matrix + conf_matrix      
        kappa_sum = kappa_sum + kappa
    final_confusion_matrix = sudo_confusion_matrix    
    final_kappa = kappa_sum/n
    return final_confusion_matrix, final_kappa

#--------------------------------------------------
confusion_matrix_train, kappa_train = conf_matrix(batch_mask, pred_all, num_classes = num_classes)
print('Confusion Matrix for testing')
print(confusion_matrix_train)
print('Kappa Coeff for training without unclassified pixels')
print(kappa_train)
confMat = pd.DataFrame(confusion_matrix_train)
confMat.to_excel('confMat_savi_elu_cor2.xlsx', index = False)
#-------------------------------------------------------

# Function to compute accuracy for individual classes
def acc_of_class(class_label, conf_matrix, num_classes = num_classes):   
    numerator = conf_matrix[class_label, class_label] 
    denorminator = 0   
    for i in range(num_classes):
        denorminator = denorminator + conf_matrix[class_label, i]       
    acc_of_class = numerator/denorminator   
    return acc_of_class
# Compute acc
acc_of_class(6, confusion_matrix_train,num_classes)
#------------------------------------------------------------------
