# coding: utf-8

# In[1]:

import os
import sys
import shutil
import numpy as np
from random import shuffle
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD


#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

# In[2]:

# train, validation and test pathes of the datasets
original_path = './data/train'

train_path = './data/trainset'
valid_path = './data/validset'
#test_path = './data/test'
#aug_path = '/home/alan/Downloads/intel_dataset/augmented'
#os.mkdir(aug_path)


# In[3]:

classes = ['Type_1', 'Type_2', 'Type_3']


# In[4]:

# create training and validation set folders by splitting the
# original 'train' folder

#create train and validation folders
if not os.path.isdir(train_path):
    os.mkdir(train_path)
    
if not os.path.isdir(valid_path):
    os.mkdir(valid_path)

# proportion of the split training set relative to the original training images
train_proportion = 0.9
n_origin = [] # number of original train images per class
n_train = [] # number of split train images per class
n_valid = [] # number of validation images per class

for c in classes:
    source = os.path.join(original_path, c)
    

    file_list = os.listdir(source)
    shuffle(file_list)
    
    # the index at which the list is to be split
    split_idx = int(len(file_list) * train_proportion)
    train_list = file_list[:split_idx]
    valid_list = file_list[split_idx:]

    #split the original image folder if it is not already done
    train_type_path = os.path.join(train_path, c)
    valid_type_path = os.path.join(valid_path, c)
    
    if not os.path.isdir(train_type_path) or        not os.path.isdir(valid_type_path):
            
        #create type subfolders
        os.mkdir(train_type_path)
        os.mkdir(valid_type_path)

        #copy files in new folders
        for im_name in train_list:
            os.symlink(os.path.relpath(os.path.join(source, im_name),
                                       os.path.join(train_path, c) ),
                        os.path.join(train_path, c, im_name))

        for im_name in valid_list:
            os.symlink(os.path.relpath(os.path.join(source, im_name),
                                       os.path.join(valid_path, c)),
                        os.path.join(valid_path, c, im_name))
        
    # sizes of different image sets
    n_origin.append(len(file_list))
    n_train.append(len(train_list))
    n_valid.append(len(valid_list))
    
print("Number of original training images (per type):", n_origin)
print("Number of split training images (per type):", n_train)
print("Number of split validation images (per type):", n_valid)


# get the number of images in each class
n_train = []
n_valid = []

for c in classes:
    n_train.append(len(os.listdir(os.path.join(train_path,c))))
    n_valid.append(len(os.listdir(os.path.join(valid_path,c))))

print("Number of split training images (per type):", n_train)
print("Number of split validation images (per type):", n_valid)

# In[6]:

# define hyper parameter
n_classes = len(classes)
img_height = 299
img_width = 299
n_epochs = 50
batch_size = 16
learning_rate = 0.0001
momentum = 0.9
decay = 0.0

train_steps = sum(n_train) // batch_size

if sum(n_train) % batch_size != 0:
    train_steps+= 1
    
train_steps = 4 * train_steps

valid_steps =  sum(n_valid) // batch_size

if sum(n_valid) % batch_size != 0:
    valid_steps += 1



# In[7]:

# create the incpetion graph from keras applications
inception_app = InceptionV3(include_top = False,
                            weights = 'imagenet',
                            input_tensor = None,
                            input_shape = (img_height,img_width,3),
                            pooling = 'avg')


# In[8]:

# print same information about the last layer
print("Last layer is: {} \nwith shape {}"
      .format(inception_app.layers[-1].name,
              inception_app.layers[-1].output_shape))


# In[9]:

# create the fully connected layer of the model
output = inception_app.output
predictions = Dense(units = n_classes,
                    activation = 'softmax')(output)


# In[10]:

# create the model to be trained
model = Model(inputs = inception_app.input,
              outputs= predictions)


# In[11]:

# compile the model with loss and optimizer

optimizer = Adam(lr = 0.001,
                 beta_1 = 0.9,
                 beta_2 = 0.999,
                 epsilon = 1e-08,
                 decay = 0.0)

"""
optimizer = SGD(lr = learning_rate,
                momentum = momentum,
                decay = decay,
                nesterov = True)
"""
model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizer,
              metrics = ['accuracy'])


# In[12]:

# define training data generators
train_gen = ImageDataGenerator(rescale = 1.0/255,
                               rotation_range = 20,
                               width_shift_range = 0.1,
                               height_shift_range = 0.1,
                               shear_range = 1.0,
                               zoom_range = 0.2,
                               horizontal_flip = True,
                               vertical_flip = True)

train_batch_provider = train_gen.flow_from_directory(directory = train_path,
                                                     target_size = (img_height, img_width),
                                                     classes = classes,
                                                     class_mode = "categorical",
                                                     batch_size = batch_size,
                                                     follow_links = True)
                                                     

# define validation data generators
test_gen = ImageDataGenerator(rescale = 1.0/255)

valid_batch_provider = test_gen.flow_from_directory(directory = valid_path,
                                                    target_size = (img_height, img_width),
                                                    classes = classes,
                                                    class_mode = "categorical",
                                                    batch_size = batch_size,
                                                    follow_links = True)


# In[13]:

# set a callback
best_model_file = "./weights.{epoch:02d}-{val_loss:.2f}.hdf5"
best_model = ModelCheckpoint(filepath = best_model_file,
                             monitor = 'val_loss',
                             verbose = 1,
                             save_best_only = True,
                             mode = 'auto')
                             


# In[ ]:

# train the model
#model = load_model('./weights.05-0.73.hdf5')

model.fit_generator(generator = train_batch_provider,
                    steps_per_epoch = sum(n_train),
                    epochs  = n_epochs,
                    validation_data = valid_batch_provider,
                    validation_steps = sum(n_valid),
                    callbacks = [best_model])



