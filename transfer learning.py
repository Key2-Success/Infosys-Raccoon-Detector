import numpy as np
import keras

from keras.applications import vgg16, inception_v3, resnet50, mobilenet
 
#Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')

#Load the Inception_V3 model
inception_model = inception_v3.InceptionV3(weights='imagenet')
 
#Load the ResNet50 model
resnet_model = resnet50.ResNet50(weights='imagenet')
 
#Load the MobileNet model
mobilenet_model = mobilenet.MobileNet(weights='imagenet')


#####################

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt

filename = "C:/Users/KK/Documents/Kitu/College/Senior Year/Important Info/Infosys/Project/training/pumpkin/pumpkin.jpeg"

# load an image in PIL format
original = load_img(filename, target_size=(224, 224))
print('PIL image size',original.size)
plt.imshow(original)
plt.show()
 
# convert the PIL image to a numpy array
# IN PIL - image is in (width, height, channel)
# In Numpy - image is in (height, width, channel)
numpy_image = img_to_array(original)
plt.imshow(np.uint8(numpy_image))
plt.show()
print('numpy array size',numpy_image.shape)
 
# Convert the image / images into batch format
# expand_dims will add an extra dimension to the data at a particular axis
# We want the input matrix to the network to be of the form (batchsize, height, width, channels)
# Thus we add the extra dimension to the axis 0.
image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)
plt.imshow(np.uint8(image_batch[0]))


# prepare the image for the VGG model
processed_image = vgg16.preprocess_input(image_batch.copy())
 
# get the predicted probabilities for inception
predictions = inception_model.predict(processed_image)
 
# convert the probabilities to class labels
# We will get top 5 predictions which is the default
label = decode_predictions(predictions)
print (label)


####### object detection
from keras.preprocessing import image
from inception_v3 import InceptionV3, preprocess_input


# load pretrained models
vgg_model = vgg16.VGG16(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
resnet_model = resnet50.ResNet50(weights='imagenet')
mobilenet_model = mobilenet.MobileNet(weights='imagenet')


# images
# tomato, car, pumpkin, watermelon
img_path = "C:/Users/KK/Documents/Kitu/College/Senior Year/Important Info/Infosys/tomato/test/tomato.jpg"
img_path = "C:/Users/KK/Documents/Kitu/College/Senior Year/Important Info/Infosys/Project/car/993front.jpg"
img_path = "C:/Users/KK/Documents/Kitu/College/Senior Year/Important Info/Infosys/pumpkin/pumpkin-fields.jpg"
img_path = "C:/Users/KK/Documents/Kitu/College/Senior Year/Important Info/Infosys/watermelon/W020070111678783415229.jpg"


# inceptionv3
# preprocess image
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = inception_v3.preprocess_input(x)

# plot original image
original = load_img(img_path, target_size=(299, 299))
print('PIL image size',original.size)
plt.imshow(original)

# predict object class
preds = inception_model.predict(x)
print('Predicted:')
decode_predictions(preds)


# mobilenet
# preprocess image
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = mobilenet.preprocess_input(x)

# plot original image
original = load_img(img_path, target_size=(224, 224))
print('PIL image size',original.size)
plt.imshow(original)

# predict object class
preds = mobilenet_model.predict(x)
print('Predicted:')
decode_predictions(preds)


# resnet
# preprocess image
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = resnet50.preprocess_input(x)

# plot original image
original = load_img(img_path, target_size=(224, 224))
print('PIL image size',original.size)
plt.imshow(original)

# predict object class
preds = resnet_model.predict(x)
print('Predicted:')
decode_predictions(preds)


##### transfer learning
# load pretrained model
import keras
from keras.applications import inception_v3
 
resnet_conv =keras.applications.resnet50.ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

# re run
import numpy as np
import matplotlib.pyplot as plt
from __future__ import print_function
import keras
from keras.utils import to_categorical
import os
from keras.preprocessing.image import ImageDataGenerator, load_img


from keras.applications import VGG16
 
vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

vgg_conv.summary()


# extract features
#train_dir = "C:\Users\KK\Documents\Kitu\College\Senior Year\Important Info\Infosys\Project\training"
train_dir = "C:/Users/KK/Documents/Kitu/College/Senior Year/Important Info/Infosys/Project/training"
test_dir = "C:/Users/KK/Documents/Kitu/College/Senior Year/Important Info/Infosys/Project/testing"
#test_dir = "C:\Users\KK\Documents\Kitu\College\Senior Year\Important Info\Infosys\Project\testing"

nTrain = 600
nVal = 150

# generate batches of images and labels
datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
batch_size = 20
 
train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain,3))
 
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle= True)

# train features
i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * batch_size : (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTrain:
        break
         
train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))


# validation features
validation_features = np.zeros(shape=(nVal, 7, 7, 512))
validation_labels = np.zeros(shape=(nVal,3))

validation_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle="shuffle")

i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nVal:
        break

validation_features = np.reshape(validation_features, (nVal, 7 * 7 * 512))

# create own model
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features,
                    train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_data=(validation_features,validation_labels))
# save model
from keras.models import load_model
model.save('my_model.h5') 
model = load_model('my_model.h5') # returns model

# look into which are wrong
fnames = validation_generator.filenames
ground_truth = validation_generator.classes
label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

# look into predictions
predictions = model.predict_classes(validation_features)
prob = model.predict(validation_features)

# print # of errors
errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),nVal))

# which images are wrong
from keras.preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt

for i in range(len(errors)):
    pred_class = np.argmax(prob[errors[i]])
    pred_label = idx2label[pred_class]
    
    print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        prob[errors[i]][pred_class]))
    
    original = load_img('{}\{}'.format(test_dir,fnames[errors[i]]))
    plt.imshow(original)
    plt.show()