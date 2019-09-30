import os
import numpy as np
from skimage.io import imread
from skimage.filters import threshold_otsu
import keras
from keras import optimizers
from keras.layers import *
from keras.models import Model
from keras.utils import to_categorical

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

def read_training_data(training_directory):
    training_data=[]
    for each_letter in letters:
        for each in range(10):
            image_path = os.path.join(training_directory, each_letter, each_letter + '_' + str(each) + '.jpg')
            # read each image of each character
            img_details = imread(image_path, as_gray=True)
            # converts each character image to binary image
            binary_image = img_details < threshold_otsu(img_details)
            # 20*20 image becomes 1*400
            # in machine learning terms that's 400 features
            flat_bin_image = binary_image.reshape(-1)
            training_data.append([flat_bin_image, each_letter])

    return (np.array(training_data))


# current_dir = os.path.dirname(os.path.realpath(__file__))
# training_dataset_dir = os.path.join(current_dir, 'train')
print('reading data')
training_dataset_dir = './train20X20'
training_data = read_training_data(training_dataset_dir)
print(training_data.shape)
print('reading data completed')

#Randomizing the data for better training of the model 
import random 
random.shuffle(training_data)

X=[]
y=[]
for features, labels in training_data:
    X.append(features)
    y.append(labels)

X=np.array(X)
print(X.shape)
print(y)

#encodeing string labels to integers
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
#print(integer_encoded)
labels_encoded=to_categorical(integer_encoded, num_classes=34) #since it is multiclass classification it converts 34 class vectors to binary class matrix
#print(labels_encoded)

X=X/255
print('=======================training model============================')      #keras module used 
Inp=Input((400,))
x=(Dense(300, activation='tanh'))(Inp)
x=(Dense(200, activation='tanh'))(x)
x=(Dense(100, activation='tanh'))(x)
x=(Dense(100, activation='tanh'))(x)
x=(Dense(100, activation='tanh'))(x)
x=(Dense(100, activation='tanh'))(x)
output= (Dense(34, activation='softmax'))(x)

model=Model(Inp, output)
model.summary()

learning_rate=0.001
epochs=30
batch_size=20
adam=optimizers.Adam(lr=learning_rate)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, labels_encoded, batch_size=batch_size, epochs=epochs, validation_split=0.2)
model.save('text.pb') # accuraxcy of  
