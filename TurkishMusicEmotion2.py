# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:02:16 2023

This program analyse the characteristics of an audio sample, 
             build a CNN deep learning model,
             train it with given data set and
             predict given song types by emotion.

@author: ovunc
"""
print("""
For librosa library, code explanations and inspiration thanks to Yazılım Teknolojileri Akademisi... :)    
      
Problem: Multiclass Classification

Objective: Prediction of Song Types by Emotion

Source: https://www.kaggle.com/datasets/blaler/turkish-music-emotion-dataset

Bilal Er, M., & Aydilek, I. B. (2019). Music emotion recognition by using chroma spectrogram and deep visual features. 
Journal of Computational Intelligent Systems, 12(2), 1622–1634. International Journal of Computational Intelligence Systems, 
DOI: [Web Link] https://doi.org/10.2991/ijcis.d.191216.001

Introduction:
    
This is a database of music for happy, sad, angry, relax emotions. 
To prepare the dataset, verbal and non-verbal music are selected from different genres of Turkish music. 
The dataset is designed as a discrete model, and there are four classes. 
A total of 100 music pieces are determined for each class in the database to have an equal number of samples in each class. 
There are 400 samples in the original dataset as 30 seconds from each sample.

In this project we will use this samples to train our deep learning model and so try to classify songs by given emotions: happy, sad, angry, relax. 

To do this:
    i. We will prepare our dataset for analysis and extract sound signal features from audio files using MFCC (Mel-Frequency Cepstral Coefficients). 
    For detailed info about MFC: https://www.youtube.com/watch?v=4_SH2nfbQZ8&t=0s
    ii. Then we will build a CNN (Convolutional Neural Networks) model and train our model with our dataset.  
    iii. Finally we predict an audio file's emotion class using our model.
""")

print("""
      Please wait. Importing libraries...
      """)
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


print("""
      Every audio signal has its own characteristics. 
      By using librosa library we can get characteristics of every audio signal.
      Librosa converts any stereo signal into mono. 
      This means given audio signal is converted in a one dimensional form.
      Let's extract MFCC's for every audio file in the dataset...
      """)
mfcc_num = 50
seed = 20

audio_dataset_path = input("""
                           Enter the absolute path where all audio files exist to train model or 
                           Drag the folder into the terminal and
                           Press 'enter':
                           """).strip("'") 
                           
metadata = pd.read_csv(input("""
                             Enter the absolute path where the list of all audio files names, 
                             categories etc. or 
                             Drag the file into the terminal and
                             Press 'enter':
                             """).strip("'"))
                             
# We don't need date, time and size features for analysis. So we can get rid of them.
metadata = metadata.drop(["date", "time", "size"], axis = 1)

def features_extractor(filename):
    audio, sample_rate = librosa.load(filename, res_type = "kaiser_fast") 
    mfccs_features = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = mfcc_num) #n_mfcc: number of MFCCs to return
    mfccs_scaled_features = np.mean(mfccs_features.T, axis = 0)
    
    return mfccs_scaled_features

print("""
      Iterating through audio files and extract features using MFCC's. This may take a while...
      """) 
extracted_features = []
for index_num, row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path), str(row["fold"])+'/', str(row["name"]))
    final_class_labels = row["class"]
    data = features_extractor(file_name)
    extracted_features.append([data, final_class_labels])
    
# Converting extracted_features to dataframe
extracted_features_df = pd.DataFrame(extracted_features, columns = ["feature", "class"])
extracted_features_df    
    
# Splitting data into features and target
X = np.array(extracted_features_df["feature"].tolist())
y = np.array(extracted_features_df["class"].tolist())

# One hot encoding
labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

# Splitting data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = seed)
  
#### Creating CNN model:
num_batch_size = 32
epochscount = 150
num_labels = 4

model = Sequential()
#### 1.Hidden layer
model.add(Dense(128, input_shape = (X_train.shape[1], ) ))
model.add(Activation("relu"))
model.add(Dropout(0.5))
#### 2.Hidden layer
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
#### 3.Hidden layer
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.5))
#### Output layer
model.add(Dense(num_labels))
model.add(Activation("softmax"))
#### Model summary
model.summary()
#### Compiling
model.compile(loss = "categorical_crossentropy", metrics = ["accuracy"], optimizer = "adam")

print("""
      Training the model may take a while, too...
      """)
mdl = model.fit(X_train, y_train, batch_size = num_batch_size, epochs = epochscount, validation_data = (X_test, y_test), verbose = 1)

# Predictions and model accuracy
dl_acc = model.evaluate(X_test, y_test)[1]
print("""
      Deep Learning Model Accuracy:
      """, dl_acc)
dl_acc_tr = model.evaluate(X_train, y_train)[1]
print("""
      Deep Learning Training Accuracy:
      """, dl_acc_tr)
y_pred = np.argmax(model.predict(X_test), axis = -1)
print("""
      Predictions:
      """, y_pred)

print("""
      ^^^The history of the model's accuracy and loss is above^^^
      """)  
plt.plot(mdl.history['accuracy'])
plt.plot(mdl.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc ='center right')
plt.rcParams['figure.dpi'] = 500
plt.show()
        
plt.plot(mdl.history['loss'])
plt.plot(mdl.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc = 'center right')
plt.rcParams['figure.dpi'] = 500
plt.show()

print(     
"""
Since we have trained our model once, we don't need to do it again. 
So we can make unlimited predictions.
Let's read the audio sample using librosa and predict it's emotion type.
""")

while True:
    
    filename = input("""
                     To quit please type 'exit' or
                     Enter the absolute path of a song file or 
                     Drag a song file into the terminal you want to predict type of emotion and
                     Press 'enter':
                     """).strip("'") 
                     
    if filename.lower().strip() == "exit":
        print("Thank you for your trial. See you later...")
        break
    else:
        librosa_audio_data, librosa_sample_rate = librosa.load(filename) 
        print("""
              ^^^Please wait. The librosa audio data plot of your song will appear above...^^^
              """)
        plt.figure(figsize = (12, 4))
        plt.title(filename.split("/")[-1])
        plt.plot(librosa_audio_data)
        plt.show()
        
        print(
        """
        The MFCC summarises the frequency distribution across the window size, 
        so it is possible to analyse both the frequency and time characteristics of the sound. 
        These audio representations will allow us to identify features for classification.
        """)    
        
        mfccs = librosa.feature.mfcc(y = librosa_audio_data, 
                                     sr = librosa_sample_rate, 
                                     n_mfcc = mfcc_num) 
        
        print("""
              MFCC's
              """, mfccs, mfccs.shape)

        mfccs_scaled_features = features_extractor(filename).reshape(1,-1) #for right shape
        result_array = model.predict(mfccs_scaled_features)
        result_classes = ["ANGRY", "HAPPY", "RELAX","SAD"]
        result = np.argmax(result_array[0])
        print("""
              Type of this song by emotion is""", result_classes[result]) 
        
        

    