import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Read the json file and store it in a variable
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['!', '?','.',',']

for intent in intents['intents']: # we will loop in the json file from they key intents we will do the following
    for pattern in intent['patterns']: #word extraction from the  pattern
        world_list = nltk.word_tokenize(pattern)
        words.extend(world_list) #Add each word to the list
        documents.append((world_list, intent['tag'])) #Append each word_list with thier tag
        if intent['tag'] not in classes: #Add new tag to the classes list if it is not already present in the classes list
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters] #Lemmatization of the results
words = sorted(set(words))  #set eliminate duplicate
classes = sorted(set(classes)) #set eliminate duplicate

#Save then to use them in training

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
print(words)

#Training

# Change the words into numerical values :

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1)if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle the data

random.shuffle(training) # Shuffle the training data
training = np.array(training) #transforming training into np array

#X & Y

train_x = list(training[:,0]) #Features
train_y = list(training[:,1]) #Labelds

# Building the model :
import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(3000, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1500, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=30, batch_size =5 )

model.save('check3.model', hist)

print(words)

