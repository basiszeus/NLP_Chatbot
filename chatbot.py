import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow
from tensorflow import keras
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model("check3.model")

#4 functions to processe predict and answer

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence) # a bag of 0 according to the length of the words (list)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1 #if the word is the same as the one in the loop we will mark it as 1
    return np.array(bag) #return a numpy array of it (creating a bag of words)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    error_threshold = 0.20
    results = [[i,r] for i,r in enumerate(res) if r >error_threshold]

    results.sort(key=lambda x:x[1], reverse=True) #Hifgest probability first
    return_list = []
    for r in results:
        return_list.append({"intent":classes[r[0]], 'probability':str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result = random.choice(i['responses'])
            break
    return result

while True:
    message = input('')
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)