import os, re

from flask import Flask, render_template, request
from flask_cors import CORS

import pickle
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)
CORS(app)

# loading 
model_path = os.path.join(app.root_path, 'model')
pickles_path = os.path.join(app.root_path, 'pickles')

model = keras.models.load_model(model_path)

with open(pickles_path+'/preproc_french_sentences.pickle', 'rb') as handle:
    preproc_french_sentences = pickle.load(handle)
with open(pickles_path+'/english_tokenizer.pickle', 'rb') as handle:
    english_tokenizer = pickle.load(handle)
with open(pickles_path+'/french_tokenizer.pickle', 'rb') as handle:
    french_tokenizer = pickle.load(handle)


# predictions helpers
def logits_to_text(logits, tokenizer):
  index_to_words = {id: word for word, id in tokenizer.word_index.items()}
  index_to_words[0] = '<PAD>'
  return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

def final_predictions(text):
  y_id_to_word = {value: key for key, value in french_tokenizer.word_index.items()}
  y_id_to_word[0] = '<PAD>'
  sentence = [english_tokenizer.word_index[word] for word in text.split()]
  sentence = pad_sequences([sentence], maxlen=preproc_french_sentences.shape[-2], padding='post')
  prediction =  logits_to_text(model.predict(sentence[:1])[0], french_tokenizer)
  return prediction

# route
@app.route("/prediction", methods=['GET', 'POST'])
def predict():
    args = request.args
    if (args and args['text']):
        text = args['text']
        try:
            prediction = final_predictions(re.sub(r'[^\w]', ' ', text))
            prediction = prediction.replace("<PAD>"," ").strip()
            return {'prediction':prediction}
        except Exception as e:
            return {
                'message':"Something went wrong", 
                "exception" : str(e)+ " is not in trained vocabulary"}
    else:
        return {'message':"no text"}


if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 3030))
    app.run(host='0.0.0.0', port=port)