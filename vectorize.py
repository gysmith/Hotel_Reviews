# from sklearn.feature_extraction.text import TfidVectorizer


# def vectorize(X_train['review_text']):
# 	vectorizer = TfidVectorizer()
# 	vectorized_review = vectorizer.fit_transform(X_train['review_text'])
# 	return vectorized_review

# vectorize()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np

# data = """The cat and her kittens
# They put on their mittens,
# To eat a Christmas pie.
# The poor little kittens
# They lost their mittens,
# And then they began to cry.
# O mother dear, we sadly fear
# We cannot go to-day,
# For we have lost our mittens."
# "If it be so, ye shall not go,
# For ye are naughty kittens."""

data = 'review.txt'
def dataset_preparation(data):

	# basic cleanup
	corpus = data.lower().split("\n")

	# tokenization	
	tokenizer.fit_on_texts(corpus)
	total_words = len(tokenizer.word_index) + 1

	# create input sequences using list of tokens
	input_sequences = []
	for line in corpus:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(1, len(token_list)):
			n_gram_sequence = token_list[:i+1]
			input_sequences.append(n_gram_sequence)

	# pad sequences 
	max_sequence_len = max([len(x) for x in input_sequences])
	input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

	# create predictors and label
	predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
	label = ku.to_categorical(label, num_classes=total_words)

	return predictors, label, max_sequence_len, total_words

tokenizer = Tokenizer()
# def data_prepare(data):
# 	tokenizer.fit_on_texts(data)
# 	number_of_words = len(tokenizer.word_index) + 1
# 	input_sequences = []

# 	for  line in data:
# 		token_list = tokenizer.texts_to_sequences([data])[0]
# 		for i in range(1, len(token_list)):
# 			n_gram_sequence = token_list[:i+1]
# 			input_sequences.append(n_gram_sequence)
#     max_sequence_len = max([len(x) for x in input_sequences])
#     input_sequences = np.array(pad_sequences(input_sequences,maxlen=max_sequence_len, padding='pre'))
#     predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
#     label = ku.to_categorical(label, num_classes=total_words)
#     return predictors, label, max_sequence_len, total_words



def create_model(predictors, label, max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=input_len))
    model.add(LSTM(150))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(predictors, label, epochs=100, verbose=1)
    return model

def generate_text(seed_text, next_words, max_sequence_len):
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		predicted = model.predict_classes(token_list, verbose=0)
		
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	return seed_text



X, Y, max_len, total_words = dataset_preparation(data)
model = create_model(X, Y, max_len, total_words)



text = generate_text("cat and", 3, max_len)
print(text)
texty = generate_text("we naughty", 3, max_len)
print(texty)
