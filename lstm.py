from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import keras.utils as cate
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential


data = """The cat and her kittens
They put on their mittens,
To eat a Christmas pie.
The poor little kittens
They lost their mittens,
And then they began to cry.
O mother dear, we sadly fear
We cannot go to-day,
For we have lost our mittens."
"If it be so, ye shall not go,
For ye are naughty kittens."""


tokenizer = Tokenizer()

def preprocess(data):
	
	data = data.split('\n') #split each line
	tokenizer.fit_on_texts(data)
	word_lenght = len(tokenizer.word_index) + 1
	sequences = []
	for line in data:
		token_list = tokenizer.texts_to_sequences([line])[0]

		for i in range(1, len(token_list)):
			n_gram_sequence = token_list[:i+1]
			sequences.append(n_gram_sequence)

	max_sequence_len = max([len(x) for x in sequences])
	input_sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))


	predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
	label = cate.to_categorical(label, num_classes=word_lenght)

	return predictors, label, max_sequence_len, word_lenght

	# return predictors

def create_model(predictors, label, max_sequence_len, word_lenght):
    input_len = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(word_lenght, 10, input_length=input_len))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(word_lenght, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(predictors, label, epochs=10, verbose=1)
    return model

# print(preprocess(data))
# print(create_model)
# print(dataset_preparation(data))

predictors, label, max_sequence_len, word_lenght = preprocess(data)
model = create_model(predictors, label, max_sequence_len, word_lenght)


#text generation
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


text = generate_text("cat and", 3, max_sequence_len)
print(text)

