# Adapted from https://www.tensorflow.org/tutorials/text/text_classification_rnn
# Also uses https://www.tensorflow.org/guide/keras/rnn

import tensorflow as tf
import matplotlib.pyplot as plt
import trainingData
import numpy as np
import copy, random

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

VOCAB_SIZE = 10000
#MAX_SEQUENCE_LENGTH = 1000
NUM_LANGUAGES = len(trainingData.language_map)

# Import samples of (text, language), where language is an integer
# corresponding to a certain written language.
train_dataset, test_dataset = trainingData.loadData()

# Create text vectorization layer
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE,
    #output_sequence_length=MAX_SEQUENCE_LENGTH
)
# Set the vocabulary of the encoder
encoder.adapt(train_dataset.map(lambda text, label: text))

##vocab = np.array(encoder.get_vocabulary())
##print(vocab[:40])

# nnet = tf.keras.models.load_model("./models/my_nnet")

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
##    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64)),
##    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    # Softmax generates probability distribution
    tf.keras.layers.Dense(NUM_LANGUAGES, activation='softmax')
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-3)
metrics = ['accuracy']
model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)

# 30 to 50 epochs seems to be enough to fit without causing extreme overfitting
history = model.fit(train_dataset, epochs=50, validation_data=test_dataset, verbose=2)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# model.summary()

plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plot_graphs(history, 'accuracy')
plt.ylim(None,1)
plt.axhline(y=1/NUM_LANGUAGES, color='r', linestyle='-')
plt.subplot(1,2,2)
plot_graphs(history, 'loss')
plt.ylim(0,None)
plt.show()

print()
while True:
  text = input("Enter text to guess the language of:\n")
  if text == "*save":
    name = input("Enter a name for the neural net save file: ")
    model.save("./models/" + name)
    print("Done saving.")
  out = model.predict([text])[0]
  out = [(trainingData.language_map[i].capitalize(), e) for i, e in enumerate(out)]
  # print(out)
  out.sort(key=lambda v: -v[1]) # Sort by probability, descending
  print("Language predictions:")
  for lang, prob in out:
    print(f"{lang+':':<10}\t{100*prob:.3f}%")
  print()
