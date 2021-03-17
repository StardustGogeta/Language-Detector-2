import tensorflow as tf
import random

data_dir = "./data"
# Keep track of which index maps to which written language
# Dragon language doesn't have much to work with, so it is omitted until a larger dataset can be gathered
language_map = ['english', 'spanish', 'latin']

seed = random.randint(0, 255)
print(f"Using seed {seed} for input data randomization...")

def loadData():
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        data_dir,
        class_names=language_map,
        seed=seed,
        validation_split=0.2,
        subset='training')
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        data_dir,
        class_names=language_map,
        seed=seed,
        validation_split=0.2,
        subset='validation')
    
##    for text_batch, label_batch in raw_train_ds.take(1):
##      for i in range(10):
##        print("Question: ", text_batch.numpy()[i][:20])
##        print("Label:", label_batch.numpy()[i])

    train_len = len(raw_train_ds)
    test_len = len(raw_test_ds)

    return raw_train_ds.shuffle(train_len), raw_test_ds.shuffle(test_len)
