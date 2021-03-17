import tensorflow as tf

train_dir = "./data/train"
test_dir = "./data/test"
# Keep track of which index maps to which written language
# Dragon language doesn't have much to work with, so it is omitted until a larger dataset can be gathered
language_map = ['english', 'spanish', 'latin']

def loadData():
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(train_dir, class_names=language_map)
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(test_dir, class_names=language_map)
    
##    for text_batch, label_batch in raw_train_ds.take(1):
##      for i in range(10):
##        print("Question: ", text_batch.numpy()[i][:20])
##        print("Label:", label_batch.numpy()[i])

    train_len = len(raw_train_ds)
    test_len = len(raw_test_ds)

    return raw_train_ds.shuffle(train_len), raw_test_ds.shuffle(test_len)
