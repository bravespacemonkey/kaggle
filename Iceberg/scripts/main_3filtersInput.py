import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, GlobalMaxPooling2D, Dense

base_path = os.path.join('..', 'input')


def load_and_format(in_path):
    out_df = pd.read_json(in_path)
    out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'], c_row['band_2'], np.add(c_row['band_1'], c_row['band_2'])/2], -1).reshape((75, 75, 3))], 1)
    out_images = np.stack(out_images).squeeze()
    return out_df, out_images


train_df, train_images = load_and_format(os.path.join(base_path, 'train.json'))
print('training', train_df.shape, 'loaded', train_images.shape)
test_df, test_images = load_and_format(os.path.join(base_path, 'test.json'))
print('testing', test_df.shape, 'loaded', test_images.shape)
train_df.sample(3)


X_train, X_test, y_train, y_test = train_test_split(train_images,
                                                    to_categorical(train_df['is_iceberg']),
                                                    random_state=2017,
                                                    test_size=0.2
                                                    )
print('Train', X_train.shape, y_train.shape)
print('Validation', X_test.shape, y_test.shape)


simple_cnn = Sequential()
simple_cnn.add(BatchNormalization(input_shape=(75, 75, 3)))
simple_cnn.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding="same"))
simple_cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same"))
simple_cnn.add(MaxPooling2D((2, 2), padding="same"))
simple_cnn.add(Dropout(0.45))
simple_cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"))
simple_cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"))
simple_cnn.add(MaxPooling2D((2, 2), padding="same"))
simple_cnn.add(Dropout(0.45))
simple_cnn.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))
simple_cnn.add(MaxPooling2D((3, 3), padding="same"))
simple_cnn.add(Dropout(0.2))
simple_cnn.add(GlobalMaxPooling2D())
simple_cnn.add(Dropout(0.45))
simple_cnn.add(Dense(8, activation='relu'))
simple_cnn.add(Dense(2, activation='softmax'))
simple_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
simple_cnn.summary()

weights_file = os.path.join(base_path, "best_val_weight_3filters")
pred_file = os.path.join(base_path, "best_val_weight_3filters_predictions.csv")

callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto'),
    ModelCheckpoint(weights_file, monitor='val_loss', save_best_only=True, verbose=1)
]

history = simple_cnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, shuffle=True, callbacks=callbacks)

simple_cnn.load_weights(weights_file)

test_predictions = simple_cnn.predict(test_images)

pred_df = test_df[['id']].copy()
pred_df['is_iceberg'] = test_predictions[:, 1]
pred_df.to_csv(pred_file, index=False)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()