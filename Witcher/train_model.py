import numpy as np
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from alexnet import alexnet

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCH = 8

MODEL_NAME = 'witcher3-horse-{}-{}-{}-epochs.model'.format(LR, 'alexnet', EPOCH)


count = 0
train_data = np.load('training_data.npy', allow_pickle=True)

##for i,j in train_data:
##    if j == [1, 0, 1]:
##        count += 1
##print(count)

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = np.array([i[1] for i in test])

checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

callbacks_list = [checkpoint, tensorboard]

model = alexnet(WIDTH, HEIGHT, LR)
model.fit(X, Y, epochs=EPOCH, validation_split=0.15, callbacks=callbacks_list, verbose=1)

##print(len(X), len(Y))


model.save(MODEL_NAME)

#tensorboard --logdir path_to_current_dir/Graph
#tensorboard --logdir ./Graph


