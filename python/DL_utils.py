import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras import Model
import numpy as np

## Encoders
class MLPenc(Model):
  def __init__(self, latent_dim=4, name='enc'):
    super(MLPenc, self).__init__(name=name)
    self.dense1 = Dense(24, activation='relu')
    self.bn1 = BatchNormalization()
    self.dense2 = Dense(12, activation='relu')
    self.bn2 = BatchNormalization()
    self.dense3 = Dense(8, activation='relu')
    self.bn3 = BatchNormalization()
    self.latent = Dense(latent_dim, activity_regularizer=tf.keras.regularizers.l1(10e-5))
    self.bn4 = BatchNormalization()

  def call(self, x):
    x = self.dense1(x)
    x = self.bn1(x)
    x = self.dense2(x)
    x = self.bn2(x)
    x = self.dense3(x)
    x = self.bn3(x)
    x = self.latent(x)
    return self.bn4(x)

class MLPenc_beta(Model):
  def __init__(self, latent_dim=4, name='enc'):
    super(MLPenc_beta, self).__init__(name=name)
    self.dense1 = Dense(246, activation='relu')
    self.bn1 = BatchNormalization()
    self.dense2 = Dense(128, activation='relu')
    self.bn2 = BatchNormalization()
    self.dense3 = Dense(16, activation='relu')
    self.bn3 = BatchNormalization()
    self.latent = Dense(latent_dim, activity_regularizer=tf.keras.regularizers.l1(10e-5))
    self.bn4 = BatchNormalization()

  def call(self, x):
    x = self.dense1(x)
    x = self.bn1(x)
    x = self.dense2(x)
    x = self.bn2(x)
    x = self.dense3(x)
    x = self.bn3(x)
    x = self.latent(x)
    return self.bn4(x)

class CNNenc(Model):
  def __init__(self, latent_dim=4, name='enc'):
    super(CNNenc, self).__init__(name=name)
    self.conv1 = Conv2D(32,(3,2), activation='relu', strides=1, padding="same")
    self.bn1 = BatchNormalization()
    self.conv2 = Conv2D(32,3, activation='relu', strides=2, padding="same")
    self.bn2 = BatchNormalization()
    self.flatten = Flatten()
    self.dense1 = Dense(16, activation='relu')
    self.bn3 = BatchNormalization()
    self.latent = Dense(latent_dim, activity_regularizer=tf.keras.regularizers.l1(10e-5))
    self.bn4 = BatchNormalization()

  def call(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.bn3(x)
    x = self.latent(x)
    return self.bn4(x)

## Classifier
class CLF(Model):
  def __init__(self, n_class=7, name='clf'):
    super(CLF, self).__init__(name=name)
    self.dense1 = Dense(n_class, activation='softmax')

  def call(self, x):
    return self.dense1(x)

## Full models
class MLP(Model):
  def __init__(self, n_class=7):
    super(MLP, self).__init__()
    self.enc = MLPenc()
    self.clf = CLF(n_class)
  
  def call(self, x):
    x = self.enc(x)
    return self.clf(x)

class MLPbeta(Model):
  def __init__(self, n_class=7):
    super(MLPbeta, self).__init__()
    self.enc = MLPenc_beta()
    self.clf = CLF(n_class)
  
  def call(self, x):
    x = self.enc(x)
    return self.clf(x)
  
class CNN(Model):
  def __init__(self, n_class=7):
    super(CNN, self).__init__()
    self.enc = CNNenc()
    self.clf = CLF(n_class)
  
  def call(self, x):
    x = self.enc(x)
    return self.clf(x)

def eval_nn(x, y, mod, clean):
    y_pred = np.argmax(mod(x).numpy(),axis=1)
    clean_acc = np.sum(y_pred[:clean] == np.argmax(y[:clean,...],axis=1))/y_pred[:clean].size
    noise_acc = np.sum(y_pred[clean:] == np.argmax(y[clean:,...],axis=1))/y_pred[clean:].size
    return clean_acc, noise_acc

## TRAIN TEST MLP
@tf.function
def train_mlp(x, y, mlp, loss_fn, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        y_out = mlp(x)
        loss = loss_fn(y, y_out)
    gradients = tape.gradient(loss, mlp.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mlp.trainable_variables))

    train_loss(loss)
    train_accuracy(y, y_out)

@tf.function
def test_mlp(x, y, mlp, loss_fn, test_loss, test_accuracy):
    y_out = mlp(x)
    t_loss = loss_fn(y, y_out)

    test_loss(t_loss)
    test_accuracy(y, y_out)

@tf.function
def train_mlpbeta(x, y, mlp_beta, loss_fn, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        y_out = mlp_beta(x)
        loss = loss_fn(y, y_out)
    gradients = tape.gradient(loss, mlp_beta.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mlp_beta.trainable_variables))

    train_loss(loss)
    train_accuracy(y, y_out)

@tf.function
def test_mlpbeta(x, y, mlp_beta, loss_fn, test_loss, test_accuracy):
    y_out = mlp_beta(x)
    t_loss = loss_fn(y, y_out)

    test_loss(t_loss)
    test_accuracy(y, y_out)

@tf.function
def train_cnn(x, y, cnn, loss_fn, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        y_out = cnn(x)
        loss = loss_fn(y, y_out)
    gradients = tape.gradient(loss, cnn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, cnn.trainable_variables))

    train_loss(loss)
    train_accuracy(y, y_out)

@tf.function
def test_cnn(x, y, cnn, loss_fn, test_loss, test_accuracy):
    y_out = cnn(x)
    t_loss = loss_fn(y, y_out)

    test_loss(t_loss)
    test_accuracy(y, y_out)