import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras import Model

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
  def __init__(self):
    super(MLP, self).__init__()
    self.enc = MLPenc()
    self.clf = CLF()
  
  def call(self, x):
    x = self.enc(x)
    return self.clf(x)

class MLPbeta(Model):
  def __init__(self):
    super(MLPbeta, self).__init__()
    self.enc = MLPenc_beta()
    self.clf = CLF()
  
  def call(self, x):
    x = self.enc(x)
    return self.clf(x)
  
class CNN(Model):
  def __init__(self):
    super(CNN, self).__init__()
    self.enc = CNNenc()
    self.clf = CLF()
  
  def call(self, x):
    x = self.enc(x)
    return self.clf(x)