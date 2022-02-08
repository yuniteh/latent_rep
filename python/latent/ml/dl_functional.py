from matplotlib import pyplot as plt
import os
import numpy as np

from tensorflow.keras.layers import Lambda, Input, Dense, Conv2D, Flatten, Conv2DTranspose, Reshape, concatenate, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy, categorical_crossentropy
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import tensorflow as tf
import time

## SUPERVISED VARIATIONAL AUTOENCODER (NER model)
def build_svae_manual(latent_dim, n_class, input_type='feat', sparse='True',lr=0.001):
    if input_type == 'feat':
        input_shape = (6,4,1)
        inter_shape = (3,2,1)
    elif input_type == 'raw':
        input_shape = (6,50,1)
        inter_shape = (3,25,1)

    weight = Input(shape=(2,))
    # build encoder model
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3,2), activation="relu", strides=1, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = BatchNormalization()(x)

    if sparse:
        z_mean = Dense(latent_dim, name="z_mean", activity_regularizer=regularizers.l1(10e-5))(x)
        z_log_var = Dense(latent_dim, name="z_log_var", activity_regularizer=regularizers.l1(10e-5))(x)
        clf_input = Dense(latent_dim, name="clf_input", activity_regularizer=regularizers.l1(10e-5))(x)
    else:
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)        
        clf_input = Dense(latent_dim, name="clf_input")(x)

    z_mean = BatchNormalization()(z_mean)
    z_log_var = BatchNormalization()(z_log_var)
    clf_input = BatchNormalization()(clf_input)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z, clf_input], name="encoder")
    # encoder.summary()

    # New: add a linear classifier
    clf_latent_inputs = Input(shape=(latent_dim,), name='z_sampling_clf')
    clf_outputs = Dense(n_class, activation='softmax', name='class_output')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')   
    # clf_supervised.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,))
    clf_in = Input(shape=(1,))
    # clf_in = Input(shape=(n_class,))
    clf_dec = Dense(latent_dim, activation="relu")(clf_in)
    cat_inputs = concatenate([latent_inputs, clf_dec],axis=-1)
    x = Dense(inter_shape[0]*inter_shape[1]*32, activation="relu")(cat_inputs)
    x = BatchNormalization()(x)
    x = Reshape((inter_shape[0], inter_shape[1], 32))(x)
    x = Conv2DTranspose(32, (3,3), activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    # x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(1, 3, activation="relu", padding="same")(x)
    decoder = Model([latent_inputs,clf_in], decoder_outputs, name="decoder")
    # decoder.summary()

    # instantiate VAE model
    outputs = [decoder([encoder(inputs)[2],tf.expand_dims(tf.math.argmax(clf_supervised(encoder(inputs)[3]),axis=1),axis=1)]), clf_supervised(encoder(inputs)[3]), encoder(inputs)[2]]
    # outputs = [decoder([encoder(inputs)[2],clf_supervised(encoder(inputs)[3])]), clf_supervised(encoder(inputs)[3]), encoder(inputs)[2]]
    # outputs = [decoder(encoder(inputs)[2]), clf_supervised(encoder(inputs)[2])]
    vae = Model([inputs,weight], outputs, name='vae_mlp')

    def VAE_loss(x_origin,x_out):
        # x_origin=K.flatten(x_origin)
        # x_out=K.flatten(x_out)
        # reconstruction_loss = input_shape[0]*input_shape[1] * binary_crossentropy(x_origin, x_out)
        reconstruction_loss = K.mean(mse(x_origin, x_out))
        # reconstruction_loss *= input_shape[0] * input_shape[1]

        vae_loss = 10*reconstruction_loss
        return vae_loss
    
    def kloss(x_origin,x_out):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        kl_loss = weight[1]*K.mean(kl_loss)
        return kl_loss

    opt = optimizers.Adam(learning_rate=lr)
    vae.compile(optimizer=opt, loss=[VAE_loss, 'categorical_crossentropy', kloss],experimental_run_tf_function=False,metrics=['accuracy'])
    return vae, encoder, decoder, clf_supervised

## SUPERVISED VARIATIONAL AUTOENCODER (NER model)
def build_M2(latent_dim, n_class, input_type='feat', sparse='True',lr=0.001):
    if input_type == 'feat':
        input_shape = (6,4,1)
        inter_shape = (3,2,1)
    elif input_type == 'raw':
        input_shape = (6,50,1)
        inter_shape = (3,25,1)
    elif input_type == 'tdar':
        input_shape = (6,10,1)
        inter_shape = (3,5,1)

    weight = Input(shape=(2,))
    # build encoder model
    x_in = Input(shape=input_shape)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x_in)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = BatchNormalization()(x)

    if sparse:
        z_mean = Dense(latent_dim, name="z_mean", activity_regularizer=regularizers.l1(10e-5))(x)
        z_log_var = Dense(latent_dim, name="z_log_var", activity_regularizer=regularizers.l1(10e-5),kernel_initializer='zeros',bias_initializer='zeros')(x)
    else:
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)        

    z_mean = BatchNormalization()(z_mean)
    z_log_var = BatchNormalization()(z_log_var)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(x_in, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()

    # New: add a linear classifier
    # x_in_clf = Input(shape=input_shape)
    x_clf = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x_in)
    x_clf = BatchNormalization()(x_clf)
    x_clf = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x_clf)
    x_clf = BatchNormalization()(x_clf)
    x_clf = Flatten()(x)
    x_clf = Dense(16, activation="relu")(x_clf)
    x_clf = BatchNormalization()(x_clf)
    x_clf = Dense(latent_dim)(x_clf)
    x_lat = BatchNormalization()(x_clf)
    # clf_latent_inputs = Input(shape=(latent_dim,), name='z_sampling_clf')
    clf_outputs = Dense(n_class, activation='softmax', name='class_output')(x_lat)
    clf_supervised = Model(x_in, [clf_outputs, x_lat], name='clf')   
    # clf_supervised.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,),name="latent_in")
    # clf_in = Input(shape=(1,))
    clf_in = Input(shape=(n_class,),name="clf_in")
    clf_dec = Dense(latent_dim, activation="relu")(clf_in)
    cat_inputs = concatenate([latent_inputs, clf_dec],axis=-1)
    x = Dense(inter_shape[0]*inter_shape[1]*32, activation="relu")(cat_inputs)
    x = BatchNormalization()(x)
    x = Reshape((inter_shape[0], inter_shape[1], 32))(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    # x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(1, 3, activation="relu", padding="same")(x)
    decoder = Model([latent_inputs,clf_in], decoder_outputs, name="decoder")
    # decoder.summary()

    # instantiate VAE model
    vae_clf = Input(shape=(n_class,),name="vae_clf")
    outputs = [decoder([encoder(x_in)[2],vae_clf]), clf_supervised(x_in)[0], clf_supervised(x_in)[1]]
    # outputs = [decoder([encoder(x_in)[2],clf_supervised(encoder(x_in)[3])]), clf_supervised(encoder(x_in)[3]), encoder(x_in)[2]]
    # outputs = [decoder(encoder(inputs)[2]), clf_supervised(encoder(inputs)[2])]
    vae = Model([x_in,vae_clf,weight], outputs, name='vae_mlp')

    def VAE_loss(x_origin,x_out):
        # x_origin=K.flatten(x_origin)
        # x_out=K.flatten(x_out)
        # reconstruction_loss = input_shape[0]*input_shape[1] * binary_crossentropy(x_origin, x_out)
        reconstruction_loss = K.mean(mse(x_origin, x_out))
        # reconstruction_loss *= input_shape[0] * input_shape[1]

        vae_loss = reconstruction_loss
        return vae_loss
    
    def kloss(x_origin,x_out):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        kl_loss = weight[1]*K.mean(kl_loss)
        return kl_loss

    opt = optimizers.Adam(learning_rate=lr)
    vae.compile(optimizer=opt, loss=['mean_squared_error', 'categorical_crossentropy', kloss],experimental_run_tf_function=False,metrics=['accuracy'])
    return vae, encoder, decoder, clf_supervised

## SUPERVISED VARIATIONAL AUTOENCODER (NER model)
def build_M2S2_manual(latent_dim, n_class, input_type='feat', sparse='True',lr=0.001):
    if input_type == 'feat':
        input_shape = (6,4,1)
        inter_shape = (3,2,1)
    elif input_type == 'raw':
        input_shape = (6,50,1)
        inter_shape = (3,25,1)

    weight = Input(shape=(2,))
    # build encoder model
    x_in = Input(shape=input_shape)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x_in)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = BatchNormalization()(x)

    if sparse:
        z_mean = Dense(latent_dim, name="z_mean", activity_regularizer=regularizers.l1(10e-5))(x)
        z_log_var = Dense(latent_dim, name="z_log_var", activity_regularizer=regularizers.l1(10e-5),kernel_initializer='zeros',bias_initializer='zeros')(x)
    else:
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)        

    z_mean = BatchNormalization()(z_mean)
    z_log_var = BatchNormalization()(z_log_var)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    clf_lat = concatenate([z_mean, z_log_var], axis=-1)
    # clf_in = Dense(latent_dim, activation="relu")(z_all)
    encoder = Model(x_in, [z_mean, z_log_var, z, clf_lat], name="encoder")
    # encoder.summary()

    # New: add a linear classifier
    # x_in_clf = Input(shape=input_shape)
    clf_latent_inputs = Input(shape=(latent_dim*2,), name='z_sampling_clf')
    clf_outputs = Dense(n_class, activation='softmax', name='class_output')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')   
    # clf_supervised.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,),name="latent_in")
    # clf_in = Input(shape=(1,))
    clf_in = Input(shape=(n_class,),name="clf_in")
    clf_dec = Dense(latent_dim, activation="relu")(clf_in)
    cat_inputs = concatenate([latent_inputs, clf_dec],axis=-1)
    x = Dense(inter_shape[0]*inter_shape[1]*32, activation="relu")(cat_inputs)
    x = BatchNormalization()(x)
    x = Reshape((inter_shape[0], inter_shape[1], 32))(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    # x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(1, 3, activation="relu", padding="same")(x)
    decoder = Model([latent_inputs,clf_in], decoder_outputs, name="decoder")
    # decoder.summary()

    # instantiate VAE model
    vae_clf = Input(shape=(n_class,),name="vae_clf")
    outputs = [decoder([encoder(x_in)[2],vae_clf]), clf_supervised(encoder(x_in)[3])]
    # outputs = [decoder([encoder(x_in)[2],clf_supervised(encoder(x_in)[3])]), clf_supervised(encoder(x_in)[3]), encoder(x_in)[2]]
    # outputs = [decoder(encoder(inputs)[2]), clf_supervised(encoder(inputs)[2])]
    vae = Model([x_in,vae_clf,weight], outputs, name='vae_mlp')

    def VAE_loss(x_origin,x_out):
        reconstruction_loss = K.mean(input_shape[0]*input_shape[1]*mse(x_origin,x_out))
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = reconstruction_loss + K.mean(kl_loss)/100
        return vae_loss

    opt = optimizers.Adam(learning_rate=lr)
    vae.compile(optimizer=opt, loss=[VAE_loss, 'categorical_crossentropy'], experimental_run_tf_function=False,metrics=['accuracy'])
    return vae, encoder, decoder, clf_supervised

def build_M2S2(latent_dim, n_class, input_type='feat', sparse='True',lr=0.001):
    if input_type == 'feat':
        input_shape = (6,4,1)
        inter_shape = (3,2,1)
    elif input_type == 'raw':
        input_shape = (6,50,1)
        inter_shape = (3,25,1)    
    elif input_type == 'tdar':
        input_shape = (6,10,1)
        inter_shape = (3,5,1)    

    weight = Input(shape=(2,))
    # build encoder model
    x_in = Input(shape=input_shape)
    x = Conv2D(32, (3,2), activation="relu", strides=1, padding="same")(x_in)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = BatchNormalization()(x)

    if sparse:
        z_mean = Dense(latent_dim, name="z_mean", activity_regularizer=regularizers.l1(10e-5),activation="relu")(x)
        z_log_var = Dense(latent_dim, name="z_log_var", activity_regularizer=regularizers.l1(10e-5),kernel_initializer='zeros',bias_initializer='zeros',activation="relu")(x)
    else:
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)        

    z_mean = BatchNormalization()(z_mean)
    z_log_var = BatchNormalization()(z_log_var)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    clf_lat = concatenate([z_mean, z_log_var], axis=-1)
    # clf_in = Dense(latent_dim, activation="relu")(z_all)
    encoder = Model(x_in, [z_mean, z_log_var, z, clf_lat], name="encoder")
    # encoder.summary()

    # New: add a linear classifier
    # x_in_clf = Input(shape=input_shape)
    clf_latent_inputs = Input(shape=(latent_dim*2,), name='z_sampling_clf')
    clf_outputs = Dense(n_class, activation='softmax', name='class_output')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')   
    # clf_supervised.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,),name="latent_in")
    # clf_in = Input(shape=(1,))
    clf_in = Input(shape=(n_class,),name="clf_in")
    clf_dec = Dense(latent_dim, activation="relu")(clf_in)
    cat_inputs = concatenate([latent_inputs, clf_dec],axis=-1)
    x = Dense(inter_shape[0]*inter_shape[1]*32, activation="relu")(cat_inputs)
    x = BatchNormalization()(x)
    x = Reshape((inter_shape[0], inter_shape[1], 32))(x)
    x = Conv2DTranspose(32, (3,2), activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    # x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(1, 3, activation="relu", padding="same")(x)
    decoder = Model([latent_inputs,clf_in], decoder_outputs, name="decoder")
    # decoder.summary()

    # instantiate VAE model
    vae_clf = Input(shape=(n_class,),name="vae_clf")
    outputs = [decoder([encoder(x_in)[2],vae_clf]), clf_supervised(encoder(x_in)[3])]
    # outputs = [decoder([encoder(x_in)[2],clf_supervised(encoder(x_in)[3])]), clf_supervised(encoder(x_in)[3]), encoder(x_in)[2]]
    # outputs = [decoder(encoder(inputs)[2]), clf_supervised(encoder(inputs)[2])]
    vae = Model([x_in,vae_clf], outputs, name='vae_mlp')

    def VAE_loss(x_origin,x_out):
        reconstruction_loss = K.mean(input_shape[0]*input_shape[1]*mse(x_origin,x_out))
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = reconstruction_loss/10 + K.mean(kl_loss)/100
        return vae_loss

    opt = optimizers.Adam(learning_rate=lr)
    vae.compile(optimizer=opt, loss=[VAE_loss, 'categorical_crossentropy'], experimental_run_tf_function=False,metrics=['accuracy'])
    return vae, encoder, decoder, clf_supervised

def build_M2S(latent_dim, n_class, input_type='feat', sparse='True',lr=0.001):
    if input_type == 'feat':
        input_shape = (6,4,1)
        inter_shape = (3,2,1)
    elif input_type == 'raw':
        input_shape = (6,50,1)
        inter_shape = (3,25,1)

    weight = Input(shape=(2,))
    # build encoder model
    x_in = Input(shape=input_shape)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x_in)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = BatchNormalization()(x)

    if sparse:
        z_mean = Dense(latent_dim, name="z_mean", activity_regularizer=regularizers.l1(10e-5))(x)
        z_log_var = Dense(latent_dim, name="z_log_var", activity_regularizer=regularizers.l1(10e-5),kernel_initializer='zeros',bias_initializer='zeros')(x)
    else:
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)        

    z_mean = BatchNormalization()(z_mean)
    z_log_var = BatchNormalization()(z_log_var)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(x_in, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()

    # New: add a linear classifier
    # x_in_clf = Input(shape=input_shape)
    clf_latent_inputs = Input(shape=(latent_dim,), name='z_sampling_clf')
    clf_outputs = Dense(n_class, activation='softmax', name='class_output')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')   
    # clf_supervised.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,),name="latent_in")
    # clf_in = Input(shape=(1,))
    clf_in = Input(shape=(n_class,),name="clf_in")
    clf_dec = Dense(latent_dim, activation="relu")(clf_in)
    cat_inputs = concatenate([latent_inputs, clf_dec],axis=-1)
    x = Dense(inter_shape[0]*inter_shape[1]*32, activation="relu")(cat_inputs)
    x = BatchNormalization()(x)
    x = Reshape((inter_shape[0], inter_shape[1], 32))(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    # x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(1, 3, activation="relu", padding="same")(x)
    decoder = Model([latent_inputs,clf_in], decoder_outputs, name="decoder")
    # decoder.summary()

    # instantiate VAE model
    vae_clf = Input(shape=(n_class,),name="vae_clf")
    outputs = [decoder([encoder(x_in)[2],vae_clf]), clf_supervised(encoder(x_in)[2]), encoder(x_in)[2]]
    # outputs = [decoder([encoder(x_in)[2],clf_supervised(encoder(x_in)[3])]), clf_supervised(encoder(x_in)[3]), encoder(x_in)[2]]
    # outputs = [decoder(encoder(inputs)[2]), clf_supervised(encoder(inputs)[2])]
    vae = Model([x_in,vae_clf], outputs, name='vae_mlp')

    def VAE_loss(x_origin,x_out):
        # x_origin=K.flatten(x_origin)
        # x_out=K.flatten(x_out)
        # reconstruction_loss = input_shape[0]*input_shape[1] * binary_crossentropy(x_origin, x_out)
        # reconstruction_loss = input_shape[0]*input_shape[1]*binary_crossentropy(K.flatten(x_origin), K.flatten(x_out))
        reconstruction_loss = K.mean(input_shape[0]*input_shape[1]*mse(x_origin,x_out))
        # reconstruction_loss = K.mean(reconstruction_loss)
        # reconstruction_loss *= input_shape[0] * input_shape[1]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # kl_loss = kl_loss
        # print(reconstruction_loss)
        # print(kl_loss)

        vae_loss = reconstruction_loss + K.mean(kl_loss)/100
        return vae_loss

    opt = optimizers.Adam(learning_rate=lr)
    vae.compile(optimizer=opt, loss=[VAE_loss, 'categorical_crossentropy'], experimental_run_tf_function=False,metrics=['accuracy'])
    return vae, encoder, decoder, clf_supervised

## SUPERVISED VARIATIONAL AUTOENCODER (NER model)
def build_M2S_manual(latent_dim, n_class, input_type='feat', sparse='True',lr=0.001):
    if input_type == 'feat':
        input_shape = (6,4,1)
        inter_shape = (3,2,1)
    elif input_type == 'raw':
        input_shape = (6,50,1)
        inter_shape = (3,25,1)

    weight = Input(shape=(2,))
    # build encoder model
    x_in = Input(shape=input_shape)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x_in)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = BatchNormalization()(x)

    if sparse:
        z_mean = Dense(latent_dim, name="z_mean", activity_regularizer=regularizers.l1(10e-5))(x)
        z_log_var = Dense(latent_dim, name="z_log_var", activity_regularizer=regularizers.l1(10e-5),kernel_initializer='zeros',bias_initializer='zeros')(x)
    else:
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)        

    z_mean = BatchNormalization()(z_mean)
    z_log_var = BatchNormalization()(z_log_var)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(x_in, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()

    # New: add a linear classifier
    # x_in_clf = Input(shape=input_shape)
    clf_latent_inputs = Input(shape=(latent_dim,), name='z_sampling_clf')
    clf_outputs = Dense(n_class, activation='softmax', name='class_output')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')   
    # clf_supervised.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,),name="latent_in")
    # clf_in = Input(shape=(1,))
    clf_in = Input(shape=(n_class,),name="clf_in")
    clf_dec = Dense(latent_dim, activation="relu")(clf_in)
    cat_inputs = concatenate([latent_inputs, clf_dec],axis=-1)
    x = Dense(inter_shape[0]*inter_shape[1]*32, activation="relu")(cat_inputs)
    x = BatchNormalization()(x)
    x = Reshape((inter_shape[0], inter_shape[1], 32))(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    # x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(1, 3, activation="relu", padding="same")(x)
    decoder = Model([latent_inputs,clf_in], decoder_outputs, name="decoder")
    # decoder.summary()

    # instantiate VAE model
    vae_clf = Input(shape=(n_class,),name="vae_clf")
    outputs = [decoder([encoder(x_in)[2],vae_clf]), clf_supervised(encoder(x_in)[2]), encoder(x_in)[2]]
    # outputs = [decoder([encoder(x_in)[2],clf_supervised(encoder(x_in)[3])]), clf_supervised(encoder(x_in)[3]), encoder(x_in)[2]]
    # outputs = [decoder(encoder(inputs)[2]), clf_supervised(encoder(inputs)[2])]
    vae = Model([x_in,vae_clf,weight], outputs, name='vae_mlp')

    def VAE_loss(x_origin,x_out):
        # x_origin=K.flatten(x_origin)
        # x_out=K.flatten(x_out)
        # reconstruction_loss = input_shape[0]*input_shape[1] * binary_crossentropy(x_origin, x_out)
        # reconstruction_loss = input_shape[0]*input_shape[1]*binary_crossentropy(K.flatten(x_origin), K.flatten(x_out))
        reconstruction_loss = K.mean(input_shape[0]*input_shape[1]*mse(x_origin,x_out))
        # reconstruction_loss = K.mean(reconstruction_loss)
        # reconstruction_loss *= input_shape[0] * input_shape[1]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # kl_loss = kl_loss
        # print(reconstruction_loss)
        # print(kl_loss)

        vae_loss = reconstruction_loss + K.mean(kl_loss)/100
        return vae_loss
    
    def kloss(x_origin,x_out):
        kl_loss2 = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss2 = K.sum(kl_loss2, axis=-1)
        kl_loss2 *= -0.5
        kl_loss2 = K.mean(kl_loss2)
        return kl_loss2

    opt = optimizers.Adam(learning_rate=lr)
    vae.compile(optimizer=opt, loss=[VAE_loss, 'categorical_crossentropy', kloss], loss_weights = [1,1,0], experimental_run_tf_function=False,metrics=['accuracy'])
    return vae, encoder, decoder, clf_supervised

## SUPERVISED VARIATIONAL AUTOENCODER (NER model)
def build_svae(latent_dim, n_class, input_type='feat', sparse='True',lr=0.001):
    if input_type == 'feat':
        input_shape = (6,4,1)
        inter_shape = (3,2,1)
    elif input_type == 'raw':
        input_shape = (6,50,1)
        inter_shape = (3,25,1)

    # build encoder model
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = BatchNormalization()(x)

    if sparse:
        z_mean = Dense(latent_dim, name="z_mean", activity_regularizer=regularizers.l1(10e-5))(x)
        z_log_var = Dense(latent_dim, name="z_log_var", activity_regularizer=regularizers.l1(10e-5))(x)
    else:
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        
    z_mean = BatchNormalization()(z_mean)
    z_log_var = BatchNormalization()(z_log_var)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(inter_shape[0]*inter_shape[1]*32, activation="relu")(latent_inputs)
    x = BatchNormalization()(x)
    x = Reshape((inter_shape[0], inter_shape[1], 32))(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    # x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(1, 3, activation="relu", padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    # decoder.summary()

    # New: add a linear classifier
    clf_latent_inputs = Input(shape=(latent_dim,), name='z_sampling_clf')
    clf_outputs = Dense(n_class, activation='softmax', name='class_output')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')
    # clf_supervised.summary()

    # instantiate VAE model
    outputs = [decoder(encoder(inputs)[2]), clf_supervised(encoder(inputs)[2])]
    vae = Model(inputs, outputs, name='vae_mlp')

    def VAE_loss(x_origin,x_out):
        # x_origin=K.flatten(x_origin)
        # x_out=K.flatten(x_out)
        # reconstruction_loss = input_shape[0]*input_shape[1] * binary_crossentropy(x_origin, x_out)
        reconstruction_loss = K.mean(mse(x_origin, x_out))
        # reconstruction_loss *= input_shape[0] * input_shape[1]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = (reconstruction_loss + K.mean(kl_loss))/100.0
        return vae_loss

    opt = optimizers.Adam(learning_rate=lr)
    vae.compile(optimizer=opt, loss=[VAE_loss,'categorical_crossentropy'],experimental_run_tf_function=False,metrics=['accuracy'])
    return vae, encoder, decoder, clf_supervised

# VCNN NEW
def build_vcnn_manual(latent_dim, n_class, input_type='feat', sparse='True',lr=0.001):
    if input_type == 'feat':
        input_shape = (6,4,1)
        inter_shape = (3,2,1)
    elif input_type == 'raw':
        input_shape = (6,50,1)
        inter_shape = (3,25,1)

    weight = Input(shape=(2,))
    # build encoder model
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = BatchNormalization()(x)

    if sparse:
        z_mean = Dense(latent_dim, name="z_mean", activity_regularizer=regularizers.l1(10e-5))(x)
        z_log_var = Dense(latent_dim, name="z_log_var", activity_regularizer=regularizers.l1(10e-5),kernel_initializer='zeros',bias_initializer='zeros')(x)
        # clf_in = Dense(latent_dim, name="clf_in", activity_regularizer=regularizers.l1(10e-5))(x)
    else:
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)   
        # clf_in = Dense(latent_dim, name="clf_in")(x)     

    z_mean = BatchNormalization()(z_mean)
    z_log_var = BatchNormalization()(z_log_var)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()

    # New: add a linear classifier
    # mean_in = Input(shape=(latent_dim,))
    # var_in = Input(shape=(latent_dim,))
    # clf_latent_inputs = concatenate([mean_in, var_in],axis=-1)
    clf_latent_inputs = Input(shape=(latent_dim,), name='z_sampling_clf')
    clf_outputs = Dense(n_class, activation='softmax', name='class_output')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')   
    # clf_supervised.summary()

    # instantiate VAE model
    # outputs = [clf_supervised([encoder(inputs)[0],encoder(inputs)[1]]), clf_supervised([encoder(inputs)[0],encoder(inputs)[1]])]
    outputs = clf_supervised(encoder(inputs)[2])
    vae = Model([inputs,weight], outputs, name='vae_mlp')
    
    def kloss(x_origin,x_out):
        class_loss = K.mean(categorical_crossentropy(x_origin,x_out))
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        kl_loss = weight[1]*K.mean(kl_loss)
        # kl_loss = K.mean(kl_loss)
        return class_loss + kl_loss

    opt = optimizers.Adam(learning_rate=lr)
    vae.compile(optimizer=opt, loss=kloss,experimental_run_tf_function=False,metrics=['accuracy'])
    return vae, encoder, clf_supervised

## VARIATIONAL LATENT SPACE CLASSIFIER - NO DECODER
def build_vcnn(latent_dim, n_class, input_type='feat',sparse='True',lr=0.001):
    
    if input_type == 'feat':
        input_shape = (6,4,1)
    elif input_type == 'raw':
        input_shape = (6,50,1)
    elif input_type == 'tdar':
        input_shape = (6,10,1)

    # build encoder model
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3,2), activation="relu", strides=1, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = BatchNormalization()(x)

    if sparse:
        z_mean = Dense(latent_dim, name="z_mean", activity_regularizer=regularizers.l1(10e-5))(x)
        z_log_var = Dense(latent_dim, name="z_log_var", activity_regularizer=regularizers.l1(10e-5))(x)
    else:
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        
    z_mean = BatchNormalization()(z_mean)
    z_log_var = BatchNormalization()(z_log_var)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

    # classifier
    clf_latent_inputs = Input(shape=(latent_dim,), name='z_sampling_clf')
    clf_outputs = Dense(n_class, activation='softmax', name='class_output')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')

    # instantiate VAE model
    outputs = clf_supervised(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    def VAE_loss(x_origin, x_out):
        # x_origin=K.flatten(x_origin)
        # x_out=K.flatten(x_out)
        # xent_loss = input_shape[0]*input_shape[1] * binary_crossentropy(x_origin, x_out)
        class_loss = input_shape[0]*input_shape[1]*categorical_crossentropy(x_origin, x_out)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = class_loss + (K.mean(kl_loss)/100)
        # vae_loss = kl_loss
        return vae_loss

    opt = optimizers.Adam(learning_rate=lr)
    vae.compile(optimizer=opt, loss=VAE_loss,experimental_run_tf_function=False,metrics=['accuracy'])
    return vae, encoder, clf_supervised

def build_cnn(latent_dim, n_class, input_type='feat',sparse='True',lr=0.001):
    
    if input_type == 'feat':
        input_shape = (6,4,1)
    elif input_type == 'raw':
        input_shape = (6,50,1)
    elif input_type == 'tdar':
        input_shape = (6,10,1)

    # build encoder model
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3,2), activation="relu", strides=1, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = BatchNormalization()(x)
    if sparse:
        z = Dense(latent_dim, name="z", activity_regularizer=regularizers.l1(10e-5))(x)
    else:
        z = Dense(latent_dim, name="z")(x)
    z = BatchNormalization()(z)
    encoder = Model(inputs, z, name="encoder")

    # classifier
    clf_latent_inputs = Input(shape=(latent_dim,), name='z_clf')
    clf_outputs = Dense(n_class, activation='softmax', name='class_output')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')

    # instantiate VAE model
    outputs = clf_supervised(encoder(inputs))
    vae = Model(inputs, outputs, name='vae_mlp')

    opt = optimizers.Adam(learning_rate=lr)
    vae.compile(optimizer=opt, loss='categorical_crossentropy',experimental_run_tf_function=False,metrics=['accuracy'])
    return vae, encoder, clf_supervised

def build_cnn_ext(latent_dim, n_class, input_type='feat',sparse='True',lr=0.001):
    
    if input_type == 'feat':
        input_shape = (8,4,1)
    elif input_type == 'raw':
        input_shape = (8,50,1)

    # build encoder model
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = BatchNormalization()(x)
    if sparse:
        z = Dense(latent_dim, name="z", activity_regularizer=regularizers.l1(10e-5))(x)
    else:
        z = Dense(latent_dim, name="z")(x)
    z = BatchNormalization()(z)
    encoder = Model(inputs, z, name="encoder")

    # classifier
    clf_latent_inputs = Input(shape=(latent_dim,), name='z_clf')
    clf_outputs = Dense(n_class, activation='softmax', name='class_output')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')

    # instantiate VAE model
    outputs = clf_supervised(encoder(inputs))
    vae = Model(inputs, outputs, name='vae_mlp')

    opt = optimizers.Adam(learning_rate=lr)
    vae.compile(optimizer=opt, loss='categorical_crossentropy',experimental_run_tf_function=False,metrics=['accuracy'])
    return vae, encoder, clf_supervised

def build_cnn_old(latent_dim, n_class, input_type='feat',sparse='True'):
    
    if input_type == 'feat':
        input_shape = (6,4,1)
    elif input_type == 'raw':
        input_shape = (6,50,1)

    # build encoder model
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(inputs)
    x = MaxPooling2D((1,3))(x)
    z = Flatten()(x)
    encoder = Model(inputs, z, name="encoder")

    latent_dim = 3072
    # classifier
    clf_latent_inputs = Input(shape=(latent_dim,), name='z_clf')
    clf_outputs = Dense(n_class, activation='softmax', name='class_output')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')

    # instantiate VAE model
    outputs = clf_supervised(encoder(inputs))
    vae = Model(inputs, outputs, name='vae_mlp')

    vae.compile(optimizer='adam', loss='categorical_crossentropy',experimental_run_tf_function=False,metrics=['accuracy'])
    return vae, encoder, clf_supervised

## LATENT SPACE CLASSIFIER - NO DECODER
def build_sae(latent_dim, n_class, input_type='feat', sparse='True',lr=0.001):
    
    if input_type == 'feat':
        input_shape = (24,)
    elif input_type == 'raw':
        input_shape = (300,)
    elif input_type == 'mav':
        input_shape = (6,)
    elif input_type == 'tdar':
        input_shape = (60,)

    # build encoder model
    inputs = Input(shape=input_shape)
    x = Dense(24, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dense(12, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(8, activation="relu")(x)
    x = BatchNormalization()(x)
    if sparse:
        z = Dense(latent_dim, name="z", activity_regularizer=regularizers.l1(10e-5))(x)
    else:
        z = Dense(latent_dim, name="z")(x)
    z = BatchNormalization()(z)
    encoder = Model(inputs, z, name="encoder")

    # classifier
    clf_latent_inputs = Input(shape=(latent_dim,), name='z_clf')
    clf_outputs = Dense(n_class, activation='softmax', name='class_output')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')

    # instantiate VAE model
    outputs = clf_supervised(encoder(inputs))
    vae = Model(inputs, outputs, name='vae_mlp')

    opt = optimizers.Adam(learning_rate=lr)
    vae.compile(optimizer=opt, loss='categorical_crossentropy',experimental_run_tf_function=False,metrics=['accuracy'])
    return vae, encoder, clf_supervised


def build_sae_var(latent_dim, n_class, input_type='feat', sparse='True',lr=0.001):
    
    if input_type == 'feat':
        input_shape = (24,)
    elif input_type == 'raw':
        input_shape = (300,)
    elif input_type == 'mav':
        input_shape = (6,)

    # build encoder model
    inputs = Input(shape=input_shape)
    x = Dense(24, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dense(12, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(8, activation="relu")(x)
    x = BatchNormalization()(x)
    z_mean = Dense(latent_dim, name="z_mean", activity_regularizer=regularizers.l1(10e-5))(x)
    z_log_var = Dense(latent_dim, name="z_log_var", activity_regularizer=regularizers.l1(10e-5))(x)
    z_mean = BatchNormalization()(z_mean)
    z_log_var = BatchNormalization()(z_log_var)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    z = BatchNormalization()(z)
    encoder = Model(inputs, z, name="encoder")

    # classifier
    clf_latent_inputs = Input(shape=(latent_dim,), name='z_clf')
    clf_outputs = Dense(n_class, activation='softmax', name='class_output')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')

    # instantiate VAE model
    outputs = clf_supervised(encoder(inputs))
    vae = Model(inputs, outputs, name='vae_mlp')

    opt = optimizers.Adam(learning_rate=lr)
    vae.compile(optimizer=opt, loss='categorical_crossentropy',experimental_run_tf_function=False,metrics=['accuracy'])
    return vae, encoder, clf_supervised

## VARIATIONAL AUTOENCODER - NO CLASSIFIER
def build_vae(latent_dim, input_type='feat'):
    if input_type == 'feat':
        input_shape = (6,4,1)
        inter_shape = (3,2,1)
    elif input_type == 'raw':
        input_shape = (6,100,1)
        inter_shape = (3,50,1)

    # build encoder model
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(inputs)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(inter_shape[0]*inter_shape[1]*32, activation="relu")(latent_inputs)
    x = Reshape((inter_shape[0], inter_shape[1], 32))(x)
    # x = Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same")(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(1, 3, activation="relu", padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    # decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    def VAE_loss(x_origin,x_out):
        # x_origin=K.flatten(x_origin)
        # x_out=K.flatten(x_out)
        # xent_loss = input_shape[0]*input_shape[1] * binary_crossentropy(x_origin, x_out)
        reconstruction_loss = K.mean(mse(x_origin, x_out))
        reconstruction_loss *= input_shape[0] * input_shape[1]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean((reconstruction_loss + kl_loss)/100.0)
        return vae_loss

    vae.compile(optimizer='adam', loss=VAE_loss, experimental_run_tf_function=False)
    return vae, encoder, decoder

def eval_vae(vae, x_test, y_test):
    try:
        temp = vae.predict(x=x_test)
        y_pred = np.argmax(temp, axis=1)
    except:
        test_weights = np.array([[1,1] for _ in range(len(x_test))])
        try:
            temp = vae.predict(x=[x_test, test_weights])
            try:
                y_pred = np.argmax(temp[1], axis=1)
            except:
                y_pred = np.argmax(temp, axis=1)
        except:
            test_y = np.zeros((len(x_test),np.shape(y_test)[1]))
            y_pred = np.argmax(vae.predict(x=[x_test,test_y,test_weights])[1], axis=1)
                          
    acc = np.sum(np.argmax(y_test, axis=1) == y_pred)/y_pred.shape[0]
    return y_pred, acc

def recon_vae(vae, x_test):
    if type(vae.predict(x=x_test)) is list:
        x_pred = vae.predict(x=x_test)[0]
    else:
        x_pred = vae.predict(x=x_test)
    return x_pred

## NOT SURE IF THIS WORKS
def build_vae_old(latent_dim, input_type='feat'):
    # VAE model = encoder + decoder
    # build encoder model
    if input_type == 'feat':
        input_shape = (6,10,1)
        inter_shape = (3,5,1)
    elif input_type == 'raw':
        input_shape = (6,100,1)
        inter_shape = (3,50,1)

    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(inter_shape[0]*inter_shape[1]*32, activation="relu")(latent_inputs)
    x = Reshape((inter_shape[0], inter_shape[1], 32))(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same")(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(1, 3, activation="tanh", padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # New: add a classifier
    clf_latent_inputs = Input(shape=(latent_dim,), name='z_sampling_clf')
    clf_outputs = Dense(7, activation='softmax', name='class_output')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')

    # instantiate VAE model
    outputs = [decoder(encoder(inputs)[2]), clf_supervised(encoder(inputs)[2])]
    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = K.mean(mse(inputs, outputs[0]))
    reconstruction_loss *= input_shape[0] * input_shape[1]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean((reconstruction_loss + kl_loss)/100.0)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam', loss={'clf': 'categorical_crossentropy'})
    return vae, encoder, decoder, clf_supervised

## NOT SURE WHAT THIS IS
def build_vae_s(latent_dim, input_type='feat'):
    if input_type == 'feat':
        input_shape = (6,10,1)
        inter_shape = (3,5,1)
    elif input_type == 'raw':
        input_shape = (6,100,1)
        inter_shape = (3,50,1)

    # build encoder model
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    x = Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    encoder1 = Model(inputs,x, name="encoder1")

    enc_latent_inputs = Input(shape=(16,))
    z_mean = Dense(latent_dim, name="z_mean")(enc_latent_inputs)
    z_log_var = Dense(latent_dim, name="z_log_var")(enc_latent_inputs)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(enc_latent_inputs, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()

    # New: add a classifier
    clf_latent_inputs = Input(shape=(16,), name='z_sampling_clf')
    clf_outputs = Dense(32, activation='relu')(clf_latent_inputs)
    clf_outputs = Dense(7, activation='softmax', name='class_output')(clf_outputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')
    # clf_supervised.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,))
    label_inputs = Input(shape=(7,))
    clf_out = Dense(latent_dim, activation='relu')(label_inputs)
    x = Dense(latent_dim, activation="relu")(concatenate([clf_out, latent_inputs]))
    x = Dense(inter_shape[0]*inter_shape[1]*32, activation="relu")(x)
    x = Reshape((inter_shape[0], inter_shape[1], 32))(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same")(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(1, 3, activation="tanh", padding="same")(x)
    decoder = Model([label_inputs, latent_inputs], decoder_outputs, name="decoder")
    # decoder.summary()

    # instantiate VAE model
    outputs = decoder([clf_supervised(encoder1(inputs)), encoder(encoder1(inputs))[2]])
    vae = Model(inputs, outputs, name='vae_mlp')
    plot_model(vae, to_file='vae.png', show_shapes=True)

    reconstruction_loss = K.mean(mse(inputs, outputs))

    reconstruction_loss *= input_shape[0] * input_shape[1]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean((reconstruction_loss + kl_loss)/100.0)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam', loss={'clf': 'categorical_crossentropy'})
    return vae, encoder1, encoder, decoder, clf_supervised

def eval_vae_s(enc, clf, x_test, y_test):
    y_pred = np.argmax(clf.predict(enc.predict)[1], axis=1)
    acc = np.sum(np.argmax(y_test, axis=1) == y_pred)/y_pred.shape[0]
    return y_pred, acc

## OLD TRANSFER LEARNING STUFF
def build_pnn(source_weights, latent_dim, input_type='feat'):
    
    ## SOURCE MODEL
    # build encoder model
    if input_type == 'feat':
        input_shape = (6,10,1)
        inter_shape = (3,5,1)
    elif input_type == 'raw':
        input_shape = (6,100,1)
        inter_shape = (3,50,1)

    inputs = Input(shape=input_shape)
    enc1 = Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    enc2 = Conv2D(32, 3, activation="relu", strides=1, padding="same")(enc1)
    enc3 = Flatten()(enc2)
    enc4 = Dense(16, activation="relu")(enc3)
    z_mean = Dense(latent_dim, name="z_mean")(enc4)
    z_log_var = Dense(latent_dim, name="z_log_var")(enc4)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    # encoder.summary()
    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,))
    dec1 = Dense(inter_shape[0]*inter_shape[1]*32, activation="relu")(latent_inputs)
    dec2 = Reshape((inter_shape[0], inter_shape[1], 32))(dec1)
    dec3 = Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same")(dec2)
    dec4 = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(dec3)
    decoder_outputs = Conv2DTranspose(1, 3, activation="tanh", padding="same")(dec4)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    # decoder.summary()
    plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # New: add a classifier
    clf_inputs = Input(shape=(latent_dim,), name='z_sampling_clf')
    # clf1 = Dense(32, activation='relu')(clf_inputs)
    clf2 = Dense(7, activation='softmax', name='class_output')(clf_inputs)
    clf = Model(clf_inputs, clf2, name='clf')
    # clf_supervised.summary()

    # instantiate VAE model
    outputs = [decoder(encoder(inputs)[2]), clf(encoder(inputs)[2])]
    vae = Model(inputs, outputs, name='s_vae')

    reconstruction_loss = K.mean(mse(inputs, outputs[0]))

    reconstruction_loss *= input_shape[0] * input_shape[1]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean((reconstruction_loss + kl_loss)/100.0)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam', loss={'clf': 'categorical_crossentropy'})
    
    vae.set_weights(source_weights)
    # Freeze model
    for x in vae.layers:
        x.trainable = False

    ## TARGET MODEL
    # t_inputs = Input(shape=input_shape)
    t_enc1 = Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    t_enc2 = Conv2D(32, 3, activation="relu", strides=1, padding="same")(concatenate([enc1, t_enc1]))
    t_enc3 = Flatten()(t_enc2)
    t_enc4 = Dense(16, activation="relu")(concatenate([enc3, t_enc3]))
    t_z_mean = Dense(latent_dim, name="t_z_mean")(concatenate([enc4, t_enc4]))
    t_z_log_var = Dense(latent_dim, name="t_z_log_var")(concatenate([enc4, t_enc4]))
    t_z = Lambda(sampling, output_shape=(latent_dim,), name='t_z')([t_z_mean, t_z_log_var])
    t_encoder = Model(inputs, [t_z_mean, t_z_log_var, t_z], name="t_encoder")
    # t_encoder.summary()
    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    # t_latent_inputs = Input(shape=(latent_dim,))
    t_dec1 = Dense(inter_shape[0]*inter_shape[1]*32, activation="relu")(latent_inputs)
    t_dec2 = Reshape((inter_shape[0], inter_shape[1], 32))(t_dec1)
    t_dec3 = Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same")(concatenate([dec2, t_dec2]))
    t_dec4 = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(concatenate([dec3, t_dec3]))
    t_decoder_outputs = Conv2DTranspose(1, 3, activation="tanh", padding="same")(concatenate([dec4, t_dec4]))
    t_decoder = Model(latent_inputs, t_decoder_outputs, name="t_decoder")
    # decoder.summary()
    plot_model(t_decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # New: add a classifier
    # t_clf_inputs = Input(shape=(latent_dim,), name='t_z_sampling_clf')
    # t_clf1 = Dense(32, activation='relu')(clf_inputs)
    t_clf2 = Dense(7, activation='softmax', name='class_output')(clf_inputs)#(concatenate([clf_inputs, t_clf1]))
    t_clf = Model(clf_inputs, t_clf2, name='t_clf')

    # instantiate VAE model
    t_outputs = [t_decoder(t_encoder(inputs)[2]), t_clf(t_encoder(inputs)[2])]
    t_vae = Model(inputs, t_outputs, name='t_vae')

    reconstruction_loss = K.mean(mse(inputs, outputs[0]))

    reconstruction_loss *= input_shape[0] * input_shape[1]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean((reconstruction_loss + kl_loss)/100.0)
    t_vae.add_loss(vae_loss)
    t_vae.compile(optimizer='adam', loss={'t_clf': 'categorical_crossentropy'})

    plot_model(t_vae, to_file='pnn.png', show_shapes=True)

    return t_vae, t_encoder, t_decoder, t_clf

## FROM OLD TUTORIAL
# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """
    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "digits_over_latent.png")

    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

def get_batches(x,batch_size):
    n_batches = len(x)//(batch_size)
    x = x[:n_batches*batch_size:]
    for n in range(0, x.shape[0],batch_size):
        x_batch = x[n:n+(batch_size),...]
        yield x_batch