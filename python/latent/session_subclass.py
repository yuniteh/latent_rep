from sklearn.preprocessing import MinMaxScaler
from latent.ml.dl_subclass import MLP, CNN, eval_nn, get_train
from latent.ml.lda import train_lda, eval_lda
import numpy as np
import latent.utils.data_utils as prd
import tensorflow as tf
import pickle


class Sess():
    def __init__(self,**settings):
        self.sub_type = settings.get('sub_type','AB')
        self.sub = settings.get('sub',1)

        self.train_grp = settings.get('train_grp',2)
        self.train_scale = settings.get('train_scale',5)
        self.train = settings.get('train','fullallmix4')
        self.cv_type = settings.get('cv_type','manual')
        self.feat_type = settings.get('feat_type','feat')
        self.scaler_load = settings.get('scaler_load',True)
        self.epochs = settings.get('epochs',30)

        self.test_grp = settings.get('test_grp',4)
        self.test = settings.get('test','partposrealmixeven14')

        self.emg_scale = settings.get('emg_scale',1)
        self.scaler = settings.get('scaler',MinMaxScaler(feature_range=(0,1)))
        self.scaler_noise = settings.get('scaler',MinMaxScaler(feature_range=(0,1)))
    
    def update(self,**settings):
        for k in settings:
            setattr(self, k, settings[k])
    
    def train_sub(self,raw,params,data_load):
        train_ind = (params[:,0] == self.sub) & (params[:,3] == self.train_grp)

        if np.sum(train_ind):
            n_dof = np.max(params[train_ind,4])
        
            # Train NNs
            mlp = MLP(n_class=n_dof)
            # mlp_beta = MLPbeta(n_class=n_dof)
            cnn = CNN(n_class=n_dof)
        
            if data_load:
                try:
                    with open('subclass/train/' + str(self.sub_type) + str(self.sub) + '_' + str(self.feat_type) + '.p', 'rb') as f:
                        x_train_cnn, x_train_mlp, x_train_lda, x_train_aug, y_train, y_train_lda, self.scaler = pickle.load(f)
                except:
                    print('no training data to load')
                    data_load = False
                
                trainmlp = tf.data.Dataset.from_tensor_slices((x_train_mlp, y_train)).shuffle(x_train_mlp.shape[0],reshuffle_each_iteration=True).batch(128)
                traincnn = tf.data.Dataset.from_tensor_slices((x_train_cnn, y_train)).shuffle(x_train_cnn.shape[0],reshuffle_each_iteration=True).batch(128)
                
            if not data_load:   
                trainmlp, _, traincnn, _, y_train, _, x_train_mlp, x_train_cnn, x_train_lda, y_train_lda, x_train_aug = prd.prep_train_data(self,raw,params)
                # trainmlp, traincnn, y_train, x_train_mlp, x_train_cnn, x_train_lda, y_train_lda, x_train_aug, emg_scale, scaler, x_min, x_max = prd.prep_train_caps(data, params)
                # with open('subclass/train/' + str(self.sub_type) + str(sub) + '_' + str(self.feat_type) + '.p', 'wb') as f:
                #     pickle.dump([x_train_cnn, x_train_mlp, x_train_lda, x_train_aug, y_train, y_train_lda, self.scaler], f)

            # with open('subclass/models/tdar_lat_8/' + str(self.sub_type) + str(sub) + '_' + str(self.feat_type) + '.p','rb') as f:
            #     mlp_w, mlpb_w, cnn_w, w_mlp, c_mlp, w_mlpbeta, c_mlpbeta, w_cnn, c_cnn, w, c, w_aug, c_aug, self.emg_scale, self.scaler, mu_class,C = pickle.load(f)
            #     # _, _, _, _, _, _, _, _, _, _, _, _, _, self.emg_scale, self.scaler, _, _ = pickle.load(f)
            # Train neural networks
            optimizer = tf.keras.optimizers.Adam()
            train_loss = tf.keras.metrics.Mean(name='train_loss')
            train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
            models = [mlp, cnn]
            for model in models:
                if isinstance(model,CNN):
                    ds = traincnn
                    del trainmlp
                else:
                    ds = trainmlp
                
                train_mod = get_train()

                for epoch in range(self.epochs):
                    # Reset the metrics at the start of the next epoch
                    train_loss.reset_states()
                    train_accuracy.reset_states()

                    for x, y in ds:
                        train_mod(x, y, model, optimizer, train_loss, train_accuracy)

                    if epoch == 0 or epoch == self.epochs-1:
                        print(
                            f'Epoch {epoch + 1}, '
                            f'Loss: {train_loss.result():.2f}, '
                            f'Accuracy: {train_accuracy.result() * 100:.2f} '
                        )
                tf.keras.backend.clear_session()
                
                del train_mod

                # mlp(x_train_mlp[:1,...])
                # cnn(x_train_cnn[:1,...])
                # mlp.set_weights(mlp_w)
                # cnn.set_weights(cnn_w)
                # Train aligned LDA
                y_train_aug = np.argmax(y_train, axis=1)[...,np.newaxis]

                mlp_enc = mlp.get_layer(name='enc')
                w_mlp, c_mlp,_, _, v_mlp = train_lda(mlp_enc(x_train_mlp).numpy(),y_train_aug)

                # mlpbeta_enc = mlp_beta.get_layer(name='enc')
                # w_mlpbeta, c_mlpbeta,_, _, _ = train_lda(mlpbeta_enc(x_train_mlp).numpy(),y_train_aug)
                del x_train_mlp
                cnn_enc = cnn.get_layer(name='enc')
                # temp = cnn_enc(x_train_cnn[:x_train_cnn.shape[0]//4,...]).numpy()
                # temp2 = np.vstack((temp,cnn_enc(x_train_cnn[x_train_cnn.shape[0]//4:,...]).numpy()))
                # temp2 = np.vstack((temp,cnn_enc(x_train_cnn[x_train_cnn.shape[0]//4:,...]).numpy()))
                temp2 = np.vstack((cnn_enc(x_train_cnn[:x_train_cnn.shape[0]//4,...]),cnn_enc(x_train_cnn[x_train_cnn.shape[0]//4:x_train_cnn.shape[0]//2,...]).numpy()))
                temp2 = np.vstack((temp2, cnn_enc(x_train_cnn[x_train_cnn.shape[0]//2:3*x_train_cnn.shape[0]//4,...]),cnn_enc(x_train_cnn[3*x_train_cnn.shape[0]//4:,...]).numpy()))
                w_cnn, c_cnn,_, _, v_cnn = train_lda(temp2,y_train_aug)

                # Train LDA
                w,c, mu_class, C, v = train_lda(x_train_lda,y_train_lda)
                w_aug,c_aug, _, _, v_noise = train_lda(x_train_aug,y_train_aug)

                mlp_w = mlp.get_weights()
                # mlpb_w = mlp_beta.get_weights()
                cnn_w = cnn.get_weights()
                mlpb_w = 0
                w_mlpbeta=0
                c_mlpbeta=0
                with open('subclass/models_new/' + str(self.sub_type) + str(self.sub) + '_' + str(self.feat_type) + '_red.p', 'wb') as f:
                    pickle.dump([v_mlp, v_mlp, v_cnn, v_cnn, v_cnn, v, v_noise],f)
                with open('subclass/models_new/' + str(self.sub_type) + str(self.sub) + '_' + str(self.feat_type) + '.p','wb') as f:
                    pickle.dump([mlp_w, mlpb_w, cnn_w, w_mlp, c_mlp, w_mlpbeta, c_mlpbeta, w_cnn, c_cnn, w, c, w_aug, c_aug, self.emg_scale, self.scaler, mu_class,C],f)
        else:
            print("no training data")
    
    def test_sub(self,raw,params,data_load):
        # Load test data
        if data_load:
            try:
                with open('subclass/test/' + str(self.sub_type) + str(self.sub) + '_' + str(self.feat_type) + '_' + str(self.test) + '.p', 'rb') as f:
                    x_test_lda, y_test, clean_size = pickle.load(f)
            except:
                print('no testing data to load')
                data_load = False

            x_temp = np.transpose(x_test_lda.reshape((x_test_lda.shape[0],int(x_test_lda.shape[1]/6),-1)),(0,2,1))[...,np.newaxis]
            x_test_cnn = self.scaler.transform(x_temp.reshape(x_temp.shape[0]*x_temp.shape[1],-1)).reshape(x_temp.shape)
            x_test_cnn = x_test_cnn.astype('float32')

            # Reshape for nonconvolutional SAE
            x_test_mlp = x_test_cnn.reshape(x_test_cnn.shape[0],-1)
        if not data_load:
            x_test_cnn, x_test_mlp, x_test_lda, y_test, clean_size = prd.prep_test_data(self, raw, params, real_noise_temp)

            with open('subclass/test/' + str(self.sub_type) + str(self.sub) + '_' + str(self.feat_type) + '_' + str(self.test) + '.p', 'wb') as f:
                pickle.dump([x_test_lda, y_test, clean_size],f)

        # workaround
        mlp(x_test_mlp[:2,...])
        mlp_beta(x_test_mlp[:2,...])
        cnn(x_test_cnn[:2,...])

        mlp.set_weights(mlp_w)
        mlp_beta.set_weights(mlpb_w)
        cnn.set_weights(cnn_w)
    
        # Test
        mlp_test_aligned = mlp_enc(x_test_mlp).numpy()
        mlpbeta_test_aligned = mlpbeta_enc(x_test_mlp).numpy()
        cnn_test_aligned = cnn_enc(x_test_cnn).numpy()
        y_test_aligned = np.argmax(y_test, axis=1)[...,np.newaxis]

        # LDA
        clean_lda, noisy_lda = eval_lda(w, c, x_test_lda, y_test_aligned, clean_size)
        # AUG-LDA
        clean_aug, noisy_aug = eval_lda(w_aug, c_aug, x_test_lda, y_test_aligned, clean_size)
        # MLP
        clean_mlp, noisy_mlp = eval_nn(x_test_mlp, y_test,mlp,clean_size)
        clean_mlplda, noisy_mlplda = eval_lda(w_mlp, c_mlp, mlp_test_aligned, y_test_aligned, clean_size)
        # MLP Beta
        clean_mlpb, noisy_mlpb = eval_nn(x_test_mlp, y_test,mlp_beta,clean_size)
        clean_mlpblda, noisy_mlpblda = eval_lda(w_mlpbeta, c_mlpbeta, mlpbeta_test_aligned, y_test_aligned, clean_size)
        # CNN
        clean_cnn, noisy_cnn = eval_nn(x_test_cnn, y_test,cnn,clean_size)
        clean_cnnlda, noisy_cnnlda = eval_lda(w_cnn, c_cnn, cnn_test_aligned, y_test_aligned, clean_size)

        print( 
            f'LDA ---- '
            f'Clean: {clean_lda * 100:.2f}, '
            f'Noisy: {noisy_lda * 100:.2f}'
            f'\nAUG ---- '
            f'Clean: {clean_aug * 100:.2f}, '
            f'Noisy: {noisy_aug * 100:.2f}'
            f'\nMLP ---- '
            f'Clean: {clean_mlp * 100:.2f}, '
            f'Noisy: {noisy_mlp * 100:.2f}, '
            f'LDA Clean: {clean_mlplda * 100:.2f}, '
            f'LDA Noisy: {noisy_mlplda * 100:.2f}'
            f'\nMLPB ---- '
            f'Clean: {clean_mlpb * 100:.2f}, '
            f'Noisy: {noisy_mlpb * 100:.2f}, '
            f'LDA Clean: {clean_mlpblda * 100:.2f}, '
            f'LDA Noisy: {noisy_mlpblda * 100:.2f}'
            f'\nCNN ---- '
            f'Clean: {clean_cnn * 100:.2f}, '
            f'Noisy: {noisy_cnn * 100:.2f}, '
            f'LDA Clean: {clean_cnnlda * 100:.2f}, '
            f'LDA Noisy: {noisy_cnnlda * 100:.2f}'
        )

        clean = np.stack((clean_lda,clean_aug,clean_mlp,clean_mlplda,clean_mlpb,clean_mlpblda,clean_cnn,clean_cnnlda))
        noisy = np.stack((noisy_lda,noisy_aug,noisy_mlp,noisy_mlplda,noisy_mlpb,noisy_mlpblda,noisy_cnn,noisy_cnnlda))

        return clean, noisy