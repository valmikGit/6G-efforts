from functions_OTFS import update_mu_Sigma_OTFS, circular_padding_1d
from tensorflow.keras.layers import Input, Conv2D, Lambda
from tensorflow.keras import backend as K
import os
# import matplotlib.pyplot as plt
import argparse
from scipy import io
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import mat73
import random
import h5py
from tensorflow.keras.models import load_model


np.random.seed(2024)

tf.random.set_seed(2024)
random.seed(2024)
# %%


def custom_loss(y_true, y_pred):
    pruning_thrld = 0.1

    thresholded_pred = tf.where(tf.abs(y_pred) < pruning_thrld, tf.zeros_like(y_pred), y_pred)
    # thresholded_true = tf.where(tf.abs(y_true) < pruning_thrld, tf.zeros_like(y_true), y_true)

    # mse = K.mean(K.square(tf.abs(thresholded_true-thresholded_pred)))
    mse = K.mean(K.square(tf.abs(y_true-thresholded_pred)))

    return mse


# %% Enable GPU usage
parser = argparse.ArgumentParser()

parser.add_argument('-data_num')
parser.add_argument('-gpu_index')
#
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# %% Building SBL_net model
def SBL_net_OTFS(Nt, Nr, Ms, Gs, G, num_layers, num_filters, kernel_size):
    pruning_thrld = 0.01
    alpha = 0.0001 + 1j*0.0001
    wt = 0.1
    # scale_factor = 0.01
    y_real_imag = Input(shape=(Np2, 2))
    Psi_real_imag = Input(shape=(Np2, Ms*Gs, 2))
    sigma_2 = Input(shape=(1, 1))

    Gamma_init = tf.tile(tf.ones_like(
        Psi_real_imag[:, 0, 0:1, 0:1]), (1, Ms*Gs, num_sc))

    # update mu and Sigma
    mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma_OTFS(
        x, Np2))([Psi_real_imag, y_real_imag, Gamma_init, sigma_2])

    for i in range(num_layers):

        mu_real_old = mu_real
        mu_imag_old = mu_imag

        # mu_complex = tf.cast(mu_real, tf.complex64) + 1j * tf.cast(mu_imag, tf.complex64)
        # mu_complex_expanded = tf.transpose((tf.reshape(mu_complex, (-1, Gs, Ms, 1))), (0, 2, 1, 3))
        # thresholded_MU = tf.where(tf.abs(mu_complex_expanded) < pruning_thrld, tf.zeros_like(mu_complex_expanded), mu_complex_expanded)
        # MU_real= tf.math.real(thresholded_MU)
        # MU_imag = tf.math.imag(thresholded_MU)

        # Pre-processing before Conv2D

        mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])
        MU_square = tf.transpose((tf.reshape(mu_square, (-1, Gs, Ms, 1))), (0, 2, 1, 3))
        diag_Sigma_expanded = tf.transpose((tf.reshape(diag_Sigma_real, (-1, Gs, Ms, 1))), (0, 2, 1, 3))
        temp_orig = Lambda(lambda x: tf.concat(x, axis=-1))([MU_square, diag_Sigma_expanded])

        mu_complex = tf.cast(mu_real, tf.complex64) + 1j * tf.cast(mu_imag, tf.complex64)
        mu_complex_expanded = tf.transpose((tf.reshape(mu_complex, (-1, Gs, Ms, 1))), (0, 2, 1, 3))
        
        # alpha = tf.reduce_mean(mu_complex_expanded)
        # mean_per_map = tf.reduce_mean(mu_complex_expanded, axis=[1,2,3], keepdims=True)
        # mask = tf.abs(mu_complex_expanded) >= pruning_thrld
        # thresholded_MU =  tf.where(mask,(scale_factor*mean_per_map), 0)

        thresholded_MU = tf.where(tf.abs(mu_complex_expanded) < pruning_thrld, tf.zeros_like(mu_complex_expanded), alpha)
        MU_real = tf.math.real(thresholded_MU)
        MU_imag = tf.math.imag(thresholded_MU)
        # MU_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([MU_real, MU_imag])
        F3 = tf.concat([MU_real, MU_imag], axis=-1)
        # F3 = MU_square

        temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp_orig, F3])
        # temp = temp_orig
#
        # Convolution Filtering

        # conv_layer1 = Conv2D(name='SBL_%d1' % i, filters=128, kernel_size=9,
        #                      strides=1, padding='valid', activation='relu')
        # temp_padded1 = circular_padding_1d(temp, kernel_size=9, strides=1)
        # temp_conv1 = conv_layer1(temp_padded1)
        # temp_conv1 = Lambda(lambda x: tf.concat(x, axis=-1))([temp_conv1, F3])

        conv_layer1 = Conv2D(name='SBL_%d1' % i, filters=64, kernel_size=7,
                             strides=1, padding='valid', activation='relu')
        temp_padded1 = circular_padding_1d(temp, kernel_size=7, strides=1)
        temp_conv1 = conv_layer1(temp_padded1)
        temp_conv1 = Lambda(lambda x: tf.concat(x, axis=-1))([temp_conv1, F3])

        conv_layer2 = Conv2D(name='SBL_%d2' % i, filters=32, kernel_size=5,
                             strides=1, padding='valid', activation='relu')
        temp_padded2 = circular_padding_1d(
            temp_conv1, kernel_size=5, strides=1)
        temp_conv2 = conv_layer2(temp_padded2)
        temp_conv2 = Lambda(lambda x: tf.concat(x, axis=-1))([temp_conv2, F3])

        conv_layer3 = Conv2D(name='SBL_%d3' % i, filters=16, kernel_size=3,
                             strides=1, padding='valid', activation='relu')
        temp_padded3 = circular_padding_1d(
            temp_conv2, kernel_size=3, strides=1)
        temp_conv3 = conv_layer3(temp_padded3)
        temp_conv3 = Lambda(lambda x: tf.concat(x, axis=-1))([temp_conv3, F3])

        conv_layer4 = Conv2D(name='SBL_%d4' % i, filters=1, kernel_size=3,
                             strides=1, padding='valid', activation='relu')
        temp_padded4 = circular_padding_1d(
            temp_conv3, kernel_size=3, strides=1)
        temp_conv4 = conv_layer4(temp_padded4)
        # temp_conv4 = Lambda(lambda x: tf.concat(x, axis=-1))([temp_conv4, F3])

        # conv_layer5 = Conv2D(name='SBL_%d5' % i, filters=1, kernel_size=3,
        #                      strides=1, padding='valid', activation='relu')
        # temp_padded5 = circular_padding_1d(
        #     temp_conv4, kernel_size=3, strides=1)
        # temp_conv5 = conv_layer5(temp_padded5)

        Gamma = temp_conv4

        # Update mu and Sigma using new Gamma
        mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma_OTFS(x, Np2))(
            [Psi_real_imag, y_real_imag, tf.reshape(tf.transpose(Gamma, (0, 2, 1, 3)), (-1, Ms*Gs, 1)), sigma_2])

        mu_real = wt*mu_real_old + (1-wt)*mu_real
        mu_imag = wt*mu_imag_old+(1-wt)*mu_imag

    x_hat = Lambda(lambda x: tf.concat(
        [x[0], x[1]], axis=-1))([mu_real, mu_imag])

    model = Model(inputs=[Psi_real_imag, y_real_imag, sigma_2], outputs=x_hat)
    return model
# %% Load channel

# To investigate the impact of nn structure, use typical parameters


num_layers = 7
num_filters = 8
kernel_size = 8

epochs = 1000
batch_size = 16

resolution = 1  # resolution of angle grids
Nt = 32
Nr = 32

Ms = 16
Gs = 32
Np2 = 150
G = resolution*np.max([Ms, Gs])
G_A = 64
G_B = 64

div = 8

# SNRdb = 10
# sigma_2 = 1/10**(SNRdb/10)
num_sc = 1
mse_final = []
nmse_final = []

mode = ['LCConv3D']
data_num = int(args.data_num)
analyse_num = 1000



# h_list = io.loadmat('./Unpruned/hDL_10dB_20k_randomddc_Lp5_unpruned.mat')['hDL'][-data_num:]
# y_list = io.loadmat('./Unpruned/yDL_10dB_20k_randomddc_Lp5_unpruned.mat')['yDL'][-data_num:]
# Psi_list = mat73.loadmat('./Unpruned/PsiDL_10dB_20k_randomddc_Lp5_unpruned.mat')['PsiDL'][-data_num:]
# sigma2_list = io.loadmat('./Unpruned/sigma2DL_10dB_20k_randomddc_Lp5_unpruned.mat')['sigma2DL'][-data_num:]




# hm5 = io.loadmat('./data/hDL_m5dB_1k_150pilots_prunede04_ipjp.mat')['hDL'][-analyse_num:]
# h0 = io.loadmat('./data/hDL_0dB_1k_150pilots_prunede04_ipjp.mat')['hDL'][-analyse_num:]
# h5 = io.loadmat('./data/hDL_5dB_1k_150pilots_prunede04_ipjp.mat')['hDL'][-analyse_num:]
# h10 = io.loadmat('./data/hDL_10dB_1k_150pilots_prunede04_ipjp.mat')['hDL'][-analyse_num:]
# h15 = io.loadmat('./data/hDL_15dB_1k_150pilots_prunede04_ipjp.mat')['hDL'][-analyse_num:]


# h_list = io.loadmat(
#     './hDL_15dB_15k_150pilots_prunede04_ipjp.mat')['hDL'][-data_num:]
# y_list = io.loadmat(
#     './yDL_15dB_15k_150pilots_prunede04_ipjp.mat')['yDL'][-data_num:]
# Psi_list = mat73.loadmat(
#     './PsiDL_15dB_15k_150pilots_prunede04_ipjp.mat')['PsiDL'][-data_num:]
# sigma2_list = io.loadmat(
#     './sigma2DL_15dB_15k_150pilots_prunede04_ipjp.mat')['sigma2DL'][-data_num:]


# h_list = io.loadmat('./Unpruned/hDL_10dB_20k_150pilots_ipjp.mat')['hDL'][-data_num:]
# y_list = io.loadmat('./Unpruned/yDL_10dB_20k_150pilots_ipjp.mat')['yDL'][-data_num:]
# Psi_list = mat73.loadmat('./Unpruned/PsiDL_10dB_20k_150pilots_ipjp.mat')['PsiDL'][-data_num:]
# sigma2_list = io.loadmat('./Unpruned/sigma2DL_10dB_20k_150pilots_ipjp.mat')['sigma2DL'][-data_num:]

# h_list = io.loadmat('./hDL_10dB_40k_150pilots_ipjp.mat')['hDL'][-data_num:]
# y_list = io.loadmat('./yDL_10dB_40k_150pilots_ipjp.mat')['yDL'][-data_num:]
# Psi_list = mat73.loadmat('./PsiDL_10dB_40k_150pilots_ipjp.mat')['PsiDL'][-data_num:]
# sigma2_list = io.loadmat('./sigma2DL_10dB_40k_150pilots_ipjp.mat')['sigma2DL'][-data_num:]


# h = h_list1[:, :, 0] + 1j*h_list1[:, :, 1]
# thresholded_h = np.where(np.abs(h) < 0.00001, np.zeros_like(h), h)
# h_real = np.real(thresholded_h)
# h_imag = np.imag(thresholded_h)
# h = np.concatenate([np.expand_dims(h_real, axis=2), np.expand_dims(h_imag, axis=2)], axis = 2)
# h_list = h

# abs_h = abs(h)

# h_list = io.loadmat('./data/hDL_1015dB_10k_6th_150pilots_pruned_ipjp.mat')['hDL'][-data_num:]
# y_list = io.loadmat('./data/yDL_1015dB_10k_6th_150pilots_pruned_ipjp.mat')['yDL'][-data_num:]
# Psi_list = mat73.loadmat('./data/PsiDL_1015dB_10k_6th_150pilots_pruned_ipjp.mat')['PsiDL'][-data_num:]
# sigma2_list = io.loadmat('./data/sigma2DL_1015dB_10k_6th_150pilots_pruned_ipjp.mat')['sigma2DL'][-data_num:]

# h_list = io.loadmat('./hDL_10dB_10k_fixedipjp_pruned_REVIEWphase.mat')['hDL'][-data_num:]
# y_list = io.loadmat('./yDL_10dB_10k_fixedipjp_pruned_REVIEWphase.mat')['yDL'][-data_num:]
# Psi_list = mat73.loadmat('./PsiDL_10dB_10k_fixedipjp_pruned_REVIEWphase.mat')['PsiDL'][-data_num:]
# sigma2_list = io.loadmat('./sigma2DL_10dB_10k_fixedipjp_pruned_REVIEWphase.mat')['sigma2DL'][-data_num:]
# sigma2_list_new = sigma2_list.astype(np.complex64)

h_list = io.loadmat('./hDL_10dB_40k_150pilots_ipjp.mat')['hDL'][-data_num:]
y_list = io.loadmat('./yDL_10dB_40k_150pilots_ipjp.mat')['yDL'][-data_num:]
Psi_list = mat73.loadmat('./PsiDL_10dB_40k_150pilots_ipjp.mat')['PsiDL'][-data_num:]
sigma2_list = io.loadmat('./sigma2DL_10dB_40k_150pilots_ipjp.mat')['sigma2DL'][-data_num:]
sigma2_list_new = sigma2_list.astype(np.complex64)

# print("Shape of Psi_list:", Psi_list.shape)
# print("Shape of h_list:", h_list.shape)
# print("Shape of y_list:", y_list.shape)
# print("Shape of sigma2_list:", sigma2_list_new.shape)

# %%

# %%  #%%Analysis

# SNRdb = 10
# pruning_thrld = 0.0001
# alpha = 0.0001 + 1j*0.0001
# wt = 0.1

# y_real_imag = y_list 
# Psi_real_imag =  Psi_list
# sigma_2 = 1/10**(SNRdb/10) 


# Gamma_init = tf.tile(tf.ones_like(Psi_real_imag[:, 0, 0:1, 0:1]), (1, Ms*Gs, num_sc))

# # update mu and Sigma
# mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma_OTFS(
# x, Np2))([Psi_real_imag, y_real_imag, Gamma_init, sigma_2])

# for i in range(1):

#     mu_real_old = mu_real
#     mu_imag_old = mu_imag

#     # mu_complex = tf.cast(mu_real, tf.complex64) + 1j * tf.cast(mu_imag, tf.complex64)
#     # mu_complex_expanded = tf.transpose((tf.reshape(mu_complex, (-1, Gs, Ms, 1))), (0, 2, 1, 3))
#     # thresholded_MU = tf.where(tf.abs(mu_complex_expanded) < pruning_thrld, tf.zeros_like(mu_complex_expanded), mu_complex_expanded)
#     # MU_real= tf.math.real(thresholded_MU)
#     # MU_imag = tf.math.imag(thresholded_MU)

#     # Pre-processing before Conv2D

#     mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])
#     MU_square = tf.transpose((tf.reshape(mu_square, (-1, Gs, Ms, 1))), (0, 2, 1, 3))
#     diag_Sigma_expanded = tf.transpose((tf.reshape(diag_Sigma_real, (-1, Gs, Ms, 1))), (0, 2, 1, 3))
#     temp_orig = Lambda(lambda x: tf.concat(x, axis=-1))([MU_square, diag_Sigma_expanded])

#     mu_complex = tf.cast(mu_real, tf.complex64) + 1j * tf.cast(mu_imag, tf.complex64)
#     mu_complex_expanded = tf.transpose((tf.reshape(mu_complex, (-1, Gs, Ms, 1))), (0, 2, 1, 3))
    
#     # alpha = tf.reduce_mean(mu_complex_expanded)
#     # mean_per_map = tf.reduce_mean(mu_complex_expanded, axis=[1,2,3], keepdims=True)
#     # mask = tf.abs(mu_complex_expanded) >= pruning_thrld
#     # thresholded_MU =  tf.where(mask,(scale_factor*mean_per_map), 0)

#     thresholded_MU = tf.where(tf.abs(mu_complex_expanded) < pruning_thrld, tf.zeros_like(mu_complex_expanded), alpha)
#     MU_real = tf.math.real(thresholded_MU)
#     MU_imag = tf.math.imag(thresholded_MU)
#     # MU_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([MU_real, MU_imag])
#     F3 = tf.concat([MU_real, MU_imag], axis=-1)
#     # F3 = MU_square

#     temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp_orig, F3])
#     # temp = temp_orig
# #
#     # Convolution Filtering

#     # conv_layer1 = Conv2D(name='SBL_%d1' % i, filters=128, kernel_size=9,
#     #                      strides=1, padding='valid', activation='relu')
#     # temp_padded1 = circular_padding_1d(temp, kernel_size=9, strides=1)
#     # temp_conv1 = conv_layer1(temp_padded1)
#     # temp_conv1 = Lambda(lambda x: tf.concat(x, axis=-1))([temp_conv1, F3])

#     conv_layer1 = Conv2D(name='SBL_%d1' % i, filters=64, kernel_size=7,
#                          strides=1, padding='valid', activation='relu')
#     temp_padded1 = circular_padding_1d(temp, kernel_size=7, strides=1)
#     temp_conv1 = conv_layer1(temp_padded1)
#     temp_conv1 = Lambda(lambda x: tf.concat(x, axis=-1))([temp_conv1, F3])

#     conv_layer2 = Conv2D(name='SBL_%d2' % i, filters=32, kernel_size=5,
#                          strides=1, padding='valid', activation='relu')
#     temp_padded2 = circular_padding_1d(
#         temp_conv1, kernel_size=5, strides=1)
#     temp_conv2 = conv_layer2(temp_padded2)
#     temp_conv2 = Lambda(lambda x: tf.concat(x, axis=-1))([temp_conv2, F3])

#     conv_layer3 = Conv2D(name='SBL_%d3' % i, filters=16, kernel_size=3,
#                          strides=1, padding='valid', activation='relu')
#     temp_padded3 = circular_padding_1d(
#         temp_conv2, kernel_size=3, strides=1)
#     temp_conv3 = conv_layer3(temp_padded3)
#     temp_conv3 = Lambda(lambda x: tf.concat(x, axis=-1))([temp_conv3, F3])

#     conv_layer4 = Conv2D(name='SBL_%d4' % i, filters=1, kernel_size=3,
#                          strides=1, padding='valid', activation='relu')
#     temp_padded4 = circular_padding_1d(
#         temp_conv3, kernel_size=3, strides=1)
#     temp_conv4 = conv_layer4(temp_padded4)
#     # temp_conv4 = Lambda(lambda x: tf.concat(x, axis=-1))([temp_conv4, F3])

#     # conv_layer5 = Conv2D(name='SBL_%d5' % i, filters=1, kernel_size=3,
#     #                      strides=1, padding='valid', activation='relu')
#     # temp_padded5 = circular_padding_1d(
#     #     temp_conv4, kernel_size=3, strides=1)
#     # temp_conv5 = conv_layer5(temp_padded5)

#     Gamma = temp_conv4

#   # Update mu and Sigma using new Gamma
#     mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma_OTFS(x, Np2))(
#       [Psi_real_imag, y_real_imag, tf.reshape(tf.transpose(Gamma, (0, 2, 1, 3)), (-1, Ms*Gs, 1)), sigma_2])

#     mu_real = wt*mu_real_old + (1-wt)*mu_real
#     mu_imag = wt*mu_imag_old+(1-wt)*mu_imag

# x_hat = Lambda(lambda x: tf.concat(
#       [x[0], x[1]], axis=-1))([mu_real, mu_imag])

# %% TRAINING

model = SBL_net_OTFS(Nt, Nr, Ms, Gs, G, num_layers, num_filters, kernel_size)
model.summary()

best_model_path = './TESTmodel_for_review_comments.h5'
# best_model_path = './TESTmodel10.h5'



# define callbacks
checkpointer = ModelCheckpoint(
    best_model_path, verbose=1, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                              cooldown=1, verbose=1, mode='auto', min_delta=1e-5, min_lr=1e-5)
# early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=50)

model.compile(loss=custom_loss, optimizer=Adam(learning_rate=1e-4))

loss_history = model.fit([Psi_list, y_list, sigma2_list_new], h_list, epochs=epochs, batch_size=batch_size,
                          verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer, reduce_lr])

# save loss history for plotting
train_loss = np.squeeze(loss_history.history['loss'])
val_loss = np.squeeze(loss_history.history['val_loss'])
io.savemat('./Unpruned/TESTmodel_for_review_comments.mat',
            {'train_loss': train_loss, 'val_loss': val_loss})

# %% TESTING
# test performance and save
# model.load_weights(best_model_path)



# test_num = 1000

# h_list = io.loadmat(
#     './Unpruned/hDL_15dB_2k_150pilots_unpruned_ipjp.mat')['hDL'][-test_num:]
# y_list = io.loadmat(
#     './Unpruned/yDL_15dB_2k_150pilots_unpruned_ipjp.mat')['yDL'][-test_num:]
# Psi_list = mat73.loadmat(
#     './Unpruned/PsiDL_15dB_2k_150pilots_unpruned_ipjp.mat')['PsiDL'][-test_num:]
# sigma2_list_new = io.loadmat(
#     './Unpruned/sigma2DL_15dB_2k_150pilots_unpruned_ipjp.mat')['sigma2DL'][-test_num:]

# h_list = io.loadmat(
#     './Unpruned/hDL_10dB_2k_150pilots_unpruned_ipjp.mat')['hDL'][-test_num:]
# y_list = io.loadmat(
#     './Unpruned/yDL_10dB_2k_150pilots_unpruned_ipjp.mat')['yDL'][-test_num:]
# Psi_list = mat73.loadmat(
#     './Unpruned/PsiDL_10dB_2k_150pilots_unpruned_ipjp.mat')['PsiDL'][-test_num:]
# sigma2_list_new = io.loadmat(
#     './Unpruned/sigma2DL_10dB_2k_150pilots_unpruned_ipjp.mat')['sigma2DL'][-test_num:]

# h_list = io.loadmat(
#     './Unpruned/hDL_5dB_2k_150pilots_ipjp.mat')['hDL'][-test_num:]
# y_list = io.loadmat(
#     './Unpruned/yDL_5dB_2k_150pilots_ipjp.mat')['yDL'][-test_num:]
# Psi_list = mat73.loadmat(
#     './Unpruned/PsiDL_5dB_2k_150pilots_ipjp.mat')['PsiDL'][-test_num:]
# sigma2_list_new = io.loadmat(
#     './Unpruned/sigma2DL_5dB_2k_150pilots_ipjp.mat')['sigma2DL'][-test_num:]

# h_list = io.loadmat(
#     './Unpruned/hDL_0dB_2k_150pilots_ipjp.mat')['hDL'][-test_num:]
# y_list = io.loadmat(
#     './Unpruned/yDL_0dB_2k_150pilots_ipjp.mat')['yDL'][-test_num:]
# Psi_list = mat73.loadmat(
#     './Unpruned/PsiDL_0dB_2k_150pilots_ipjp.mat')['PsiDL'][-test_num:]
# sigma2_list_new = io.loadmat(
    # './Unpruned/sigma2DL_0dB_2k_150pilots_ipjp.mat')['sigma2DL'][-test_num:]

# h_list = io.loadmat(
#     './Unpruned/hDL_m5dB_2k_150pilots_ipjp.mat')['hDL'][-test_num:]
# y_list = io.loadmat(
#     './Unpruned/yDL_m5dB_2k_150pilots_ipjp.mat')['yDL'][-test_num:]
# Psi_list = mat73.loadmat(
#     './Unpruned/PsiDL_m5dB_2k_150pilots_ipjp.mat')['PsiDL'][-test_num:]
# sigma2_list_new = io.loadmat(
#     './Unpruned/sigma2DL_m5dB_2k_150pilots_ipjp.mat')['sigma2DL'][-test_num:]



# true_h = h_list[-test_num:, :, 0] + 1j*h_list[-test_num:, :, 1]
# ####true_H = np.reshape(h_list, [test_num, Ms, Gs, 2])


# predictions_h = model.predict([Psi_list[-test_num:], y_list[-test_num:], sigma2_list_new[-test_num:]])
# predictions_h = predictions_h[:, :, 0] + 1j*predictions_h[:, :, 1]
# io.savemat('./Testmodel10_copy3/m5 dB/h_hat_DL_m5dB_1k',{'h_hat_DL': predictions_h})
# io.savemat('./Testmodel10_copy3/0 dB/h_hat_DL_0dB_1k',{'h_hat_DL': predictions_h})
# io.savemat('./Testmodel10_copy3/5 dB/h_hat_DL_5dB_1k',{'h_hat_DL': predictions_h})
# io.savemat('./Testmodel10_copy3/10 dB/h_hat_DL_10dB_1k',{'h_hat_DL': predictions_h})
# io.savemat('./Testmodel10_copy3/15 dB/h_hat_DL_15dB_1k',{'h_hat_DL': predictions_h})
# io.savemat('./Testmodel10_copy3/15 dB/h_hat_DL_15dB_0.5k',{'h_hat_DL': predictions_h})




# ###predictions_H = np.reshape(predictions_h, [test_num, Ms, Gs, 2])


# h = true_h
# h_hat = predictions_h

# pruning_thrld = 1e-01

# for r in range(h.shape[0]):
#     for c in range(h.shape[1]):
#         if np.abs(h[r, c]) < pruning_thrld:
#             h[r, c] = 0

# for r in range(h_hat.shape[0]):
#     for c in range(h_hat.shape[1]):
#         if np.abs(h_hat[r, c]) < pruning_thrld:
#             h_hat[r, c] = 0

# abs_h = np.abs(h)
# abs_h_hat = np.abs(h_hat)

# error = np.mean((np.abs(h_hat - h)**2), axis=1)
# mse = np.mean(error)
# print(mse)


# %%


# h = h_list
# h = h[:, :, 0] + 1j*h[:, :, 1]


# pruning_thrld = 0.00001
# h = np.where(np.abs(h) < pruning_thrld, np.zeros_like(h), h)

# abs_h = abs(h)

# column = abs_h[:,32]
# non_zero_values = column[column != 0]
# column_min = np.min(non_zero_values)
# print(column_min)
