import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv3D,Reshape,Lambda,Cropping3D,Cropping2D,ZeroPadding3D,Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

def circular_padding_2D(x, kernel_size, strides):
    in_height = int(x.get_shape()[1])
    in_width = int(x.get_shape()[2])
    pad_along_height = max(kernel_size - strides, 0)
    pad_along_width = max(kernel_size - strides, 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    # top and bottom side padding
    pad_top = Cropping3D(cropping=((in_height - pad_top, 0), (0, 0), (0, 0)))(x)
    pad_bottom = Cropping3D(cropping=((0, in_height - pad_bottom), (0, 0), (0, 0)))(x)
    # add padding to incoming image
    conc = Concatenate(axis=1)([pad_top, x, pad_bottom])

    # top and bottom side padding
    pad_left = Cropping3D(cropping=((0, 0), (in_width - pad_left, 0), (0, 0)))(conc)
    pad_right = Cropping3D(cropping=((0, 0), (0, in_width - pad_right), (0, 0)))(conc)
    # add padding to incoming image
    conc = Concatenate(axis=2)([pad_left, conc, pad_right])

    return conc

def circular_padding_1d(x, kernel_size, strides):
    in_height = int(x.get_shape()[1])
    in_width = int(x.get_shape()[2])
    pad_along_height = max(kernel_size - strides, 0)
    pad_along_width = max(kernel_size - strides, 0)
   

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left


    # top and bottom side padding
    pad_top = Cropping2D(cropping=((in_height - pad_top, 0), (0, 0)))(x)
    pad_bottom = Cropping2D(cropping=((0, in_height - pad_bottom), (0, 0)))(x)

    # add padding to incoming image
    conc = Concatenate(axis=1)([pad_top, x, pad_bottom])

    pad_left = Cropping2D(cropping=((0, 0), (in_width - pad_left, 0)))(conc)
    pad_right = Cropping2D(cropping=((0, 0), (0, in_width - pad_right)))(conc)

    # add padding to incoming image
    conc = Concatenate(axis=2)([pad_left, conc, pad_right])
    
    return conc

def update_mu_Sigma_OTFS(inputs, Np2):
    Psi = tf.cast(inputs[0][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, 1], tf.complex64)
    y = tf.cast(inputs[1][:,:,0], tf.complex64) + 1j * tf.cast(inputs[1][:,:,1], tf.complex64)
    y = tf.expand_dims(y, axis= 2)    
    Gamma_init = tf.cast(inputs[2][:,:,:], tf.complex64)
    sigma_2 = tf.cast(inputs[3], tf.complex64)
    
    mu_real_list = []
    mu_imag_list = []
    diag_Sigma_real_list = []

    # for i in range(num_sc):
    Rx_PsiH = tf.multiply(Gamma_init, tf.transpose(Psi, (0, 2, 1), conjugate=True))
    tempp = sigma_2 * tf.eye(Np2, dtype=tf.complex64)
    inv = tf.linalg.inv(tf.matmul(Psi, Rx_PsiH) + tempp )
    z = tf.matmul(Rx_PsiH, inv)
    mu = tf.matmul(z, y)
    diag_Sigma =  Gamma_init - tf.expand_dims((tf.reduce_sum(tf.multiply(z, tf.math.conj((Rx_PsiH))),axis=-1)), axis = 2)

    # return the updated parameters
    mu_real_list.append(tf.math.real(mu))
    mu_imag_list.append(tf.math.imag(mu))
    diag_Sigma_real_list.append(tf.math.real(diag_Sigma))

    mu_real_list = tf.concat(mu_real_list, axis=-1)
    mu_imag_list = tf.concat(mu_imag_list, axis=-1)
    diag_Sigma_real_list = tf.concat(diag_Sigma_real_list, axis=-1)   
       
    return mu_real_list, mu_imag_list, diag_Sigma_real_list

def update_mu_Sigma_ICASSP(inputs):
    
    Psi = tf.cast(inputs[0][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, 1], tf.complex64)
    y = tf.cast(inputs[1][:,:,0], tf.complex64) + 1j * tf.cast(inputs[1][:,:,1], tf.complex64)
    y = tf.expand_dims(y, axis= 2)    
    
    Gamma_init = inputs[2][:,:,:]
    B = tf.linalg.pinv(Gamma_init)
    B = tf.cast(B, tf.complex64)
    
    # Gamma_init = tf.cast(inputs[2][:,:,:], tf.complex64)

    sigma_2 = tf.cast(inputs[3], tf.complex64)
    sigma_2_ex = tf.expand_dims(sigma_2, axis = 2)


    # SBL equations as Suraj Paper
    # Sigma
    Psi_H = tf.transpose(Psi, (0, 2, 1), conjugate=True)
    Psi_H_Psi = tf.matmul(Psi_H, Psi)
    A = sigma_2_ex*Psi_H_Psi
    C = A + B
    C = tf.cast(C, tf.float32)
    Sigma = tf.linalg.pinv(C)
    diag_Sigma = tf.linalg.diag_part(Sigma)
    diag_Sigma = tf.expand_dims(diag_Sigma, axis =2)

    # mu
    Psi_H_y = tf.matmul(Psi_H,y)
    
    sigma_2_ex = tf.cast(sigma_2_ex, tf.float32)
    Psi_H_y = tf.cast(Psi_H_y, tf.float32)
    mu = sigma_2_ex*(tf.matmul(Sigma,Psi_H_y))


    # Extracting the real and imaginary parts
    mu_real = tf.math.real(mu)
    mu_imag = tf.math.imag(mu)
    diag_Sigma_real = tf.math.real(diag_Sigma)
    
    return mu_real, mu_imag, diag_Sigma_real