import numpy as np
import tensorflow as tf
from tensorflow import keras

class modulated_conv2d(keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides = 1,
                 padding = "SAME"
                 ):
    
        super().__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.padding = padding

    def build(self, input_shape): #[x, style]
        channel_axis = -1
        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[0][channel_axis]

        if input_shape[1][-1] != input_dim:
            raise ValueError('The last dimension of modulation input should be equal to input dimension.')  
             
        self.kernel = self.add_weight(name='kernel',
                                      shape=[self.kernel_size, self.kernel_size, input_dim, self.filters], #[KKIO]
                                      initializer = 'he_uniform')
    
    def call(self, input): #[x, style], style : affined latent vector from mapping network
        x, style = input
        batch_size = x.shape[0]

        #weight modulation
        w = tf.expand_dims(self.kernel, axis=0) #[BKKIO]
        s = tf.reshape(style, [batch_size, 1, 1, -1, 1]) #[BKKIO]
        w = w * s #[BKKIO]

        #weight demodulation
        sigma = tf.math.rsqrt(tf.reduce_sum(tf.square(w), axis=[1, 2, 3]) + 1e-8) #[BO]
        sigma = tf.reshape(sigma, [batch_size, 1, 1, 1, -1]) #[BKKIO]
        w = w * sigma #[BKKIO]

        #reshape examples into inputs channels?
        x = tf.expand_dims(x, axis=0) #[1BHWI]
        x = tf.transpose(x, [0, 2, 3, 1, 4])  # [1HWBI]
        x = tf.reshape(x, [1, x.shape[1], x.shape[2], -1])  # [1HW(B*I)]

        w = tf.transpose(w, [1, 2, 3, 0, 4])  # [kkIBO]
        w = tf.reshape(w, [w.shape[0], w.shape[1], w.shape[2], -1])  # [kkI(B*O)]

        #convolution
        x = tf.nn.conv2d(x, w, self.strides, self.padding, data_format="NHWC")

        #reshape output back into [BHWO]
        x = tf.reshape(x, [x.shape[1], x.shape[2], -1, self.filters])  # [HWBO]
        x = tf.transpose(x, [2, 0, 1, 3])  # [BHWO]

        return x

class synthesis_layer(keras.layers.Layer):
    def __init__(self, style_dim, filters, kernel_size = 3): #style_dim = input channels of x
        super().__init__()
        self.affine = keras.layers.Dense(style_dim, tf.nn.leaky_relu)
        self.conv = modulated_conv2d(filters, kernel_size)
        self.noise_scale = self.add_weight(name='noise_scale', shape=[], initializer='zeros', dtype=tf.float32)
        self.bias = self.add_weight(name='bias', shape=[filters], initializer='zeros')

    def call(self, input): #[x, w]  w : latent vector from mapping network
        x, w = input
        s = self.affine(w)
        x = self.conv([x, s])
        noise = tf.random.normal([x.shape[0], x.shape[1], x.shape[2], 1])
        x = x + noise * self.noise_scale
        x = x + tf.reshape(self.bias, [1, 1, 1, -1])
        x = tf.nn.leaky_relu(x)
        return x

class to_rgb(keras.layers.Layer):
    def __init__(self, style_dim, filters = 3, kernel_size = 1): #style_dim = input channels of x
        super().__init__()
        self.affine = keras.layers.Dense(style_dim, tf.nn.leaky_relu)
        self.conv = modulated_conv2d(filters, kernel_size)
        self.bias = self.add_weight(name='bias', shape=[filters], initializer='zeros')
    
    def call(self, input): #[x, w]  w : latent vector from mapping network
        x, w = input
        s = self.affine(w)
        x = self.conv([x, s])
        x = x + tf.reshape(self.bias, [1, 1, 1, -1])
        x = tf.nn.tanh(x)
        return x

class synthesis_block(keras.layers.Layer):
    def __init__(self,
                mode,               # sys_block : rgb, res_block : mask, skip
                style_dim,          # style_dim = input channels of x
                filters,
                kernel_size = 3): 
        super().__init__()
        self.mode = mode
        self.layer1 = synthesis_layer(style_dim, filters)
        self.layer2 = synthesis_layer(filters, filters)
        if self.mode == "mask":
            self.torgb = to_rgb(filters, 1)
        elif self.mode == "rgb":
            self.torgb = to_rgb(filters)
    
    def call(self, input): 
        #[x, w, res_x]  w : latent vector from mapping network, res_x : defect feature map with mask
        x, w, res_x = input
        x = self.layer1([x, w])
        x = self.layer2([x, w])
        if self.mode == "skip":
            return x
        elif self.mode == "mask":
            img = self.torgb([x, w])
            return x, img
        else:
            if res_x is not None:
                x = x + res_x
            img = self.torgb([x, w])
            return x, img

class synthesis_network(keras.Model):
    def __init__(self,
                img_resolution,             # Output image resolution.
                channel_base    = 32768,    # Overall multiplier for the number of channels.
                channel_max     = 512,      # Maximum number of channels in any layer.
                batch_size      = 8):
        super().__init__()
        self.defect = False
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        self.channels_list = [channel_max] + [min(channel_base // res, channel_max) for res in self.block_resolutions]
        self.upsample = keras.layers.UpSampling2D()

        self.constant_input = self.add_weight(name='constant_input',
                                            shape=[batch_size, 4, 4, channel_max],
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        
        for i in range(len(self.block_resolutions)):
            block = synthesis_block("rgb", self.channels_list[i], self.channels_list[i+1])
            setattr(self, f"block{self.block_resolutions[i]}", block)            
        
    def set_defect(self):
        # attach residual blocks
        # start at resolution 64
        self.defect = True
        res_block = synthesis_block("mask", self.channels_list[4], self.channels_list[4+1])
        setattr(self, f"res_block{self.block_resolutions[4]}", res_block)
        for i in range(5, len(self.block_resolutions)):
            res_block = synthesis_block("skip", self.channels_list[i], self.channels_list[i+1])
            setattr(self, f"res_block{self.block_resolutions[i]}", res_block)
    
    def call(self, input):
        # [w_object, w_defect] w : latent vector from mapping network, defect could be None
        w_object, w_defect = input
        mask = None
        img = None
        res_x = None
        x = self.constant_input
        for i in range(len(self.block_resolutions)):
            # upsampling
            if img is not None:
                img = self.upsample(img)
            if mask is not None:
                mask = self.upsample(mask)
            if i > 0:
                x = self.upsample(x)

            # residual block
            if self.defect and i >= 4:
                res_block = getattr(self, f"res_block{self.block_resolutions[i]}")
                if i == 4:
                    res_x, mask = res_block([x, w_defect, None])
                else:
                    res_x = res_block([x, w_defect, None])
            
            # apply mask
            if mask is not None:
                res_x = res_x * tf.where(mask > 0, 1.0, 0.0)
            
            # synthesis block
            block = getattr(self, f"block{self.block_resolutions[i]}")            
            x, i = block([x, w_object, res_x])
            if img is None:
                img = i
            else:
                img = (img + i) / 2.0

        return img, mask

class mapping_network(keras.Model):
    def __init__(self, num_layers = 8):
        super().__init__()
        self.num_layers = num_layers
        for i in range(self.num_layers):
            layer = keras.layers.Dense(512, tf.nn.leaky_relu)
            setattr(self, f"layer{i}", layer)
    
    def call(self, input):
        x = input
        for i in range(self.num_layers):
            layer = getattr(self, f"layer{i}")
            x = layer(x)
        return x

class generator(keras.Model):
    def __init__(self,
                img_resolution,             # Output image resolution.
                batch_size = 8):
        super().__init__()
        self.mapping = mapping_network()
        self.synthesis = synthesis_network(img_resolution, batch_size=batch_size)
    
    def set_defect(self):
        # attach residual blocks
        self.defect_mapping = mapping_network()
        self.synthesis.set_defect()

    def call(self, input):
        #[z_object, z_defect] z : random latent vectors in size [batch_size, z_dim], defect could be None
        z_object, z_defect = input
        w_object = w_defect = None
        if z_defect is not None:
            w_defect = self.defect_mapping(z_defect)
        w_object = self.mapping(z_object)
        img, mask = self.synthesis([w_object, w_defect])
        img = (img + 1.0) / 2.0    # transfer to range [0, 1]
        return w_object, w_defect, img, mask

class minibatch_std(keras.layers.Layer):
    def __init__(self, group_size = 4, n_new_features = 1):
        super().__init__()
        self.group_size = group_size
        self.n_new_features = n_new_features

    def call(self, x):
        x = tf.transpose(x, [0, 3, 1, 2])
        s = x.shape  # Tranposed input shape: [BCHW]
        y = tf.reshape(x, [self.group_size, -1, self.n_new_features, s[1]//self.n_new_features,
                           s[2], s[3]])  # [GMncHW]
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # Subtract mean over group
        y = tf.sqrt(tf.reduce_mean(tf.square(y), axis=0) + 1e-8)  # [MncHW] Calc stddev over group
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)  # [Mn111] take mean over feature maps and pixels
        y = tf.reduce_mean(y, axis=[2])  # [Mn11]
        y = tf.tile(y, [self.group_size, 1, s[2], s[3]])  # [BCHW] copy over group and pixel indices
        return tf.transpose(tf.concat([x, y], axis=1), [0, 2, 3, 1])

class discriminator_block(keras.layers.Layer):
    def __init__(self, filters, kernel_size = 3):
        super().__init__()
        self.downsample = keras.layers.AveragePooling2D()
        self.filters = filters
        self.kernel_size = kernel_size
        self.scale = 1 / np.sqrt(2)
    
    def build(self, input_shape):
        in_channel = input_shape[-1]
        self.res_conv = keras.layers.Conv2D(self.filters, 1, padding="same")
        self.conv1 = keras.layers.Conv2D(in_channel, self.kernel_size, padding="same", activation=tf.nn.leaky_relu)
        self.conv2 = keras.layers.Conv2D(self.filters, self.kernel_size, padding="same", activation=tf.nn.leaky_relu)
    
    def call(self, input):
        x = input
        res = self.downsample(x)
        res = self.res_conv(res)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.downsample(x)
        x = (x + res) * self.scale
        return x

class discriminator(keras.Model):
    def __init__(self,
                activation,                     # activation of output layer
                img_resolution,                 # Input resolution.
                channel_base        = 32768,    # Overall multiplier for the number of channels.
                channel_max         = 512,      # Maximum number of channels in any layer.):
                ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_list = [min(channel_base // res, channel_max) for res in self.block_resolutions + [4]]

        self.fromrgb = keras.layers.Conv2D(channels_list[0], 3, padding="same", activation=tf.nn.leaky_relu)

        for i in range(1, len(self.block_resolutions)):
            block = discriminator_block(channels_list[i])
            setattr(self, f"block{self.block_resolutions[i]}", block)
        
        self.out = keras.layers.Dense(1, activation=activation)

    def call(self, input):
        x = input
        x = self.fromrgb(x)
        for i in range(1, len(self.block_resolutions)):
            block = getattr(self, f"block{self.block_resolutions[i]}")
            x = block(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = self.out(x)
        return x
