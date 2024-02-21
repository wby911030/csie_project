import numpy as np
import tensorflow as tf 

def g_logistic_loss(fake_scores):
    return tf.reduce_mean(tf.nn.softplus(-fake_scores))

def d_logistic_loss(fake_scores, real_scores):
    return tf.reduce_mean(tf.nn.softplus(fake_scores) + tf.nn.softplus(-real_scores))

def r1_regularization(real_images, D, gamma = 10):
    """
    Forward pass to compute the regularization
    :param prediction_real: (tf.Tensor) Prediction of the discriminator for a batch of real images
    :param real_sample: (tf.Tensor) Batch of the corresponding real images
    :return: (tf.Tensor) Loss value
    """
    with tf.GradientTape() as tape:
        # Calc prediction_real.sum()
        tape.watch(real_images)
        real_pred = D(real_images)
        prediction_sum = tf.reduce_sum(real_pred)

    # Calc gradient
    grad_real = tape.gradient(prediction_sum, real_images)

    # Calc regularization
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(grad_real), axis=[1, 2, 3]))

    return loss * gamma

def mode_seeking_loss(w1, w2, m1, m2):
    return np.abs(w1-w2) / np.abs(m1-m2)

class path_length_penalty(tf.Module):
    def __init__(self, beta):
        """
        Initializes the PathLengthPenalty module.

        Args:
            beta (float): The constant β used to calculate the exponential moving average a.
        """
        self.beta = beta
        self.steps = tf.Variable(0., trainable=False)
        self.exp_sum_a = tf.Variable(0., trainable=False)

    def call(self, w, G):
        """
        Calculates the Path Length Penalty.

        Args:
            w (tf.Tensor): Batch of latent vectors of shape [batch_size, d_latent].
            x (tf.Tensor): Batch of generated images of shape [batch_size, height, width, 3].

        Returns:
            tf.Tensor: Path Length Penalty.
        """

        if w[1] is None:
            with tf.GradientTape() as tape:
                tape.watch(w[0])
                x, mask = G.synthesis(w)
                image_size = x.shape[1] * x.shape[2]
                y = tf.random.normal(shape=x.shape, dtype=tf.float32)
                output = tf.reduce_sum(x * y) / tf.sqrt(tf.cast(image_size, dtype=tf.float32))
                gradients = tape.gradient(output, w[0])
            norm = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=1) / tf.cast(w[0].shape[1], dtype=tf.float32))
        else: 
            with tf.GradientTape() as tape:
                tape.watch(w)
                x, mask = G.synthesis(w)
                image_size = x.shape[1] * x.shape[2]
                y = tf.random.normal(shape=x.shape, dtype=tf.float32)
                output = tf.reduce_sum(x * y) / tf.sqrt(tf.cast(image_size, dtype=tf.float32))
                gradients = tape.gradient(output, w)
            norm = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=2) / tf.cast(w[0].shape[1], dtype=tf.float32))

        if self.steps > 0:
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            loss = tf.reduce_mean((norm - a)**2)
        else:
            loss = tf.constant(0, dtype=tf.float32)

        mean = tf.reduce_mean(norm).numpy()
        self.exp_sum_a.assign(self.exp_sum_a * self.beta + mean * (1 - self.beta))
        self.steps.assign_add(1.)

        return loss