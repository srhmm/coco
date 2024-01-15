import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

from util import _h
from data_gen import DataGen

# latent_dim = 64
# tf.keras.backend.set_floatx('float64')

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='sigmoid'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class MatMulLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, latent_dim=0):
        super(MatMulLayer, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def build(self, input_shape):
        self.kernel = self.add_weight("A", shape=[self.input_dim, self.input_dim])

    def call(self, input):
        I = tf.eye(self.input_dim)
        M = I - self.kernel
        return tf.matmul(input, M)

class CustomMatMulLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, latent_dim=0):
        super(CustomMatMulLayer, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def build(self, input_shape):
        self.kernel = self.add_weight("B", shape=[self.latent_dim, self.input_dim])

    def call(self, input):
        I = tf.eye(self.input_dim)
        B = self.kernel
        x = tf.split(input, num_or_size_splits=[self.input_dim, self.latent_dim], axis=1)
        return tf.matmul(x[0], I) + tf.matmul(x[1], B)

class InverseMatMulLayer(tf.keras.layers.Layer):
    def __init__(self, dim, layer):
        super(InverseMatMulLayer, self).__init__()
        self.dim = dim
        self.layer = layer

    def build(self, input_shape):
        self.kernel = self.layer.get_weights()[0]

    def call(self, input):
        I = tf.eye(self.dim)
        M = tf.linalg.inv(I - self.kernel)
        return tf.matmul(input, M)


class ConfoundedAE(Model):
    def __init__(self, input_dim, latent_dim):
        super(ConfoundedAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.define_encoder()
        self.define_decoder()

    def define_encoder(self):
        self.ML = MatMulLayer(self.input_dim)
        self.encoder = tf.keras.Sequential([
            tf.keras.Input(shape=(self.input_dim,)),
            layers.Dense(self.input_dim, activation='relu'),
            self.ML,
            layers.Dense(2*self.input_dim + 2*self.latent_dim, activation='linear')
        ])

        self.A = self.ML.kernel

    def define_decoder(self):
        self.CML = CustomMatMulLayer(self.input_dim, self.latent_dim)
        self.decoder = tf.keras.Sequential([
            tf.keras.Input(shape=(self.input_dim+self.latent_dim,)),
            self.CML,
            InverseMatMulLayer(self.input_dim, self.ML),
            layers.Dense(self.input_dim+self.input_dim, activation='relu')
        ])

        self.B = self.CML.kernel


    def sample(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        z = mu + tf.math.multiply(tf.exp(logvar*.5), eps) 
        return z

    def encode(self, x):
        enc = self.encoder(x)
        mu_e, logvar_e, mu_z, logvar_z = tf.split(enc, [self.input_dim, self.input_dim, self.latent_dim, self.latent_dim], axis=1)
        e = self.sample(mu_e, logvar_e)
        z = self.sample(mu_z, logvar_z)

        return mu_e, logvar_e, mu_z, logvar_z, e, z

    def decode(self, z):
        dec = self.decoder(z)
        mu, logvar = tf.split(dec, [self.input_dim, self.input_dim], axis=1)

        xh = self.sample(mu, logvar)

        return mu, logvar, xh

    def call(self, x):
        _, _, _, _, e, z = self.encode(x)
        ez = tf.concat([e, z], axis=1)
        _, _, xh = self.decode(ez)

        return xh

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    score = tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
                          axis=raxis)
    return score


def score(model, x):
    mu_e, logvar_e, mu_z, logvar_z, e, z = model.encode(x)

    logqz_x = log_normal_pdf(z, mu_z, logvar_z)
    logpz = log_normal_pdf(z, 0., 0.)

    logqe_x = log_normal_pdf(e, mu_e, logvar_e)
    logpe = log_normal_pdf(e, 0., 0.)

    logq = logqz_x + logqe_x
    logpez = logpz + logpe

    ez = tf.concat([e, z], axis=1)
    mu_x, logvar_x, xh = model.decode(ez)

    logpx_z = log_normal_pdf(xh, mu_x, logvar_x)

    return -tf.reduce_mean(logpx_z + logpez - logq)

optimizer = tf.keras.optimizers.Adam(1e-4)


def train_step(model, x, optimizer, lam_A=0, c_A=1, tau_A=0, tau_B=0):
    A = model.A
    B = model.B
    m = A.shape[0]

    with tf.GradientTape() as tape:

        # Pure VAE score
        _loss = score(model, x)


        # Sparse B
        loss = _loss + tau_B * tf.norm(B, 1)

        # Sparse A
        loss += tau_A * tf.norm(A, 1)

        # h(A)
        h_A = _h(A, m)
        loss += lam_A * h_A + 0.5 * c_A * h_A * h_A + 100.*tf.linalg.trace(A*A)

    if not np.isnan(loss.numpy()):
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    else:
      print('skipping nan loss batch')

    return h_A, _loss, loss

def train(model, x):
    epochs = 300
    k_max_iter = 100

    h_A_old = np.inf
    c_A = 1
    lam_A = 0
    tau_A = 10
    tau_B = 10
    h_tol = 1e-8
    best_loss = np.inf

    for i in range(k_max_iter):
      while c_A < 1e20:
        for _ in range(epochs):
          idx = np.arange(1000)
          np.random.shuffle(idx)
          _x = x[idx[:32], :]
          h_A, _loss, loss = train_step(model, _x, optimizer, lam_A, c_A, tau_A, tau_B)
          if loss < best_loss:
            best_A = model.A
            best_B = model.B
            best_ca = c_A
            best_la = lam_A
          last_A = model.A
          last_B = model.B

        if h_A > 0.25 * h_A_old:
          c_A *= 10
        else:
          break

      h_A_old = h_A
      lam_A += c_A * h_A

      if h_A < h_tol:
        break

      if not (i % 50):
        print(score(model, x))
    return best_A, best_B


if __name__ == '__main__':
    m = 10
    c = ConfoundedAE(m, 1)
    dg = DataGen(1000, m, frac_confounded=0.3, alpha=8)
    dg.net()
    x = dg.data()

    train(c, x)



