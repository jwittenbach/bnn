import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tfk.layers
tfd = tfp.distributions
tfpl = tfp.layers

def dense_normal_fn(w_loc=None, w_scale=None, b_loc=None, b_scale=None, constant_init=False):

  def make_dist_fn(w_size, b_size, dtype):
    
    idx0 = 0
    
    if w_loc is None:
      idx1 = idx0 + w_size
      make_w_loc = lambda t: t[idx0:idx1]
    else:
      idx1 = idx0
      make_w_loc = lambda t: w_size * [float(w_loc)]

    if w_scale is None:
      idx2 = idx1 + w_size
      make_w_scale = lambda t: 1e-5 + tf.nn.relu(t[idx1:idx2])
    else:
      idx2 = idx1
      make_w_scale = lambda t: w_size * [float(w_scale)]

    if b_loc is None:
      idx3 = idx2 + b_size
      make_b_loc = lambda t: t[idx2:idx3]
    else:
      idx3 = idx2
      make_b_loc = lambda t: b_size * [float(b_loc)]

    if b_scale is None:
      idx4 = idx3 + b_size
      make_b_scale = lambda t: 1e-5 + tf.nn.relu(t[idx3:idx4])
    else:
      idx4 = idx3
      make_b_scale = lambda t: b_size * [float(b_scale)]

    make_loc = lambda t: tf.concat((make_w_loc(t), make_b_loc(t)), axis=0)
    make_scale = lambda t: tf.concat((make_w_scale(t), make_b_scale(t)), axis=0)

    n_vars = idx4

    if constant_init:
      init = tfpl.BlockwiseInitializer(
        ('zeros', 'ones', 'zeros', 'ones'),
        (idx1 - idx0, idx2 - idx1, idx3 - idx2, idx4 - idx3)
      )
    else:
      init = tfpl.BlockwiseInitializer(
        ('glorot_uniform', 'ones', 'zeros', 'ones'),
        (idx1 - idx0, idx2 - idx1, idx3 - idx2, idx4 - idx3)
      )

    return tfk.Sequential([
      tfpl.VariableLayer(n_vars, initializer=init),
      tfpl.DistributionLambda(lambda t: tfd.Independent(
        tfd.Normal(loc=make_loc(t), scale=make_scale(t)),
        reinterpreted_batch_ndims=1
      ))
    ])

  return make_dist_fn


def bnn(layer_sizes, variational_params=None, prior_params=[0, 1, 0, 1], activation=tf.nn.softplus, kl_weight=1.0):
  if prior_params is None:
    prior_params = 4 * [None]
  if variational_params is None:
    variational_params = 4 * [None]

  if len(prior_params) == 2:
    prior_params = 2 * prior_params
  if len(variational_params) == 2:
    variational_params = 2 * variational_params
  
  layers = [tfkl.InputLayer(layer_sizes[0])]

  def make_layer(size, activation=activation):
    return tfpl.DenseVariational(size,
      make_posterior_fn=dense_normal_fn(*variational_params, constant_init=False),
      make_prior_fn=dense_normal_fn(*prior_params, constant_init=True),
      activation=activation,
      kl_use_exact=False,
      kl_weight=kl_weight
    )

  for l in layer_sizes[1:-1]:
    layers.append(make_layer(l))

  layers.append(make_layer(layer_sizes[-1], activation=None))

  layers.append(tfpl.DistributionLambda(lambda t: tfd.Independent(
    tfd.Normal(loc=t, scale=1.0),
    reinterpreted_batch_ndims=1
  )))

  return tfk.Sequential(layers)


class NLL(tf.keras.losses.Loss):

    # def __init__(self):
    #   super().__init__(reduction=tf.keras.losses.Reduction.SUM)

    def call(self, y_true, y_pred):
        return -y_pred.log_prob(y_true)