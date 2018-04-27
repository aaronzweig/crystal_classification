from gcn.layers import *
from gcn.metrics import *

from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS

A = [0, 35, 70, 105, 140, 175, 210]
B = [34, 69, 104, 139, 174, 209, 224]

def reconstruct_graph(emb, normalize = True):
    embT = tf.transpose(emb, [0, 2, 1])
    graph = tf.matmul(emb, embT)
    if normalize:
      graph = tf.nn.sigmoid(graph)
      d = tf.reduce_sum(graph, 1)
      d = tf.pow(d, -0.5)
      graph = tf.expand_dims(d, 1) * graph * tf.expand_dims(d, 2)
    return graph

def cube(m):
    square = tf.matmul(m, m)
    return tf.matmul(m, square)

def kl(mean, log_std):
    return 0.5 * tf.reduce_sum(1 + 2 * log_std - tf.square(mean) - tf.square(tf.exp(log_std)), 1)

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def fit(self):
        pass

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

class GraphiteGenModel(Model):
    def __init__(self, placeholders, num_features, n_samples, **kwargs):
        super(GraphiteGenModel, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.n_samples = n_samples
        self.placeholders = placeholders
        self.weight_norm = 0

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def encoder(self, inputs):

        hidden = GlobalGraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden3,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            logging=self.logging)(inputs)

        self.z_mean = GlobalGraphConvolution(input_dim=FLAGS.hidden3,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            logging=self.logging)(hidden)

        self.z_log_std = GlobalGraphConvolution(input_dim=FLAGS.hidden3,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            logging=self.logging)(hidden)
    def decoder_layers(self):

        self.decode_z = Dense(input_dim=FLAGS.hidden4,
                                    output_dim=FLAGS.hidden5,
                                    placeholders=self.placeholders,
                                    act=tf.nn.relu,
                                    logging=self.logging)

        self.decode_z2 = Dense(input_dim=FLAGS.hidden5,
                            output_dim=FLAGS.hidden6,
                            placeholders=self.placeholders,
                            act=lambda x: x,
                            logging=self.logging)

        self.decode_graphite = GlobalGraphite(input_dim=FLAGS.hidden4,
                                    output_dim=FLAGS.hidden5,
                                    placeholders=self.placeholders,
                                    act=tf.nn.relu,
                                    logging=self.logging)

        self.decode_graphite2 = GlobalGraphite(input_dim=FLAGS.hidden5,
                                    output_dim=FLAGS.hidden6,
                                    placeholders=self.placeholders,
                                    act=lambda x: x,
                                    logging=self.logging)        

    def decode(self, z):
        hidden = self.decode_z(z)
        emb = self.decode_z2(hidden)
        graph = reconstruct_graph(emb, normalize = False)

        # hidden = self.decode_graphite((emb, graph))
        # new_emb = self.decode_graphite2((hidden, graph))
        # emb = (1 - FLAGS.autoregressive_scalar) * emb + FLAGS.autoregressive_scalar * new_emb
        # graph = reconstruct_graph(emb)

        # hidden = self.decode_graphite((emb, graph))
        # new_emb = self.decode_graphite2((hidden, graph))
        # emb = (1 - FLAGS.autoregressive_scalar) * emb + FLAGS.autoregressive_scalar * new_emb
        # graph = reconstruct_graph(emb, normalize = False)

        return graph

    def _build(self):
  
        self.encoder(self.inputs)
        self.decoder_layers()
        z = self.z_mean + tf.random_normal(tf.shape(self.z_mean)) * tf.exp(self.z_log_std)
        self.reconstruction = self.decode(z)

    def _loss(self):
        logits = self.reconstruction
        labels = self.placeholders['adj_orig']

        self.log_lik = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))

        neg_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = logits, targets = labels, pos_weight = 0), [1,2])
        pos_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = logits, targets = labels, pos_weight = 1), [1,2]) - neg_loss

        pos_totals = 1.0 * tf.reduce_sum(labels, [1,2])
        neg_totals = 1.0 * tf.reduce_sum(1 - labels, [1,2])

        pos_scale = neg_totals / (pos_totals + neg_totals)
        neg_scale = pos_totals / (pos_totals + neg_totals)

        self.loss = tf.reduce_mean(pos_scale * pos_loss + neg_scale * neg_loss)

        # degrees = tf.reduce_sum(tf.nn.sigmoid(logits), 1)
        # total = tf.nn.relu(degrees - 8)
        # self.loss += tf.reduce_mean(total)

        if FLAGS.VAE:
            self.log_lik -= (1.0 / self.n_samples) * tf.reduce_mean(kl(self.z_mean, self.z_log_std))
            self.loss -= (1.0 / self.n_samples) * tf.reduce_mean(kl(self.z_mean, self.z_log_std))

        self.log_lik *= -1.0 * self.n_samples * self.n_samples

    def sample(self, count):
        z = tf.random_normal([count, self.n_samples, FLAGS.hidden4])
        reconstruction = tf.nn.sigmoid(self.decode(z))
        return tf.round(reconstruction)

    def sample_fair(self, count):
        z = tf.random_normal([count, self.n_samples, FLAGS.hidden4])
        reconstruction = tf.nn.sigmoid(self.decode(z))
        random = tf.random_uniform(reconstruction.shape)
        condition = tf.greater(reconstruction, random)
        return tf.where(condition, tf.ones_like(random), tf.zeros_like(random))

    def _accuracy(self):
        self.accuracy = tf.reduce_mean(tf.zeros(1))
