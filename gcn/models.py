from gcn.layers import *
from gcn.metrics import *

from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS

A = [0, 35, 70, 105, 140, 175, 210]
B = [34, 69, 104, 139, 174, 209, 224]

def reconstruct_graph(emb, normalize = True):
    embT = tf.transpose(emb)
    graph = tf.matmul(emb, embT)
    if normalize:
      graph = tf.nn.sigmoid(graph)
      d = tf.reduce_sum(graph, 1)
      d = tf.pow(d, -0.5)
      graph = tf.expand_dims(d, 0) * graph * tf.expand_dims(d, 1)
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

        hidden = GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden3,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            logging=self.logging)(inputs)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden3,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            logging=self.logging)(hidden)

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden3,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            logging=self.logging)(hidden)
    def decoder_layers(self):

        self.decode_z = Dense2D(input_dim=FLAGS.hidden4,
                                    output_dim=FLAGS.hidden5,
                                    placeholders=self.placeholders,
                                    act=tf.nn.relu,
                                    logging=self.logging)

        self.decode_z2 = Dense2D(input_dim=FLAGS.hidden5,
                            output_dim=FLAGS.hidden6,
                            placeholders=self.placeholders,
                            act=lambda x: x,
                            logging=self.logging)

        self.decode_edges = Dense2D(input_dim=FLAGS.hidden6,
                                    output_dim=FLAGS.hidden7,
                                    placeholders=self.placeholders,
                                    act=tf.nn.relu,
                                    bias=True,
                                    logging=self.logging)

        self.decode_edges2 = Dense2D(input_dim=FLAGS.hidden7,
                            output_dim=1,
                            placeholders=self.placeholders,
                            act=lambda x: x,
                            bias=True,
                            logging=self.logging)

        # self.decode_graphite = GlobalGraphite(input_dim=FLAGS.hidden4,
        #                             output_dim=FLAGS.hidden5,
        #                             placeholders=self.placeholders,
        #                             act=tf.nn.relu,
        #                             logging=self.logging)

        # self.decode_graphite2 = GlobalGraphite(input_dim=FLAGS.hidden5,
        #                             output_dim=FLAGS.hidden6,
        #                             placeholders=self.placeholders,
        #                             act=lambda x: x,
        #                             logging=self.logging)        

    def reconstruct_relnet(self, emb, normalize = True):
        edges = tf.expand_dims(emb, 0) + tf.expand_dims(emb, 1)
        edges = tf.reshape(edges, [-1, FLAGS.hidden6])
        edges = self.decode_edges(edges)
        edges = self.decode_edges2(edges)
        graph = tf.reshape(edges, [self.n_samples, self.n_samples])
        if normalize:
            graph = tf.nn.sigmoid(graph)
        return graph

    def decode(self, z):
        hidden = self.decode_z(z)
        emb = self.decode_z2(hidden)
        graph = self.reconstruct_relnet(emb, normalize = False)

        # hidden = self.decode_graphite((emb, graph))
        # new_emb = self.decode_graphite2((hidden, graph))
        # emb = (1 - FLAGS.autoregressive_scalar) * emb + FLAGS.autoregressive_scalar * new_emb
        # graph = reconstruct_graph(emb)

        return graph

    def _build(self):
  
        self.encoder(self.inputs)
        self.decoder_layers()
        z = self.z_mean + tf.random_normal(tf.shape(self.z_mean)) * tf.exp(self.z_log_std)
        self.reconstruction = self.decode(z)

    def _loss(self):
        logits = self.reconstruction * (1 - tf.eye(self.n_samples))
        labels = self.placeholders['adj_orig'] * (1 - tf.eye(self.n_samples))

        self.log_lik = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))

        pos_count = tf.reduce_sum(labels) + 0.01
        norm = self.n_samples * self.n_samples / ((1.0 * self.n_samples * self.n_samples - pos_count) * 2)
        pos_weight = (1.0 * self.n_samples * self.n_samples - pos_count) / pos_count
        self.loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=pos_weight))

        # degrees = tf.reduce_sum(tf.nn.sigmoid(logits), 1)
        # total = tf.nn.relu(degrees - 8)
        # self.loss += tf.reduce_mean(total)

        self.log_lik -= (1.0 / self.n_samples) * tf.reduce_mean(kl(self.z_mean, self.z_log_std))
        self.loss -= (1.0 / self.n_samples) * tf.reduce_mean(kl(self.z_mean, self.z_log_std))

        self.log_lik *= -1.0 * self.n_samples * self.n_samples

    def sample(self):
        z = tf.random_normal([self.n_samples, FLAGS.hidden4])
        reconstruction = tf.nn.sigmoid(self.decode(z))
        return tf.round(reconstruction)

    def _accuracy(self):
        self.accuracy = tf.reduce_mean(tf.zeros(1))
