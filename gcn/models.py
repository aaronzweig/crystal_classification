from gcn.layers import *
from gcn.metrics import *

from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS

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

class GraphiteModel(Model):
    def __init__(self, placeholders, num_features, n_samples, **kwargs):
        super(GraphiteModel, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.n_samples = n_samples
        self.placeholders = placeholders
        self.weight_norm = 0

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def encoder(self, inputs):

        hidden1 = Dense(input_dim=self.input_dim,
                                    output_dim=FLAGS.hidden1,
                                    placeholders=self.placeholders,
                                    act=tf.nn.relu,
                                    logging=self.logging)(inputs)

        hidden2 = Dense(input_dim=FLAGS.hidden1,
                                    output_dim=FLAGS.hidden2,
                                    placeholders=self.placeholders,
                                    act=tf.nn.relu,
                                    logging=self.logging)(hidden1)

        hidden3 = GlobalGraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.hidden3,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            logging=self.logging)(hidden2)

        self.z = GlobalGraphConvolution(input_dim=FLAGS.hidden3,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            logging=self.logging)(hidden3)

    def _build(self):
  
        self.encoder(self.inputs)

        hidden4 = tf.concat((Mean(1)(self.z), Max(1)(self.z)), axis = 1)
        hidden5 = Dense2D(input_dim=2 * FLAGS.hidden4,
                                    output_dim=FLAGS.hidden5,
                                    placeholders=self.placeholders,
                                    act=tf.nn.relu,
                                    logging=self.logging)(hidden4)
        self.outputs = Dense2D(input_dim=FLAGS.hidden5,
                                    output_dim=self.output_dim,
                                    placeholders=self.placeholders,
                                    act=lambda x: x,
                                    logging=self.logging)(hidden5)

    def _loss(self):
        #self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])


