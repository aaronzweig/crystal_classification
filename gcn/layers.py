from gcn.inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = placeholders['dropout']
        self.act = act
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj_norm = placeholders['adj_norm']
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim], name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)

        vertex_count = int(self.adj_norm.get_shape()[1])

        x = tf.reshape(x, [-1, self.input_dim])
        output = tf.matmul(x, self.vars['weights'])
        output = tf.reshape(output, [-1, vertex_count, self.output_dim])

        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class Dense2D(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense2D, self).__init__(**kwargs)

        self.dropout = placeholders['dropout']
        self.act = act
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj_norm = placeholders['adj_norm']
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim], name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)

        output = tf.matmul(x, self.vars['weights'])

        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class Mean(Layer):
    def __init__(self, axis, **kwargs):
        super(Mean, self).__init__(**kwargs)
        self.axis = axis

    def _call(self, inputs):
        return tf.reduce_mean(inputs, self.axis)

class Max(Layer):
    def __init__(self, axis, **kwargs):
        super(Max, self).__init__(**kwargs)
        self.axis = axis

    def _call(self, inputs):
        return tf.reduce_max(inputs, self.axis)

class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = placeholders['dropout']
        self.act = act
        self.adj_norm = placeholders['adj_norm']
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim], name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)

        pre_sup = tf.matmul(x, self.vars['weights'])
        output = tf.matmul(self.adj_norm, pre_sup)

        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GlobalGraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(GlobalGraphConvolution, self).__init__(**kwargs)

        self.dropout = placeholders['dropout']
        self.act = act
        self.adj_norm = placeholders['adj_norm']
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim], name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)

        vertex_count = int(x.get_shape()[1])

        x = tf.reshape(x, [-1, self.input_dim])
        pre_sup = tf.matmul(x, self.vars['weights'])
        pre_sup = tf.reshape(pre_sup, [-1, vertex_count, self.output_dim])
        output = tf.matmul(self.adj_norm, pre_sup)

        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GlobalGraphite(Layer):

    def __init__(self, input_dim, output_dim, placeholders, act=tf.nn.relu, bias=False, **kwargs):
        super(GlobalGraphite, self).__init__(**kwargs)

        self.dropout = placeholders['dropout']
        self.act = act
        self.adj_norm = placeholders['adj_norm']
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim], name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs[0]
        adj = inputs[1]
        x = tf.nn.dropout(x, 1-self.dropout)

        vertex_count = int(x.get_shape()[1])

        x = tf.reshape(x, [-1, self.input_dim])
        pre_sup = tf.matmul(x, self.vars['weights'])
        pre_sup = tf.reshape(pre_sup, [-1, vertex_count, self.output_dim])
        output = tf.matmul(adj, pre_sup)

        if self.bias:
            output += self.vars['bias']

        return self.act(output)