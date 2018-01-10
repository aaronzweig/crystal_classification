import tensorflow as tf

def masked_weighted_cross_entropy(reconstruction, adj_orig, mask):
    edges_pos = tf.reduce_sum(adj_orig, [1,2])
    edges_neg = tf.reduce_sum(1 - adj_orig, [1,2])

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = reconstruction, labels = adj_orig)
    loss_pos = tf.reduce_sum(loss * adj_orig, [1,2])
    loss_neg = tf.reduce_sum(loss * (1 - adj_orig), [1,2])
    loss = loss_pos / edges_pos + loss_neg / edges_neg

    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_softmax_cross_entropy(preds, labels, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)