import tensorflow as tf
from loss import Loss

class WeightedSparseSoftmaxCrossEntropy(Loss):
    def __init__(self, class_weights):
        self.class_weights = class_weights
    
    def __call__(self, logits, labels):
        weights = tf.gather(self.class_weights, tf.cast(labels, tf.int64))
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.cast(labels, tf.int64)
        )
        loss_op = tf.reduce_mean(weighted_loss)
        return loss_op