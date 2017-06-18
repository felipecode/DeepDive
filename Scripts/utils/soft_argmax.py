def soft_argmax(tf, vector, beta=1000):
    vector = beta * vector
    vector -= tf.reduce_max(vector)
    return tf.reduce_sum(tf.exp(vector) / tf.reduce_sum(tf.exp(vector)) * tf.range(vector.shape[0].value, dtype=vector.dtype))
