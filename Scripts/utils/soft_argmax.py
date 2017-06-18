def soft_argmax(tf, vector, beta=3000):
    vector = beta * vector
    vector -= tf.expand_dims(tf.reduce_max(vector, (1,)), 1)
    index_vect = tf.range(vector.shape[1].value, dtype=vector.dtype)
    return tf.reduce_sum(tf.exp(vector) / np.expand_dims(tf.reduce_sum(tf.exp(vector), (1,))) * index_vect, (1,))
