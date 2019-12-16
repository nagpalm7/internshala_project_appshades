import tensorflow as tf

# The export path contains the name and the version of the model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

from keras import backend as K
K.set_session(sess)
K.set_learning_phase(0)

model = tf.keras.models.load_model('./parallel_model_best.h5')
export_path = './classifier/1/'

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as session:
    tf.compat.v1.saved_model.simple_save(
        session,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})