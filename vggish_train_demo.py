
from __future__ import print_function

from random import shuffle

import numpy as np
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_slim

import librosa
import os
import utilities as util

import tflearn

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_integer(
	'num_batches', 30,
	'Number of batches of examples to feed into the model. Each batch is of '
	'variable size and contains shuffled examples of each class of audio.')

flags.DEFINE_boolean(
	'train_vggish', True,
	'If Frue, allow VGGish parameters to change during training, thus '
	'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
	'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
	'checkpoint', 'vggish_model.ckpt',
	'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
	'dataset', 'path to dataset', 'path to dataset'
)

FLAGS = flags.FLAGS

_NUM_CLASSES = 3


def _get_examples_batch():
	"""Returns a shuffled batch of examples of all audio classes.

	Note that this is just a toy function because this is a simple demo intended
	to illustrate how the training code might work.

	Returns:
	a tuple (features, labels) where features is a NumPy array of shape
	[batch_size, num_frames, num_bands] where the batch_size is variable and
	each row is a log mel spectrogram patch of shape [num_frames, num_bands]
	suitable for feeding VGGish, while labels is a NumPy array of shape
	[batch_size, num_classes] where each row is a multi-hot label vector that
	provides the labels for corresponding rows in features.
	"""
	# Make a waveform for each class.
	num_seconds = 5
	sr = 44100  # Sampling rate.
	t = np.linspace(0, num_seconds, int(num_seconds * sr))  # Time axis.
	# Random sine wave.
	freq = np.random.uniform(100, 1000)
	sine = np.sin(2 * np.pi * freq * t)
	# Random constant signal.
	magnitude = np.random.uniform(-1, 1)
	const = magnitude * t
	# White noise.
	noise = np.random.normal(-1, 1, size=t.shape)

	# Make examples of each signal and corresponding labels.
	# Sine is class index 0, Const class index 1, Noise class index 2.
	sine_examples = vggish_input.waveform_to_examples(sine, sr)
	sine_labels = np.array([[1, 0, 0]] * sine_examples.shape[0])
	const_examples = vggish_input.waveform_to_examples(const, sr)
	const_labels = np.array([[0, 1, 0]] * const_examples.shape[0])
	noise_examples = vggish_input.waveform_to_examples(noise, sr)
	noise_labels = np.array([[0, 0, 1]] * noise_examples.shape[0])

	# Shuffle (example, label) pairs across all classes.
	all_examples = np.concatenate((sine_examples, const_examples, noise_examples))
	all_labels = np.concatenate((sine_labels, const_labels, noise_labels))
	labeled_examples = list(zip(all_examples, all_labels))
	shuffle(labeled_examples)

	# Separate and return the features and labels.
	features = [example for (example, _) in labeled_examples]
	labels = [label for (_, label) in labeled_examples]
	return (features, labels)

def get_data(path_to_dataset):
	classes = np.sort([c for c in os.listdir(path_to_dataset) if os.path.isdir(os.path.join(path_to_dataset, c))] )
	print("Classes: ", classes)

	labels = {}
	for i in range(len(classes)):
		labels[classes[i]]=i

	files_lists = len(labels)*[None]
	for i, c in enumerate(classes):
		files_lists[i], _ = util.list_files(os.path.join(path_to_dataset, c))

	Xdatas, Ylabels = [], []
	for i, fl in enumerate(files_lists):
		label = [0]*len(classes)
		label[i] = 1
		for f in fl:
			print('load file', f)
			r_data, sr = librosa.load(f)
			data = vggish_input.waveform_to_examples(r_data, sr)
			Xdatas.append(data)
			Ylabels.append(label)
	return Xdatas, Ylabels


def main(_):
	with tf.Graph().as_default(), tf.Session() as sess:
	# Define VGGish.
		embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)

	# Define a shallow classification model and associated training ops on top
	# of VGGish.
		with tf.variable_scope('mymodel'):
				# Add a fully connected layer with 100 units.
				num_units = 100
				fc = slim.fully_connected(embeddings, num_units)

				# Add a classifier layer at the end, consisting of parallel logistic
				# classifiers, one per class. This allows for multi-class tasks.
				logits = slim.fully_connected(
						fc, _NUM_CLASSES, activation_fn=None, scope='logits')
				tf.sigmoid(logits, name='prediction')

				# Add training ops.
				with tf.variable_scope('train'):
						global_step = tf.Variable(
								0, name='global_step', trainable=False,
								collections=[tf.GraphKeys.GLOBAL_VARIABLES,
																tf.GraphKeys.GLOBAL_STEP])

						# Labels are assumed to be fed as a batch multi-hot vectors, with
						# a 1 in the position of each positive class label, and 0 elsewhere.
						labels = tf.placeholder(
								tf.float32, shape=(None, _NUM_CLASSES), name='labels')

						# Cross-entropy label loss.
						xent = tf.nn.sigmoid_cross_entropy_with_logits(
								logits=logits, labels=labels, name='xent')
						loss = tf.reduce_mean(xent, name='loss_op')
						tf.summary.scalar('loss', loss)

						# We use the same optimizer and hyperparameters as used to train VGGish.
						optimizer = tf.train.AdamOptimizer(
								learning_rate=vggish_params.LEARNING_RATE,
								epsilon=vggish_params.ADAM_EPSILON)
						optimizer.minimize(loss, global_step=global_step, name='train_op')

		# Initialize all variables in the model, and then load the pre-trained
		# VGGish checkpoint.
		sess.run(tf.global_variables_initializer())
		vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

		# Locate all the tensors and ops we need for the training loop.
		features_tensor = sess.graph.get_tensor_by_name(
				vggish_params.INPUT_TENSOR_NAME)
		labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
		global_step_tensor = sess.graph.get_tensor_by_name(
				'mymodel/train/global_step:0')
		loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
		train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

		# The training loop.
		for _ in range(FLAGS.num_batches):
				(features, labels) = _get_examples_batch()
				[num_steps, loss, _] = sess.run(
						[global_step_tensor, loss_tensor, train_op],
						feed_dict={features_tensor: features, labels_tensor: labels})
				print('Step %d: loss %g' % (num_steps, loss))

def test_run():
	# X, Y = get_data(FLAGS.dataset)
	# np.save('Xdatas.npy', X)
	# np.save('Ylables.npy', Y)
	X = np.load('Xdatas.npy')
	Y = np.load('Ylabels.npy')
	with tf.Graph().as_default(), tf.Session() as sess:
		embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)
		with tf.variable_scope('mymodel'):
			num_units = 100
			fc = slim.fully_connected(embeddings, num_units)

			logits = slim.fully_connected(fc, _NUM_CLASSES, 
							activation_fn=None, scope='logits')
			tf.sigmoid(logits, name='prediction')

			with tf.variable_op_scope('train'):
				global_step = tf.Variable(0, name='global_step', trainable=False,
						collections=[tf.GraphKeys.GLOBAL_VARIABLES,
									tf.GraphKeys.GLOBAL_STEP])
				
				labels = tf.placeholder(tf.float32, shape=(None, _NUM_CLASSES), name='labels')

				# Cross-entropy label loss
				xent = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=logits, labels=labels, name='xent')
				loss = tf.reduce_mean(xent, name='loss_op')
				tf.summary.scalar('loss', loss)

				optimizer = tf.train.AdamOptimizer(
					learning_rate=vggish_params.LEARNING_RATE,
					epsilon=vggish_params.ADAM_EPSILON)
				optimizer.minimize(loss, global_step=global_step, name='train_op')

		features_tensor = sess.graph.get_tensor_by_name(
			vggish_params.INPUT_TENSOR_NAME)
		labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')

		accuracy = tf.reduce_mean(
			tf.cast(tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels_tensor, 1)), 
			tf.float32), name='acc'
		)
		
		trainOP = tflearn.TrainOp(loss=loss, optimizer=optimizer,
			metric=accuracy, batch_size=128)
		trainer = tflearn.Trainer(train_ops=trainOP, tensorboard_verbose=0, 
			tensorboard_dir='./logs', best_checkpoint_path='./out_model/vggish_model',
			session=sess)
		trainer.fit({features_tensor:X, labels_tensor:Y}, n_epoch=1000, 
		val_feed_dicts=0.1, shuffle_all=True)


if __name__ == '__main__':
#   tf.app.run()
		test_run()