import tensorflow as tf
from data_helpers import batch_iter

class TextCNN:
	def __init__(self,embedding_size,
					vocab_size,
					input_size,
					num_classes,
					filter_sizes,
					droupout,
					batch_size,
					learning_rate,
					num_filters,
					num_epochs,istrain=True):
		self.embedding_size=embedding_size
		self.vocab_size=vocab_size
		self.input_size=input_size
		self.num_classes=num_classes
		self.filter_sizes=filter_sizes
		self.droupout=droupout
		self.learning_rate=learning_rate
		self.num_filters=num_filters
		self.num_epochs=num_epochs
		self.batch_size=batch_size
		self.istrain=istrain
		


	def build(self):
		self.x_train = tf.placeholder(tf.int32,[None,self.input_size],name="x_train")
		self.y_train = tf.placeholder(tf.int32,[None,self.num_classes],name='y_train')
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		self.global_step=tf.Variable(0,name='global_step',trainable=False)

		with tf.name_scope('embeddings'):
			self.embeddings = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)
			x_train_embedding = tf.nn.embedding_lookup(self.embeddings, self.x_train)
			# . The result of the embedding operation is a 3-dimensional tensor of shape [None,input_size,embedding_size].
			self.x_train_embedding_expanded = tf.expand_dims(x_train_embedding, -1)
			# tensorflow的conv2d 需要的输入为 [None,input_size,embedding_size,1] == [batch, in_height, in_width, in_channels] 其中 1表示1个通道

		#采用不同的filter，然后合并为一个大的feature vector
		# Create a convolution + maxpool layer for each filter size
		pooled_outputs=[]
		for i,filter_size in enumerate(self.filter_sizes):
			with tf.name_scope('conv-maxpool-%s'%filter_size):
				filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
				W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1))
				#  [filter_height, filter_width, in_channels（1个通道）, out_channels]
				b=tf.Variable(tf.constant(0.1,shape=[self.num_filters]))
				conv=tf.nn.conv2d(
							input=self.x_train_embedding_expanded,
							filter=W,
							strides=[1,1,1,1],
							padding='VALID',
							name='conv2d'
							)
				h=tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
				pooled=tf.nn.max_pool(h,ksize=[1,self.input_size-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID', name='pool')
				pooled_outputs.append(pooled)

		num_filters_total = self.num_filters*len(self.filter_sizes)
		self.h_pool = tf.concat(pooled_outputs,3) # shape:[batch_size, 1, 1, num_filters_total] 按照第4(3)个维度连接
		self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])

		with tf.name_scope('dropout'):
			self.h_dropout=tf.nn.dropout(self.h_pool_flat,keep_prob=self.dropout_keep_prob)

		with tf.name_scope('output'):
			W = tf.Variable(tf.truncated_normal([num_filters_total,self.num_classes]))
			b = tf.Variable(tf.constant(0.1,shape=[self.num_classes]))
			self.scores=tf.nn.xw_plus_b(self.h_dropout,W,b,name='scores')
			self.predictions=tf.argmax(self.scores,1,name='predictions')

		with tf.name_scope('loss'):
			losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y_train)
			self.loss=tf.reduce_mean(losses)

		with tf.name_scope('accuracy'):
			correct_predictions=tf.equal(self.predictions,tf.argmax(self.y_train, 1))
			self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

		if self.train:
			optimizer=tf.train.AdamOptimizer(self.learning_rate)
			grads_and_vars=optimizer.compute_gradients(self.loss)
			#返回A list of (gradient, variable) pairs
			self.train_op=optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
			# global_step可以理解为调用train_op的次数




	def train(self,dataset):
		
		save_dir='cnnModel/'
		g=tf.Graph()
		with g.as_default():
			self.build()
			self.saver = tf.train.Saver()
			init = tf.global_variables_initializer()
			sess = tf.Session()
			with sess.as_default():
				sess.run(init)
				for batch in batch_iter(data=dataset, batch_size=self.batch_size, num_epochs=self.num_epochs):
					x_train=[x[0] for x in batch]
					y_train=[x[1] for x in batch]
					length=[x[2] for x in batch]
		
					feed_dict = {
							self.x_train: x_train,
							self.y_train: y_train,
							self.dropout_keep_prob: self.droupout
							}
					_, step, loss, accuracy = sess.run(
					[self.train_op, self.global_step, self.loss, self.accuracy],
					feed_dict)
					if step%100 == 0:
						print('train step:{} loss:{} accuracy:{}'.format(step,loss,accuracy))
				self.saver.save(sess,save_dir,global_step=step)


	def test(self,dateset):
		g=tf.Graph()
		with g.as_default():
			self.build()
			self.saver = tf.train.Saver()
			save_dir='cnnModel/'
			sess = tf.Session()
			self.saver.restore(sess,save_dir)
			with sess.as_default():
				for batch in batch_iter(data=dateset, batch_size=self.batch_size, num_epochs=1,shuffle=False):
					x_train=[x[0] for x in batch]
					y_train=[x[1] for x in batch]
					length=[x[2] for x in batch]
		
					feed_dict = {
							self.x_train: x_train,
							self.y_train: y_train,
							self.dropout_keep_prob: 1
							}
					_, _, loss, accuracy = sess.run(
					[self.train_op, self.global_step, self.loss, self.accuracy],
					feed_dict)
					
					print('test loss:{} accuracy:{}'.format(loss,accuracy))
