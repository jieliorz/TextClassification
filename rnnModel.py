import tensorflow as tf
from data_helpers import batch_iter

class TextRnn:
	def __init__(self,embedding_size,
					vocab_size,
					input_size,
					num_classes,
					droupout,
					batch_size,
					learning_rate,
					hidden_size,
					num_epochs,istrain=True):
		self.embedding_size=embedding_size
		self.vocab_size=vocab_size
		self.input_size=input_size
		self.num_classes=num_classes
		self.droupout=droupout
		self.learning_rate=learning_rate
		self.num_epochs=num_epochs
		self.batch_size=batch_size
		self.hidden_size=hidden_size
		self.istrain=istrain
		


	def build(self):
		self.x_train = tf.placeholder(tf.int32,[None,self.input_size],name="x_train")
		self.y_train = tf.placeholder(tf.int32,[None,self.num_classes],name='y_train')
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		self.seq_lengths= tf.placeholder(tf.int32,[None])
		self.global_step=tf.Variable(0,name='global_step',trainable=False)

		with tf.name_scope('embeddings'):
			self.embeddings = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)
			self.x_train_embedding = tf.nn.embedding_lookup(self.embeddings, self.x_train)
			# . The result of the embedding operation is a 3-dimensional tensor of shape [None,input_size,embedding_size].

		with tf.name_scope('rnn'):
			lstm_fw_cell=tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size) #forward direction cell
			lstm_bw_cell=tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size) #backward direction cell		
			if self.istrain:
				lstm_fw_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
				lstm_bw_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)

			outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,self.x_train_embedding,sequence_length=self.seq_lengths,dtype=tf.float32)
			#[batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
			output_rnn=tf.concat(outputs,axis=2)#[batch_size,sequence_length,hidden_size*2] 两个tensor按照第2个维度（hidden size）连接
			self.final_outputs=tf.reduce_sum(output_rnn,axis=1)#shape=[batch_size,2*hidden_size]按维度1(即senquence length)相加

		with tf.name_scope('output'):
			W = tf.Variable(tf.truncated_normal([2*self.hidden_size,self.num_classes]))
			b = tf.Variable(tf.constant(0.1,shape=[self.num_classes]))
			self.scores=tf.nn.xw_plus_b(self.final_outputs,W,b,name='scores')
			self.predictions=tf.argmax(self.scores,1,name='predictions')

		with tf.name_scope('loss'):
			losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y_train)
			self.loss=tf.reduce_mean(losses)

		with tf.name_scope('accuracy'):
			correct_predictions=tf.equal(self.predictions,tf.argmax(self.y_train, 1))
			self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

		if self.istrain:
			optimizer=tf.train.AdamOptimizer(self.learning_rate)
			grads_and_vars=optimizer.compute_gradients(self.loss)
			#返回A list of (gradient, variable) pairs
			self.train_op=optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
			# global_step可以理解为调用train_op的次数




	def train(self,dataset):
		
		save_dir='rnnModel/'
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
							self.dropout_keep_prob: self.droupout,
							self.seq_lengths:length
							}
					_, step, loss, accuracy = sess.run(
					[self.train_op, self.global_step, self.loss, self.accuracy],
					feed_dict)
					if step%100 == 0:
						print('train step:{} loss:{} accuracy:{}'.format(step,loss,accuracy))
				self.saver.save(sess,save_dir)


	def test(self,dateset):
		g=tf.Graph()
		with g.as_default():
			self.build()
			self.saver = tf.train.Saver()
			save_dir='rnnModel/'
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
							self.dropout_keep_prob: 1,
							self.seq_lengths:length
							}
					_, _, loss, accuracy = sess.run(
					[self.train_op, self.global_step, self.loss, self.accuracy],
					feed_dict)
					
					print('test loss:{} accuracy:{}'.format(loss,accuracy))
