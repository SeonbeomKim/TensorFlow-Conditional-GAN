#https://arxiv.org/pdf/1411.1784.pdf

import tensorflow as tf #version 1.4
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt


tensorboard_path = './tensorboard/'
saver_path = './saver/'
make_image_path = './generate/'

batch_size = 256


class GAN:

	def __init__(self, sess):
		self.noise_dim = 128 # 노이즈 차원
		self.input_dim = 784
		self.target_dim = 10
		self.hidden_dim = 256
		self.train_rate = 0.0002

		with tf.name_scope("placeholder"):
			#class 밖에서 모델 실행시킬때 학습데이터 넣어주는곳.
			self.X = tf.placeholder(tf.float32, [None, self.input_dim])
			#class 밖에서 모델 실행시킬때 class의 Generate_noise 실행한 결과를 넣어주는 곳.
			self.noise_source = tf.placeholder(tf.float32, [None, self.noise_dim])
			#target
			self.Y = tf.placeholder(tf.float32, [None, self.target_dim])
		

		with tf.name_scope("generate_image_from_noise"):
			#노이즈로 데이터 생성. 
			self.Gen = self.Generator(self.noise_source, self.Y) #batch_size, input_dim

		
		with tf.name_scope("result_from_Discriminator"):
			#학습데이터가 진짜일 확률
			self.D_X, self.D_X_logits = self.Discriminator(self.X, self.Y) #batch_size, 1
			#노이즈로부터 생성된 데이터가 진짜일 확률 
			self.D_Gen, self.D_Gen_logits = self.Discriminator(self.Gen, self.Y, True) #batch_size, 1


		with tf.name_scope("for_check_Discriminator_values"):
			#학습데이터 진짜일 확률 batch끼리 합친거. 나중에 총 데이터 셋으로 나눠줘서 평균 볼 용도.
			self.D_X_sum = tf.reduce_sum(self.D_X)
			self.D_Gen_sum = tf.reduce_sum(self.D_Gen)


		with tf.name_scope("loss"):
			#Discriminator 입장에서 최소화 해야 하는 값
			self.D_loss = self.Discriminator_loss_function(self.D_X_logits, self.D_Gen_logits)
			#Generator 입장에서 최소화 해야 하는 값.
			self.G_loss = self.Generator_loss_function(self.D_Gen_logits)


		with tf.name_scope("train"):
			#학습 코드
			self.optimizer = tf.train.AdamOptimizer(self.train_rate)
				
				#Discriminator와 Generator에서 사용된 variable 분리.
			self.D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator')
			self.G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
				
				#loss 함수 바꿨으므로 maximize가 아닌 minimize 해주는게 의미상 맞음. 즉, -1 안곱해도됨.
			self.D_minimize = self.optimizer.minimize(self.D_loss, var_list=self.D_variables) #G 변수는 고정하고 D로만 학습.
			self.G_minimize = self.optimizer.minimize(self.G_loss, var_list=self.G_variables) #D 변수는 고정하고 G로만 학습.


		with tf.name_scope("tensorboard"):
			#tensorboard
			self.D_X_tensorboard = tf.placeholder(tf.float32) #학습데이터가 진짜일 확률
			self.D_Gen_tensorboard = tf.placeholder(tf.float32) #노이즈로부터 생성된 데이터가 진짜일 확률 
			self.D_value_tensorboard = tf.placeholder(tf.float32) #Discriminator 입장에서 최소화 해야 하는 값
			self.G_value_tensorboard = tf.placeholder(tf.float32) #Generator 입장에서 최소화 해야 하는 값.

			self.D_X_summary = tf.summary.scalar("D_X", self.D_X_tensorboard) 
			self.D_Gen_summary = tf.summary.scalar("D_Gen", self.D_Gen_tensorboard) 
			self.D_value_summary = tf.summary.scalar("D_value", self.D_value_tensorboard) 
			self.G_value_summary = tf.summary.scalar("G_value", self.G_value_tensorboard) 
			
			self.merged = tf.summary.merge_all()
			self.writer = tf.summary.FileWriter(tensorboard_path, sess.graph)


		with tf.name_scope("saver"):
			self.saver = tf.train.Saver(max_to_keep=10000)


		sess.run(tf.global_variables_initializer())



	#노이즈 생성
	def Generate_noise(self, batch_size): #batch_size, nose_dim
		return np.random.normal(size=[batch_size, self.noise_dim])



	#데이터의 진짜일 확률
	def Discriminator(self, data, target, reuse=False): #batch_size, 1
		with tf.variable_scope('Discriminator') as scope:
			if reuse == True: #Descriminator 함수 두번 부르는데 두번째 부르는 때에 같은 weight를 사용하려고 함.
				scope.reuse_variables()
			
			#https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/concat
			concat = tf.concat(values=[data, target], axis=1)

			hidden = tf.layers.dense(concat, self.hidden_dim, activation=tf.nn.relu) #probability
			D_logits = tf.layers.dense(hidden, 1, activation=None) #sigmoid_cross_entropy에 사용하려고.
			D_P = tf.nn.sigmoid(D_logits) #probability #확률이므로 0~1 나오는 시그모이드 써야됨.

			return D_P, D_logits



	#노이즈로 진짜같은 데이터 생성
	def Generator(self, noise, target): #batch_size * input_dim
		with tf.variable_scope('Generator'):
			#https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/concat
			concat = tf.concat(values=[noise, target], axis=1)

			hidden = tf.layers.dense(concat, self.hidden_dim, activation=tf.nn.relu)
			G_Z = tf.layers.dense(hidden, self.input_dim, activation=tf.nn.sigmoid) #mnist데이터가 0~1 사이니까 sigmoid 씀.
			
			return G_Z #생성된 이미지


	
	#Discriminator 학습.
	def Discriminator_loss_function(self, D_X_logits, D_Gen_logits):
		#return tf.reduce_mean(tf.log(D_X) + tf.log(1-D_Gen)) 기존 코드.
		
		#위 식이 최대화가 되려면 D_X가 1이 되어야 하며, D_Gen이 0이 되어야 한다.
		#따라서 D_X를 계산하기 전인 D_X_logits를 입력으로 하는 sigmoid_cross_entropy_with_logits 함수로 1과의 오차 최소화 하게 하면 됨.
			#=> 최소화 : minimize함수 그대로 쓰면됨. -1 안곱해줘도 됨.
		#마찬가지로 D_Gen_logits도 함수를 통해 0과 오차 최소화 하게 하면 됨.

		#tf.ones_like(X) X와 같은 shape의 1로 이루어진 tensor를 리턴. D_X_logits을 sigmoid 한 결과와 1의 오차.
		D_X_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					labels=tf.ones_like(D_X_logits), 
					logits=D_X_logits
				)

		D_Gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					labels=tf.zeros_like(D_Gen_logits),
					logits=D_Gen_logits
				)

		#이 두 오차의 합을 최소화 하도록 학습.
		D_loss = tf.reduce_mean(D_X_loss) + tf.reduce_mean(D_Gen_loss)

		return D_loss



	#Generator 입장에서 최소화 해야 하는 값.
	def Generator_loss_function(self, D_Gen_logits):
		#return tf.reduce_mean(tf.log(D_Gen))
			# tf.reduce_mean(tf.log(1-D_Gen)) 를 최소화 하도록 해도 되지만 학습이 느림.
			#log(1-D_Gen)가 최소화 되려면 D_Gen가 커져야함(=1). == tf.log(D_Gen)를 최대화하도록 학습.

		#위 식이 최대화가 되려면 D_Gen이 1이 되어야 함. == 1과의 차이를 최소화 하도록 학습하면 됨.
		G_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					labels=tf.ones_like(D_Gen_logits), 
					logits=D_Gen_logits
				)

		G_loss = tf.reduce_mean(G_loss)

		return G_loss



def train(model, data):
	total_D_value = 0
	total_G_value = 0
	total_D_X = 0
	total_D_Genb = 0


	np.random.shuffle(data)
	iteration = int(np.ceil(len(data)/batch_size))


	for i in range( iteration ):
		#train set. mini-batch
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]

		#노이즈 생성.
		noise = model.Generate_noise(len(input_)) # len(input_) == batch_size, noise = (batch_size, model.noise_dim)
			
		#Discriminator 학습.
		_, D_value = sess.run([model.D_minimize, model.D_loss], {model.X:input_, model.noise_source:noise, model.Y:target_})

		#Generator 학습.
		_, G_value = sess.run([model.G_minimize, model.G_loss], {model.noise_source:noise, model.Y:target_})

		#학습데이터가 진짜일 확률(D_X)와 노이즈로부터 생성된 데이터가 진짜일 확률(D_Gen).  mini-batch니까 합으로 구하고 나중에 토탈크기로 나누자.
		D_X, D_Gen = sess.run([model.D_X_sum, model.D_Gen_sum], {model.X:input_, model.noise_source:noise, model.Y:target_})
		

		#parameter sum
		total_D_value += D_value
		total_G_value += G_value
		total_D_X += D_X
		total_D_Genb += D_Gen

	
	return total_D_value/iteration, total_G_value/iteration, total_D_X/len(data), total_D_Genb/len(data)



def write_tensorboard(model, D_X, D_Gen, D_value, G_value, epoch):
	summary = sess.run(model.merged, 
					{
						model.D_X_tensorboard:D_X,
						model.D_Gen_tensorboard:D_Gen,
						model.D_value_tensorboard:D_value, 
						model.G_value_tensorboard:G_value,
					}
				)

	model.writer.add_summary(summary, epoch)



def gen_image(model, epoch):
	num_generate = 10
	noise = model.Generate_noise(num_generate) # noise = (num_generate, model.noise_dim)
	target_ = np.identity(num_generate) #https://docs.scipy.org/doc/numpy/reference/generated/numpy.identity.html
	generated = sess.run(model.Gen, {model.noise_source:noise, model.Y:target_}) #num_generate, 784
	generated = np.reshape(generated, (-1, 28, 28)) #이미지 형태로. #num_generate, 28, 28
		
	fig, axes = plt.subplots(1, num_generate, figsize=(num_generate, 1))

	for i in range(num_generate):
		axes[i].set_axis_off()
		axes[i].imshow(generated[i])
		axes[i].set_title(str(i))

	plt.savefig(make_image_path+str(epoch))
	plt.close(fig)	



def run(model, train_set, restore = 0):
	#weight 저장할 폴더 생성
	if not os.path.exists(saver_path):
		os.makedirs(saver_path)
	
	#생성된 이미지 저장할 폴더 생성
	if not os.path.exists(make_image_path):
		os.makedirs(make_image_path)

	#restore인지 체크.
	if restore != 0:
		model.saver.restore(sess, saver_path+str(restore)+".ckpt")
	

	#학습 진행
	for epoch in range(restore + 1, 10001):
		#Discriminator 입장에서 최소화 해야 하는 값, #Generator 입장에서 최소화 해야 하는 값, #학습데이터가 진짜일 확률, #노이즈로부터 생성된 데이터가 진짜일 확률  
		D_value, G_value, D_X, D_Gen = train(model, train_set)

		print("epoch : ", epoch, " D_value : ", D_value, " G_value : ", G_value, " 학습데이터 진짜일 확률 : ", D_X, " 생성데이터 진짜일 확률 : ", D_Gen)

		
		if epoch % 10 == 0:
			#tensorboard
			write_tensorboard(model, D_X, D_Gen, D_value, G_value, epoch)

			#weight save
			#save_path = model.saver.save(sess, saver_path+str(epoch)+".ckpt")
		
			#image 생성
			gen_image(model, epoch)




sess = tf.Session()

#model
model = GAN(sess) #noise_dim, input_dim

#get mnist data #이미지의 값은 0~1 사이임.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#train dset
train_set = np.hstack((mnist.train.images, mnist.train.labels)) #shape = 55000, 794   => 784개는 입력, 10개는 정답.

#run
run(model, train_set)

