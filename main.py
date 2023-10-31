import argparse
import tensorflow as tf
import torch

class block(tf.keras.Model):
	def __init__(self,kernels,shortcut=False):
		super().__init__()
		self.sh = shortcut
		if shortcut==True:
			self.sh1 = tf.keras.layers.Conv2D(4 * kernels, 1, strides=2)
			self.sh2 = tf.keras.layers.BatchNormalization()
		self.l1 = tf.keras.layers.Conv2D(kernels, 1,strides=2)
		self.l2 = tf.keras.layers.BatchNormalization()
		self.l3 = tf.keras.layers.ReLU()

		self.l21 = tf.keras.layers.Conv2D(kernels, 3, strides=1,padding='SAME')
		self.l22 = tf.keras.layers.BatchNormalization()
		self.l23 = tf.keras.layers.ReLU()

		self.l31 = tf.keras.layers.Conv2D(4 * kernels, 1)
		self.l32 = tf.keras.layers.BatchNormalization()
		self.l33 = tf.keras.layers.ReLU()

		self.add = tf.keras.layers.Add()
	def call(self,x):
		if self.sh==True:
			sh_res = self.sh2(self.sh1(x))
		x = self.l3(self.l2(self.l1(x)))
		x = self.l23(self.l22(self.l21(x)))
		x = self.l33(self.l32(self.l31(x)))
		if self.sh==True:
			x = self.add([sh_res,x])
		return x

class Resnet_tf(tf.keras.Model):
	def __init__(self, depth):
		super().__init__()
		if depth=='9':
			self.select = [1,1,1,1]
		elif depth=='18':
			self.select = [2,2,2,2]
		elif depth=='50':
			self.select = [3,4,6,3] # Default depth : 50
		elif depth=='101':
			self.select = [3,4,23,3]
		elif depth=='152':
			self.select = [3,8,36,3]
		else:
			print("please select mentioned depth")
		self.initial = tf.keras.layers.Conv2D(64,7,strides=2)
		self.bn = tf.keras.layers.BatchNormalization()
		self.act = tf.keras.layers.ReLU()
		self.block1 = []
		self.block2 = []
		self.block3 = []
		self.block4 = []
		for i in range(self.select[0]):
			if i==0:
				self.block1.append(block(64,True))
			else:
				self.block1.append(block(64))
		for i in range(self.select[1]):
			if i==0:
				self.block2.append(block(128,True))
			else:
				self.block2.append(block(128))
		for i in range(self.select[2]):
			if i==0:
				self.block3.append(block(256,True))
			else:
				self.block3.append(block(256))
		for i in range(self.select[3]):
			if i==0:
				self.block4.append(block(512,True))
			else:
				self.block4.append(block(512))
		self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
		self.dense = tf.keras.layers.Dense(1000,activation='softmax')
	def call(self,x):
		x = self.initial(x)
		x = self.act(self.bn(x))
		for i in range(self.select[0]):
			x = self.block1[i](x)
		for i in range(self.select[1]):
			x = self.block2[i](x)
		for i in range(self.select[2]):
			x = self.block3[i](x)
		for i in range(self.select[3]):
			x = self.block4[i](x)
		x = self.avgpool(x)
		x = self.dense(x)
		return x

def main(modeltype, depth):
	if modeltype=='tf':
		print('Model Resnet_'+str(depth)+' will be created in Tensorflow')
		model = Resnet_tf(depth)
		model.build(input_shape=(None,224,224,3))
		model.summary()
	else:
		print('Model Resnet_'+str(depth)+' will be created in Pytorch')
		if depth=='9':
			model = model_9(modeltype)
			print(model)
		# elif depth=='18':
		# 	model = model_18(modeltype)
		# 	print(model)
		# elif depth=='34':
		# 	model = model_34(modeltype)
		# 	print(model)
		# elif depth=='50':
		# 	model = model_50(modeltype)
		# 	print(model)
		# elif depth=='101':
		# 	model = model_101(modeltype)
		# 	print(model)
		# elif depth=='152':
		# 	model = model_152(modeltype)
		# 	print(model)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create Resnet model in Tensorflow or Pytorch')
	parser.add_argument('--model',
	                    default='tf',
	                    choices=['tf', 'torch'],
	                    help='Model will be created on Tensorflow, Pytorch (default: %(default)s)')
	parser.add_argument('--depth',
	                    default='50',
	                    choices=['9', '18', '50', '101', '152'],
	                    help='Resnet model depth (default: %(default)s)')
	args = parser.parse_args()
	print('args model ',args.model)
	main(modeltype = args.model,depth = args.depth)