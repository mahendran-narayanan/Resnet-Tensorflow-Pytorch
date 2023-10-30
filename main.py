import argparse
import tensorflow as tf
import torch

def conv_block(layer, kernels,kernelsize,stride,pad,name):
	layer = []
	layer.append(tf.keras.layers.Conv2D(kernels,kernelsize,strides=stride,padding=pad))
	layer.append(tf.keras.layers.BatchNormalization())
	layer.append(tf.keras.layers.ReLU())
	return tf.keras.Sequential(layer,name='conv_bn_relu_'+str(name))

def block(temp):
	# layer = []
	# for i in range(temp):
	return conv_block(layer,32,3,1,'SAME','as')
	# return tf.keras.Sequential(layer,name='conv_bn_relu_'+str('as'))

class Resnet_tf(tf.keras.Model):
	def __init__(self, depth):
		super().__init__()
		if depth=='9':
			self.select = [1,1,1,1]
		elif depth=='18':
			self.select = [2,2,2,2]
		elif depth=='34':
			self.select = [3,4,6,3]
		elif depth=='50':
			self.select = [3,4,6,3] # Default depth : 50
		elif depth=='101':
			self.select = [3,4,23,3]
		elif depth=='152':
			self.select = [3,8,36,3]
		else:
			print("please select mentioned depth")
		self.block1 = block(self.select[0])
		self.block2 = block(self.select[1])
		self.block3 = block(self.select[2])
		self.block4 = block(self.select[3])
	def call(self,x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		return x

def main(modeltype, depth):
	if modeltype=='tf':
		print('Model Resnet_'+str(depth)+' will be created in Tensorflow')
		if depth=='9':
			model = Resnet_tf(depth)
			model.build(input_shape=(None,224,224,3))
			model.summary()
		# elif depth=='18':
		# 	model = model_18(modeltype)
		# 	model.build(input_shape=(None,224,224,3))
		# 	model.summary()
		# elif depth=='34':
		# 	model = model_34(modeltype)
		# 	model.build(input_shape=(None,224,224,3))
		# 	model.summary()
		# elif depth=='50':
		# 	model = model_50(modeltype)
		# 	model.build(input_shape=(None,224,224,3))
		# 	model.summary()
		# elif depth=='101':
		# 	model = model_101(modeltype)
		# 	model.build(input_shape=(None,224,224,3))
		# 	model.summary()
		# elif depth=='152':
		# 	model = model_152(modeltype)
		# 	model.build(input_shape=(None,224,224,3))
		# 	model.summary()
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
	                    choices=['9', '18', '34', '50', '101', '152'],
	                    help='Resnet model depth (default: %(default)s)')
	args = parser.parse_args()
	print('args model ',args.model)
	main(modeltype = args.model,depth = args.depth)