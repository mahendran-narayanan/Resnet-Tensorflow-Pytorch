import argparse
import tensorflow as tf
import tensorflow_addons as tfa
import torch

class Resnet_tf(tf.keras.Model):
	def __init__(self):
		super().__init__()
		pass

	def call(self,x):
		return x

class Resnet_torch(torch.nn.Module):
	def __init__(self):
		super().__init__()
		pass
		
	def forward(self, x):
		return x

def model_torch():
	return Resnet_torch()

def model_tf():
	return Resnet_tf()

def main(args):
	if args.model=='tf':
		print('Model Resnet_'+str(args.depth)+' will be created in Tensorflow')
		if args.depth=='9':
			model = model_tf_9()
			model.build(input_shape=(None,224,224,3))
			model.summary()
		elif args.depth=='18':
			model = model_tf_18()
			model.build(input_shape=(None,224,224,3))
			model.summary()
		elif args.depth=='34':
			model = model_tf_34()
			model.build(input_shape=(None,224,224,3))
			model.summary()
		elif args.depth=='50':
			model = model_tf_50()
			model.build(input_shape=(None,224,224,3))
			model.summary()
		elif args.depth=='101':
			model = model_tf_101()
			model.build(input_shape=(None,224,224,3))
			model.summary()
		elif args.depth=='152':
			model = model_tf_152()
			model.build(input_shape=(None,224,224,3))
			model.summary()
	else:
		print('Model Resnet_'+str(args.depth)+' will be created in Pytorch')
		if args.depth=='9':
			model = model_torch9()
			print(model)
		elif args.depth=='18':
			model = model_torch18()
			print(model)
		elif args.depth=='34':
			model = model_torch34()
			print(model)
		elif args.depth=='50':
			model = model_torch50()
			print(model)
		elif args.depth=='101':
			model = model_torch101()
			print(model)
		elif args.depth=='152':
			model = model_torch152()
			print(model)

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
	main(args)