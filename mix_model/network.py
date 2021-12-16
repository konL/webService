# 把目标当成一个输入，构成多输入模型


# BERT的理解
# 第二个问题是“有什么原则来指导Bert后面应该要接哪些层？”。
# 答案是：用尽可能少的层来完成你的任务。比如上述情感分析只是一个二分类任务，你就取出第一个向量然后加个Dense(1)就好了，
# 不要想着多加几层Dense，更加不要想着接个LSTM再接Dense；如果你要做序列标注（比如NER），那你就接个Dense+CRF就好，
# 也不要多加其他东西。总之，额外加的东西尽可能少。一是因为Bert本身就足够复杂，它有足够能力应对你要做的很多任务；
# 二来你自己加的层都是随机初始化的，加太多会对Bert的预训练权重造成剧烈扰动，容易降低效果甚至造成模型不收敛～
from bert4keras.models import build_transformer_model, Lambda
from keras.layers import Input, Embedding, LSTM, Dense, Reshape
from keras.models import Model
import keras
import tensorflow as tf
from bert4keras.backend import keras,set_gelu
# config_path='C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\bert_config.json'
# checkpoint_path='C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\bert_model.ckpt'
set_gelu('tanh')
def textcnn(inputs,kernel_initializer):
	# 3,4,5
	cnn1 = keras.layers.Conv1D(
			256,
			3,
			strides=1,
			padding='same',
			activation='relu',
			kernel_initializer=kernel_initializer
		)(inputs) # shape=[batch_size,maxlen-2,256]
	cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)  # shape=[batch_size,256]

	cnn2 = keras.layers.Conv1D(
			256,
			4,
			strides=1,
			padding='same',
			activation='relu',
			kernel_initializer=kernel_initializer
		)(inputs)
	cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)

	cnn3 = keras.layers.Conv1D(
			256,
			5,
			strides=1,
			padding='same',
			kernel_initializer=kernel_initializer
		)(inputs)
	cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)

	output = keras.layers.concatenate(
		[cnn1,cnn2,cnn3],
		axis=-1)
	output = keras.layers.Dropout(0.2)(output)
	return output
from transformers import AutoModel, RobertaConfig, RobertaModel,RobertaTokenizer
def build_bert_model(config_path,checkpoint_path,config_path_code,checkpoint_path_code,class_nums):
	# bert = RobertaModel.from_pretrained("microsoft/codebert-base")
	# RobertaTokenizer.from_pretrained()
	bert = build_transformer_model(
			config_path=config_path_code,
			checkpoint_path=checkpoint_path_code,
			model = 'bert',
			prefix='BERT-A-',
			return_keras_model=False)


	# x = keras.layers.Lambda(
	# 		lambda x: x[:, 0],
	# 		name='cls-token'
	# 	)(bert.model.output)  # shape=[batch_size,768]
	cls_features = keras.layers.Lambda(
		lambda x: x[:, 0],
		name='cls-token-A'
	)(bert.model.output)  # shape=[batch_size,768]
	all_token_embedding = keras.layers.Lambda(
		lambda x: x[:, 1:-1],
		name='all-token-A'
	)(bert.model.output)  # shape=[batch_size,maxlen-2,768]

	# 我们的方法
	# cnn_features = textcnn(
	# 	all_token_embedding, bert.initializer)  # shape=[batch_size,cnn_output_dim]
	# concat_features = keras.layers.concatenate(
	# 	[cls_features, cnn_features],
	# 	axis=-1)

	# g1
	# auxiliary_input = Input(shape=(15,), name='aux_input')
	# x = Embedding(output_dim=15,  input_dim=2, input_length=15)(auxiliary_input)
	# x=keras.layers.GlobalMaxPooling1D()(x)
	#g2
	bert_enty = build_transformer_model(
		prefix='BERT-B-',

			config_path=config_path,
			checkpoint_path=checkpoint_path,
			model='bert',
			return_keras_model=False)
	cls_features_enty = keras.layers.Lambda(
		lambda x: x[:, 0],
		name='cls-token-B'
	)(bert_enty.model.output)  # shape=[batch_size,768]
	all_token_embedding_enty = keras.layers.Lambda(
		lambda x: x[:, 1:-1],
		name='all-token-B'
	)(bert_enty.model.output)  # shape=[batch_size,maxlen-2,768]
	# auxiliary_input = Input(shape=(20,), name='aux_input')
	# x = Embedding(output_dim=20, input_dim=2, input_length=20)(auxiliary_input)
	# allF = keras.layers.concatenate([x, cls_features])


	# x1=keras.layers.GlobalMaxPooling1D()(all_token_embedding)
	# x2=keras.layers.GlobalMaxPooling1D()(all_token_embedding_enty)
	# subtracted = keras.layers.Subtract()([x1,x2])
	# textCnn做Pooling
	x1 = textcnn(all_token_embedding,bert.initializer)
	x2 = textcnn(all_token_embedding_enty,bert.initializer)
	subtracted = keras.layers.Subtract()([x1, x2])

	allF = keras.layers.concatenate([x1,x2,subtracted])

	dense = keras.layers.Dense(
		units=512,
		activation='relu',
		kernel_initializer=bert.initializer
	)(allF)
	# dense = keras.layers.Dense(
	# 		units=512,
	# 		activation='relu',
	# 		kernel_initializer=bert.initializer
	# 	)(cls_features)





	main_output = keras.layers.Dense(
		units=class_nums,
		activation='softmax',
		kernel_initializer=bert.initializer
	)(dense)



	import numpy as np


	# model = Model(inputs=[bert.model.input[0],bert.model.input[1], auxiliary_input], outputs=[main_output])
	model = Model(inputs=[bert.model.input[0], bert.model.input[1], bert_enty.model.input[0], bert_enty.model.input[1]], outputs=[main_output])




	model.summary()
	# from tensorflow.keras.utils import plot_model
	# plot_model(model, './model_mix_simanse.png', show_shapes=True)
	return model

# if __name__ == '__main__':
#     config_path = 'C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\bert_config.json'
#     checkpoint_path = 'C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\bert_model.ckpt'
#     class_nums=2
#     build_bert_model(config_path,checkpoint_path,class_nums)

