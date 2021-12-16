# 把目标当成一个输入，构成多输入模型
# 把loss写成一个层，作为最后的输出
# 搭建模型的时候，就只需要将模型的output定义为loss
# 而compile的时候，直接将loss设置为y_pred（因为模型的输出就是loss，所以y_pred就是loss），无视y_true，
# 训练的时候，y_true随便扔一个符合形状的数组进去就行了。
# 最后我们得到的是问题和答案的编码器，也就是问题和答案都分别编码出一个向量来，我们只需要比较cos，就可以选择最优答案了。

# BERT的理解
# 第二个问题是“有什么原则来指导Bert后面应该要接哪些层？”。
# 答案是：用尽可能少的层来完成你的任务。比如上述情感分析只是一个二分类任务，你就取出第一个向量然后加个Dense(1)就好了，
# 不要想着多加几层Dense，更加不要想着接个LSTM再接Dense；如果你要做序列标注（比如NER），那你就接个Dense+CRF就好，
# 也不要多加其他东西。总之，额外加的东西尽可能少。一是因为Bert本身就足够复杂，它有足够能力应对你要做的很多任务；
# 二来你自己加的层都是随机初始化的，加太多会对Bert的预训练权重造成剧烈扰动，容易降低效果甚至造成模型不收敛～
from bert4keras.models import build_transformer_model, Lambda
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
config_path='C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\bert_config.json'
checkpoint_path='C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\bert_model.ckpt'
# 标题输入：接收一个含有 100 个整数的序列，每个整数在 1 到 10000 之间。
# 注意我们可以通过传递一个 "name" 参数来命名任何层。
# main_input_01 = Input(shape=(100,), dtype='int32', name='main_input1')
# main_input_02 = Input(shape=(100,), dtype='int32', name='main_input2')
#
# #
# # # Embedding 层将输入序列编码为一个稠密向量的序列，
# # # 每个向量维度为 512。
# x1 = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input_01)
# x2 = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input_02)
# # LSTM 层把向量序列转换成单个向量，
# # 它包含整个序列的上下文信息
# x = keras.layers.concatenate([x1,x2])
bert = build_transformer_model(
		config_path=config_path,
		checkpoint_path=checkpoint_path,
		model='bert',
		return_keras_model=False)

# x = keras.layers.Lambda(
# 		lambda x: x[:, 0],
# 		name='cls-token'
# 	)(bert.model.output)  # shape=[batch_size,768]
x = keras.layers.Lambda(
		lambda x: x[:, 1:-1],
		name='all-token'
	)(bert.model.output)
lstm_out = LSTM(32)(x)

auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# 堆叠多个全连接网络层
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
import tensorflow as tf
import tensorflow.keras.backend as K
# 最后添加主要的逻辑回归层
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
import numpy as np

main_input=keras.layers.Lambda(lambda x:x)(bert.model.input)
model = Model(inputs=[bert.model.input[0],bert.model.input[1], auxiliary_input], outputs=[main_output])
model.summary()
# from tensorflow.keras.utils import plot_model
# plot_model(model, './model_mix.png', show_shapes=True)


