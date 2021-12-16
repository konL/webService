#! -*- coding: utf-8 -*-

import tensorflow as tf


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

import json
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import os

# 指定第一块GPU可用

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"



from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.metrics import classification_report, f1_score, roc_curve, auc
from bert4keras.optimizers import Adam

from mix_model.network import build_bert_model
from mix_model.keras4bert_dataset import load_data

#定义超参数和配置文件
class_nums = 2
#13
maxlen =512
#128
batch_size = 2

config_path = 'C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\bert_config.json'
checkpoint_path = 'C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\bert_model.ckpt'

dict_path = 'C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\vocab.txt'

tokenizer = Tokenizer(dict_path)
config_path_code='C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\finetund-code-bert\\config.json'
checkpoint_path_code='C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\finetund-code-bert\\bert_model_finetune.ckpt'

dict_path_code='C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\codeBERT\\vocab.json'
merge_path_code='C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\codeBERT\\merges.txt'


from transformers import AutoTokenizer,RobertaTokenizer
# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
tokenizer_code = RobertaTokenizer(dict_path_code,merge_path_code)

class data_generator(DataGenerator):

    """
    数据生成器
    """
    # def __iter__(self, random=False):
    #     # generator_ex = self.sample(random)
    #     # for i in generator_ex:
    #     #     print("data=", str(i))
    #     #     print("name=",str(i[1][0]))
    #     #     print("label=", str(i[1][4]))
    #     #     print("name_ids=", str(i[1][5:]))
    #     #     print("=================================================")
    #     batch_token_ids, batch_segment_ids, batch_name_ids,batch_labels = [], [], [],[]
    #     #G1
    #     for is_end, (col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,
    #                  col16,col17,col18,col19,col20,oldname,text,text1,body1,body2, edge,label) in self.sample(random):
    #     #g2
    #     # for is_end, (col1, col2, col3, col4, oldname, text, text1, body1,body2,edge, label) in self.sample(random):
    #
    #     #BERT
    #         token_ids, segment_ids = tokenizer.encode(text, text1, maxlen=maxlen)
    #         if str(oldname) == "nan":
    #             continue
    #
    #
    #         #CODEBERT
    #         # try:
    #         #     inputs = tokenizer.encode_plus(text, text1,max_length=512,truncation=True)
    #         # except UnicodeEncodeError:
    #         #     continue;
    #         #
    #         # # raw_inputs=[text,text1]
    #         # # inputs = tokenizer.encode_plus(raw_inputs)
    #         # token_ids=inputs['input_ids']
    #         # segment_ids=[]
    #         # sep = token_ids.index(2)
    #         # for i in range(0, sep + 1):
    #         #     segment_ids.append(0)
    #         # for i in range(0, len(token_ids) - sep - 1):
    #         #     segment_ids.append(1)
    #         # #
    #         # # print("token_ids=",token_ids)
    #         # decoded_string = tokenizer.decode(
    #         #    token_ids)
    #
    #
    #         name_ids = []
    #         #G1
    #         # name_ids.append(label);
    #         name_ids.append(col1);
    #         name_ids.append(col2);
    #         name_ids.append(col3);
    #         name_ids.append(col4);
    #         name_ids.append(col5);
    #         name_ids.append(col6);
    #
    #         name_ids.append(col7);
    #         name_ids.append(col8);
    #         name_ids.append(col9);
    #         name_ids.append(col10);
    #         name_ids.append(col11);
    #         name_ids.append(col12);
    #         name_ids.append(col13);
    #         name_ids.append(col14);
    #         name_ids.append(col15);
    #         name_ids.append(col16);
    #         name_ids.append(col17);
    #         name_ids.append(col18);
    #         name_ids.append(col19);
    #         name_ids.append(col20)
    #         #G2
    #         # name_ids.append(col1);
    #         # name_ids.append(col2);
    #         # name_ids.append(col3);
    #         # name_ids.append(col4);
    #         # name_ids.append(label);
    #
    #
    #
    #
    #         # print("name_ids=" + str(name_ids))
    #         # label=col20;
    #         # print("label=" + str(label))
    #
    #
    #
    #         # print("text="+text+"\n"+text1)
    #         # print("token_ids="+str(token_ids))
    #         # print("seg_ids="+str(segment_ids))
    #
    #
    #         batch_token_ids.append(token_ids)
    #         batch_segment_ids.append(segment_ids)
    #         batch_labels.append([label])
    #         batch_name_ids.append(name_ids)
    #         if len(batch_token_ids) == self.batch_size or is_end:
    #             batch_token_ids = sequence_padding(batch_token_ids)
    #             batch_segment_ids = sequence_padding(batch_segment_ids)
    #             batch_labels = sequence_padding(batch_labels)
    #             batch_name_ids=sequence_padding(batch_name_ids)
    #             #-------------------------------------简单在generator这里试一下，输入列表增加一个？
    #             yield [batch_token_ids, batch_segment_ids,batch_name_ids], batch_labels
    #             batch_token_ids, batch_segment_ids, batch_name_ids,batch_labels = [], [], [],[]

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids,batch_token_ids2, batch_segment_ids2, batch_labels = [], [], [], [],[]
        # G1
        for is_end, (oldname, newname,text, text1, oldedge,newedge,changeNum, label) in self.sample(
            random):

            # token_ids, segment_ids = tokenizer.encode(text, text1, maxlen=maxlen)
            # # token_ids_ent, segment_ids_ent = tokenizer.encode(oldedge)
            token_ids, segment_ids = tokenizer.encode(text,text1, maxlen=maxlen)
            oldRelation=oldedge.replace(oldname+',',"entity contain ");
            newRelation = newedge.replace(newname + ',', "entity contain ");
            print("old relation=",oldRelation)
            print("new relation=", newRelation)

            token_ids_ent, segment_ids_ent = tokenizer.encode(oldRelation,newRelation ,maxlen=maxlen)

            # if str(oldname) == "nan":
            #     continue
            #
            # # CODEBERT
            try:
                inputs = tokenizer_code.encode_plus(text, text1,max_length=512,truncation=True)
            except UnicodeEncodeError:
                continue;

            raw_inputs=[text,text1]
            inputs = tokenizer_code.encode_plus(raw_inputs)
            token_ids=inputs['input_ids']
            segment_ids=[]
            sep = token_ids.index(2)
            for i in range(0, sep + 1):
                segment_ids.append(0)
            for i in range(0, len(token_ids) - sep - 1):
                segment_ids.append(1)




            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_token_ids2.append(token_ids_ent)
            batch_segment_ids2.append(segment_ids_ent)

            batch_labels.append([label])

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_token_ids2= sequence_padding(batch_token_ids2)
                batch_segment_ids2 = sequence_padding(batch_segment_ids2)

                batch_labels = sequence_padding(batch_labels)

                # -------------------------------------简单在generator这里试一下，输入列表增加一个？
                yield [batch_token_ids, batch_segment_ids,batch_token_ids2,batch_segment_ids2], batch_labels
                batch_token_ids, batch_segment_ids,batch_token_ids2, batch_segment_ids2, batch_labels = [], [], [], [],[]

# ---自己按照公式实现
    def auc_calculate(labels, preds, n_bins=100):
        postive_len = sum(labels)
        negative_len = len(labels) - postive_len
        total_case = postive_len * negative_len
        pos_histogram = [0 for _ in range(n_bins)]
        neg_histogram = [0 for _ in range(n_bins)]
        bin_width = 1.0 / n_bins
        for i in range(len(labels)):
            nth_bin = int(preds[i] / bin_width)
            if labels[i] == 1:
                pos_histogram[nth_bin] += 1
            else:
                neg_histogram[nth_bin] += 1
        accumulated_neg = 0
        satisfied_pair = 0
        for i in range(n_bins):
            satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
            accumulated_neg += neg_histogram[i]

        return satisfied_pair / float(total_case)
from tensorflow.keras import backend as K
def recall(y_true,y_pred):

    true_positive = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
    possible_positive = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall=true_positive/(possible_positive+K.epsilon())
    return recall
def precision(y_true,y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision=true_positive/(predicted_positive+K.epsilon())
    return precision
def f1(y_true,y_pred):
    def recall(y_true,y_pred):

        true_positive = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
        possible_positive = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall=true_positive/(possible_positive+K.epsilon())
        return recall
    def precision(y_true,y_pred):
        true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision=true_positive/(predicted_positive+K.epsilon())
        return precision
    precision=precision(y_true,y_pred)
    recall=recall(y_true,y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from sklearn.metrics import roc_auc_score


import matplotlib.pyplot as plt
def acu_curve(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('roc_dubbo.png', dpi=300)
    plt.show()




if __name__ == '__main__':

    # proj="dubbo"
    issmall=""
    # 加载数据集
    # train_data ,train= load_data('C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\train_data_6x\\mix\\'+proj+'_train.csv',
    #                              'C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\numericData\\'+proj+"_train_G1.csv")

    # train_data, train = load_data(
    #     "C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data_extend\\test_data\\beam_test_method_2.csv",
    #     'C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data_extend\\numericData\\beam_G1_method.csv')
    # # train_data, train = load_data(
    #     'C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data_extend\\test_data\\beam_test.csv',
    #     'C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\numericData\\dubbo_Test_G2_beamtrain.csv')

    # val_data ,val= load_data('C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\test_data_6x\\no_order\\beam_test_mask_change.csv',
    #                          'C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\numericData\\val_G1.csv')




    # test_data, test = load_data('C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\test_data_6x\\mix\\'+proj+'_test_mask_change'+issmall+'.csv',
    #                             'C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\numericData\\'+proj+"_test_G1.csv")

    # test_data, test = load_data(
    #     "C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data_extend\\test_data\\dubbo_test.csv",
    #     'C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\numericData\\' + proj + "_Test_G2_new.csv")

    # test_data, test = load_data(
    #     "C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data_extend\\test_data\\dubbo_test_method_notsame.csv",
    #     'C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data_extend\\numericData\\dubbo_G1_method.csv')

    # # test_data, test = load_data('bi_train_method.csv')
    proj = "asterisk-java-iax"
    train_data, train = load_data(
        'C:\\project\\Projects_50\\VersionDB\\process_data\\train_data\\GROUP1_same\\' + proj + '_train.csv')
    # train_data, train = load_data(
    #     'C:\\project\\Projects_50\\VersionDB\\process_data\\test_data\\abdera_method_same.csv')
    val_data, val = load_data(
        'C:\\project\\Projects_50\\VersionDB\\process_data\\test_data\\101repo_method.csv')

    # test_data, test = load_data(
    #     'C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data\\test_data_6x\\no_order\\' + proj + '_test_mask_change.csv')

    # test_data, test = load_data(
    #     'C:\\project\\Projects_50\\VersionDB\\process_data\\test_data\\101repo_method_same.csv')
    test_data, test = load_data(
            'C:\\project\\Projects_50\\VersionDB\\process_data\\test_data\\GROUP1_same\\' + proj + '_method_same.csv')

    print(train['label_class'].value_counts())


    columns = train.columns
    print(columns)



    # 删除最后一列，即class列
    features_columns = columns.delete(len(columns) - 1)
    # features_columns = columns.delete(4)



    # 获取除class列以外的所有特征列
    features = train[features_columns]


    # 获取class列
    labels = train['label_class']
    print(features)
    print(labels)
    # #划分原始数据训练集和测试集用于oversample模型生成

    # RandomUnderSampler函数是一种快速并十分简单的方式来平衡各个类别的数据: 随机选取数据的子集.
    from imblearn.under_sampling import RandomUnderSampler

    rus = RandomUnderSampler(random_state=0)
    os_features, os_labels = rus.fit_resample(features, labels)
    # 新生成的数据集
    #顺序：features+labels
    train = pd.concat([os_features, os_labels], axis=1)
    print(train['label_class'].value_counts())
    train_data=train.values




    # 转换数据集
    train_generator = data_generator(train_data, batch_size)
    val_generator = data_generator(val_data, batch_size)
    test_generator = data_generator(test_data, batch_size)
    import keras_metrics as km
    model = build_bert_model(config_path,checkpoint_path,config_path_code,checkpoint_path_code,class_nums)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(5e-6),
        metrics=['accuracy',f1,precision,recall],
    )

    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        verbose=1,
        mode='min'
        )
    bast_model_filepath = 'v3_best_model.weights'
    checkpoint = keras.callbacks.ModelCheckpoint(
        bast_model_filepath,
        monitor='val_loss',
        verbose=2,
        save_best_only=True,
        mode='min'
        )
    from sklearn.utils import class_weight
    import numpy as np


    history=model.fit_generator(

        train_generator.forfit(),
        steps_per_epoch=len(val_generator),
        epochs=20,

        validation_data=val_generator.forfit(),
        validation_steps=len(val_generator),
        shuffle=True,
        verbose=1,

        callbacks=[earlystop,checkpoint]
    )

    # history_dict = history.history
    # print(history_dict.keys())
    #
    # # 训练loss
    # # 绘制训练损失，每轮都下降
    # import matplotlib.pyplot as plt
    #
    # loss_values = history_dict['loss']
    # val_loss_values = history_dict['val_loss']
    # epochs = range(1, len(loss_values) + 1)
    # plt.plot(epochs, loss_values, 'r', label='Training loss')
    # plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    #
    # plt.clf()
    # f1 = history_dict['f1']
    # val_f1 = history_dict['val_f1']
    # plt.plot(epochs, f1, 'r', label='Training f1_score')
    # plt.plot(epochs, val_f1, 'b', label='Validation f1_score')
    # plt.title('Training and validation f1_score')
    # plt.xlabel('Epochs')
    # plt.ylabel('f1_score')
    # plt.legend()
    # plt.show()
    # plt.clf()
    # f1 = history_dict['precision']
    # val_f1 = history_dict['val_precision']
    # plt.plot(epochs, f1, 'r', label='Training precision')
    # plt.plot(epochs, val_f1, 'b', label='Validation precision')
    # plt.title('Training and validation precision')
    # plt.xlabel('Epochs')
    # plt.ylabel('precision')
    # plt.legend()
    # plt.show()
    # plt.clf()
    # f1 = history_dict['recall']
    # val_f1 = history_dict['val_recall']
    # plt.plot(epochs, f1, 'r', label='Training recall')
    # plt.plot(epochs, val_f1, 'b', label='Validation recall')
    # plt.title('Training and validation recall')
    # plt.xlabel('Epochs')
    # plt.ylabel('recall')
    # plt.legend()
    # plt.show()
    # score,  f1, precision, recall = model.evaluate(test_generator, steps=50,
    #                                                           max_queue_size=10,
    #                                                           use_multiprocessing=False)
    # print('score:', score, 'f1:', f1, 'precision:', precision, 'recall', recall)
    # model.save("model.hdf5")fdubbo
    model.load_weights('v3_best_model.weights')
    test_pred = []
    test_true = []
    #+++++++++++++++++++++







    # for x,y in test_generator:
    #
    #     p = model.predict(x).argmax(axis=1)
    #     test_pred.extend(p)
    #
    # test_true = test_data[:,2].tolist()
    for x, y in test_generator:
        print("x=" + str(x))

        p = model.predict(x).argmax(axis=1)
        test_pred.extend(p)

        print("predict="+str(p))
#G1 24
    test_true = test["label_class"].tolist()

    print("项目=",proj)
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    index_i = 0
    for i in range(len(test_true)):
        pred = int(test_pred[i])


            # index_i=index_i+1

        # pred = int(test_pred[i])

        if ((test_true[i] == pred) & (test_true[i] == 0)):
            tn = tn + 1
        if ((test_true[i] == pred) & (test_true[i] == 1)):
            tp = tp + 1
        if ((test_true[i] != pred) & (test_true[i] == 0)):
            fp = fp + 1
        if ((test_true[i] != pred) & (test_true[i] == 1)):
            # print(test_true[i], " ", test_pred[i])
            # print(test_true[i] == 1)
            # print(type(test_true[i] ))
            # print(type(pred))
            fn = fn + 1

    print(tp, fp, tn, fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("f1:", 2 * ((precision * recall) / (precision + recall)))
    print("precision:", precision)
    print("recall:", recall)
    # target_names = [line.strip() for line in open('label','r',encoding='utf8')]
    # print(classification_report(test_true, test_pred,target_names=target_names))
    print(classification_report(test_true, test_pred))






    # name_set = test['oldname'].tolist()
    # #
    # #
    # def calChangeNum(i,test_pred):
    #     data=test["edge"].tolist()
    #     x=data[i]
    #     #获取第i个标识符的相关实体集合 edge
    #     edges = x.split("|")
    #
    #     changeEnt = 0
    #
    #     for e in edges:
    #         index = e.find(',')
    #         # 获取相关实体
    #         node = e[index + 1:-1].strip()
    #         # 在test_pred中
    #
    #         if node in name_set:
    #             index = name_set.index(node)
    #             if test_pred[index] == 1:
    #                 changeEnt = changeEnt + 1
    #         # else:
    #         #     print(node, changeEnt)
    #         #     changeEnt = changeEnt + 0
    #
    #
    #     #     print(sumEnt)
    #     return changeEnt




    changeNum=test['changeNum'].tolist()

    fp = 0
    tp = 0
    fn = 0
    tn = 0
    index_i = 0
    for i in range(len(test_true)):
        pred = int(test_pred[i])

        # if ((test_pred[i] == 1) & (oldStmt[i]==newStmt[i])):
            #     pred=0
        if ((test_pred[i] == 0) & (changeNum[i] >0 )):
            test_pred[i] = 1
            pred = 1
        # if ((test_pred[i] == 1) & (changeNum[i] ==0 )):
        #     test_pred[i] = 0
        #     pred = 0

            # index_i=index_i+1

        # pred = int(test_pred[i])

        if ((test_true[i] == pred) & (test_true[i] == 0)):
            tn = tn + 1
        if ((test_true[i] == pred) & (test_true[i] == 1)):
            tp = tp + 1
        if ((test_true[i] != pred) & (test_true[i] == 0)):
            fp = fp + 1
        if ((test_true[i] != pred) & (test_true[i] == 1)):
            # print(test_true[i], " ", test_pred[i])
            # print(test_true[i] == 1)
            # print(type(test_true[i] ))
            # print(type(pred))
            fn = fn + 1

    print(tp, fp, tn, fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("f1:", 2 * ((precision * recall) / (precision + recall)))
    print("precision:", precision)
    print("recall:", recall)
    acu_curve(test_true,test_pred)


