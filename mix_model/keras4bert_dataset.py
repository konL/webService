#! -*- coding: utf-8 -*-
import json
import pandas as pd

def gen_training_data(raw_data_path):
    data = pd.read_csv(raw_data_path, header=0)
                       #                   names=["label_class", "type", "oldname","newname", "oldStmt", "newStmt"] )

    print(data['label_class'].value_counts())

    # data['text_len'] = data['text'].map(lambda x: len(x))
    data['old_text_len'] = data['oldStmt'].map(lambda x: len(x))
    print(data['old_text_len'].describe())
    import matplotlib.pyplot as plt
    plt.hist(data['old_text_len'], bins=30, rwidth=0.9, density=True,)
    plt.show()

    del data['old_text_len']

    data = data.sample(frac=1.0)
    # train_num = int(0.9*len(data))
    train_num = int(0.8 * len(data))
    # val_num = int(0.9 * len(data))
    # train,val,test = data[:train_num],data[train_num:val_num],data[val_num:]
    train, val = data[:train_num], data[train_num:]
    train.to_csv("beam_zeppelin_prepocessed_train.csv",index=False)
    val.to_csv("beam_zeppelin_prepocessed_val.csv",index=False)
    # test.to_csv("bi_test_method_norepeat.csv",index=False)


# def load_data(filename,numeric_file):
#     """加载数据
#     单条格式：(文本, 标签id)
#     """
#     df = pd.read_csv(filename,header=0,encoding='unicode_escape')
#     # print(df.head())
#
#     #G1
#     # df_val = pd.read_csv(numeric_file, header=None)
#     # dd=pd.concat([df_val.iloc[:,:20],df[['oldname','oldStmt', 'newStmt','changeNum','label_class']] ],axis=1)
#
#     #G2
#     df_val = pd.read_csv(numeric_file, header=0)
#     # print(df_val.head())
#     dd = pd.concat([df_val.iloc[:, :20], df[['oldname', 'oldStmt', 'newStmt', 'oldStmt_body', 'newStmt_body','changeNum', 'label_class']]], axis=1)
#
#
#     # return df[['oldStmt','newStmt','label_class']].values,df[['oldStmt','newStmt','label_class']]
#     # return df[['oldname','oldStmt', 'newStmt' ,'changeNum','label_class']].values, df[['oldname','oldStmt', 'newStmt','changeNum','label_class']]
#     # return df[['oldname','oldStmt', 'newStmt' ,'changeNum','label_class']].values, df[['oldname','oldStmt', 'newStmt','changeNum','label_class']]
#     return dd.values, dd

def load_data(filename):

        """加载数据
        单条格式：(文本, 标签id)
        """
        df = pd.read_csv(filename, header=0)
        # return df[['oldStmt','newStmt','label_class']].values,df[['oldStmt','newStmt','label_class']]
        # return df[['oldname','oldStmt', 'newStmt' ,'changeNum','label_class']].values, df[['oldname','oldStmt', 'newStmt','changeNum','label_class']]
        return df[['oldname','newname', 'oldStmt', 'newStmt', 'edge', 'newedge', 'changeNum', 'label_class']].values, \
               df[['oldname', 'newname','oldStmt', 'newStmt', 'edge', 'newedge', 'changeNum', 'label_class']]




# if __name__ == '__main__':
#     # data_path="CMID.json"
#     data_path = "C:\\project\\IdentifierStyle\\data\\VersionDB\\changeIdentifier\\beam_zeppelin_prepocessed.csv"
#     gen_training_data(data_path)