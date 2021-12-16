import pandas as pd
def gen_training_data(raw_data_path):
    raw_data = pd.read_csv(raw_data_path, header=0)
                       #                   names=["label_class", "type", "oldname","newname", "oldStmt", "newStmt"] )

    print(raw_data['label_class'].value_counts())

    # data['text_len'] = data['text'].map(lambda x: len(x))
    data=raw_data.drop_duplicates()
    # data.to_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data_extend\\test_data\\dubbo_test1.csv")
    data = data.sample(frac=1.0)
    # train_num = int(0.9*len(data))
    train_num = int(0.8 * len(data))
    # val_num = int(0.9 * len(data))
    # train,val,test = data[:train_num],data[train_num:val_num],data[val_num:]
    train, val = data[:train_num], data[train_num:]
    train.to_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data_extend\\test_data\\dubbo_train_group.csv",index=False)
    val.to_csv("C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data_extend\\test_data\\dubbo_test_group.csv",index=False)
    # test.to_csv("bi_test_method_norepeat.csv",index=False)
gen_training_data("C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data_extend\\test_data\\dubbo_test.csv")