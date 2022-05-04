############# Binary and Ternary classifiers with MLP ################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from numpy.random import seed
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
seed(2)
tf.random.set_seed(1)

'''accuracy'''


def acc_plot_graphs(history, acc):
    plt.plot(history.history[acc], 'r', marker='s', markersize=4)
    plt.plot(history.history['val_'+acc], 'g', marker='v', markersize=4)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend([acc, 'test_'+acc], frameon=False)


''' loss'''


def loss_plot_graphs(history, loss):
    plt.plot(history.history[loss], 'k', marker='v', markersize=4)
    plt.plot(history.history['val_'+loss], 'b', marker='^', markersize=4)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend([loss, 'test_'+loss], frameon=False)


''' 准确度和损失函数 '''


def plot_graphs(history, acc, loss):
    plt.plot(history.history[acc], 'r', marker='s', markersize=4)
    plt.plot(history.history[loss], 'k', marker='.', markersize=4)
    plt.plot(history.history['val_'+acc], 'g', marker='v', markersize=4)
    plt.plot(history.history['val_'+loss], 'b', marker='^', markersize=4)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel('acc_loss', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend([acc, loss, 'val_'+acc, 'val_'+loss], frameon=False)


'''roc'''


def plot_roc_graphs(y_test, y_test_pred_probs):
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--', color='black')
    plt.title('ROC Curve',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('False Positive Rate',fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)


# read feature matrix
clinical_metabolism_train = pd.read_csv("SPPT_SNPT_Ctrl_features_selected-trainSet.csv")
clinical_metabolism_test = pd.read_csv("SPPT_SNPT_Ctrl_features_selected-trainSet.csv")

## 1. Binary classifiers with MLP
## example SPPT vs SNPT

pos_neg_clinical_metabolism_train = clinical_metabolism_train[(clinical_metabolism_train['y'] == 'pos') | (clinical_metabolism_train['y'] == 'neg')]
pos_neg_clinical_metabolism_test = clinical_metabolism_test[(clinical_metabolism_test['y'] == 'pos') | (clinical_metabolism_test['y'] == 'neg')]
pos_neg_clinical_metabolism_train['y'] = pos_neg_clinical_metabolism_train['y'].map({'pos': 1, 'neg': 0})
pos_neg_clinical_metabolism_test['y'] = pos_neg_clinical_metabolism_test['y'].map({'pos': 1, 'neg': 0})

pos_neg_clinical_metabolism_x_train = pos_neg_clinical_metabolism_train.loc[:, ['Val.Ser','Methoxyacetic.acid','Ethyl.3.hydroxybutyrate']].values
pos_neg_clinical_metabolism_y_train = pos_neg_clinical_metabolism_train.loc[:, ['y']].values
pos_neg_clinical_metabolism_x_test = pos_neg_clinical_metabolism_test.loc[:, ['Val.Ser','Methoxyacetic.acid','Ethyl.3.hydroxybutyrate']].values
pos_neg_clinical_metabolism_y_test = pos_neg_clinical_metabolism_test.loc[:, ['y']].values

pos_neg_clinical_metabolism_x_train_scaled = pos_neg_clinical_metabolism_x_train
pos_neg_clinical_metabolism_x_test_scaled = pos_neg_clinical_metabolism_x_test

def pos_neg_clinical_metabolism_create_model_kfold():
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(15, activation='relu', input_shape=pos_neg_clinical_metabolism_x_train_scaled.shape[1:]),
         tf.keras.layers.Dense(15, activation='relu'),
         tf.keras.layers.Dense(1, activation='sigmoid')
         ]
    )
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

kfold = KFold(n_splits=10, shuffle=True, random_state=6666)
pos_neg_clinical_metabolism_accuracys = []
pos_neg_clinical_metabolism_losses = []
pos_neg_clinical_metabolism_accuracy_valid = []
pos_neg_clinical_metabolism_precision_valid = []
pos_neg_clinical_metabolism_recall_valid = []
pos_neg_clinical_metabolism_f1_score_valid = []

for train, valid in kfold.split(pos_neg_clinical_metabolism_x_train_scaled, pos_neg_clinical_metabolism_y_train):
    model_kfold = pos_neg_clinical_metabolism_create_model_kfold()
    early_stopping = EarlyStopping(monitor='loss', patience=3)
    history = model_kfold.fit(pos_neg_clinical_metabolism_x_train_scaled[train], pos_neg_clinical_metabolism_y_train[train], validation_data=(pos_neg_clinical_metabolism_x_train_scaled[valid], pos_neg_clinical_metabolism_y_train[valid]), epochs=250, verbose=0, shuffle=True, callbacks=[early_stopping])
    # evaluate the model
    scores = model_kfold.evaluate(pos_neg_clinical_metabolism_x_train_scaled[valid], pos_neg_clinical_metabolism_y_train[valid], verbose=1)
    pos_neg_clinical_metabolism_accuracys.append(scores[1])
    pos_neg_clinical_metabolism_losses.append(scores[0])
    pos_neg_clinical_metabolism_y_valid_pred = model_kfold.predict(pos_neg_clinical_metabolism_x_train_scaled[valid])
    pos_neg_clinical_metabolism_y_valid_predn = (pos_neg_clinical_metabolism_y_valid_pred > 0.5).astype("int32")
    c_matrix = confusion_matrix(pos_neg_clinical_metabolism_y_train[valid], pos_neg_clinical_metabolism_y_valid_predn)
    # accuracy，precision，recall_test，f1_score_test
    Accuracy_valid = accuracy_score(pos_neg_clinical_metabolism_y_train[valid], pos_neg_clinical_metabolism_y_valid_predn)
    Precision_valid = precision_score(pos_neg_clinical_metabolism_y_train[valid], pos_neg_clinical_metabolism_y_valid_predn,
                                      average='macro')
    Recall_valid = recall_score(pos_neg_clinical_metabolism_y_train[valid], pos_neg_clinical_metabolism_y_valid_predn,
                                average='macro')
    F1_score_valid = f1_score(pos_neg_clinical_metabolism_y_train[valid], pos_neg_clinical_metabolism_y_valid_predn,
                              average='macro')
    pos_neg_clinical_metabolism_accuracy_valid.append(Accuracy_valid)
    pos_neg_clinical_metabolism_precision_valid.append(Precision_valid)
    pos_neg_clinical_metabolism_recall_valid.append(Recall_valid)
    pos_neg_clinical_metabolism_f1_score_valid.append(F1_score_valid)

    if scores[1] >= pos_neg_clinical_metabolism_accuracys[np.argmax(pos_neg_clinical_metabolism_accuracys)]:
        if scores[0] <= pos_neg_clinical_metabolism_losses[np.argmin(pos_neg_clinical_metabolism_losses)]:
            Confusion_matrix_best = pd.DataFrame(c_matrix)
            Confusion_matrix_best.columns = ['pre_neg', 'pre_pos']
            Confusion_matrix_best.index = ['Actual_neg', 'Actual_pos']
            model_test_evalute_best = pd.DataFrame([['Accuracy_valid', Accuracy_valid], ['Precision_valid', Precision_valid],
                                                    ['Recall_valid', Recall_valid], ['f1_score_valid', F1_score_valid]])
            acc_plot_graphs(history, 'accuracy')
            plt.savefig('1-best_valid_acc.pdf')
            plt.close()
            loss_plot_graphs(history, 'loss')
            plt.savefig('1-best_valid_loss.pdf')
            plt.close()
            plot_graphs(history, 'accuracy', 'loss')
            plt.savefig('1-best_valid_acc_loss.pdf')
            plt.close()
            test_acc_loss_best = pd.DataFrame({'accuracy': history.history['accuracy'], 'loss': history.history['loss']})
            with pd.ExcelWriter('2-best_model_valid_evalute_valid.xlsx') as writer:
                model_test_evalute_best.to_excel(writer, sheet_name='model_test_evalute')
                Confusion_matrix_best.to_excel(writer, sheet_name='Confusion_matrix')
                test_acc_loss_best.to_excel(writer, sheet_name='acc_loss')
                pass
            # serialize model to JSON
            model_json = model_kfold.to_json()
            with open("Historical.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model_kfold.save("Historical.h5")

clinical_metabolism_k10fold = pd.DataFrame({'accuracy': pos_neg_clinical_metabolism_accuracys, 'loss': pos_neg_clinical_metabolism_losses,
                                  'accuracy_valid': pos_neg_clinical_metabolism_accuracy_valid, 'precision_valid': pos_neg_clinical_metabolism_precision_valid,
                                  'recall_valid': pos_neg_clinical_metabolism_recall_valid, 'f1_score': pos_neg_clinical_metabolism_f1_score_valid})
clinical_metabolism_k10fold.to_excel('3-10fold_acc_loss.xlsx')

model = load_model("Historical.h5")
model.summary()
pos_neg_clinical_metabolism_y_test_pred = model.predict(pos_neg_clinical_metabolism_x_test_scaled)
pos_neg_clinical_metabolism_y_test_predn = (pos_neg_clinical_metabolism_y_test_pred > 0.5).astype("int32")
pos_neg_c_matrix_test = confusion_matrix(pos_neg_clinical_metabolism_y_test, pos_neg_clinical_metabolism_y_test_predn)
# accuracy，precision，recall_test，f1_score_test
Accuracy_test = accuracy_score(pos_neg_clinical_metabolism_y_test, pos_neg_clinical_metabolism_y_test_predn)
Precision_test = precision_score(pos_neg_clinical_metabolism_y_test, pos_neg_clinical_metabolism_y_test_predn, average='macro')
Recall_test = recall_score(pos_neg_clinical_metabolism_y_test, pos_neg_clinical_metabolism_y_test_predn, average='macro')
F1_score_test = f1_score(pos_neg_clinical_metabolism_y_test, pos_neg_clinical_metabolism_y_test_predn, average='macro')
Confusion_matrix_test = pd.DataFrame(pos_neg_c_matrix_test)
Confusion_matrix_test.columns = ['pre_neg', 'pre_pos']
Confusion_matrix_test.index = ['Actual_neg', 'Actual_pos']
model_test_evalute_test = pd.DataFrame([['Accuracy_valid', Accuracy_test], ['Precision_valid', Precision_test],
                                        ['Recall_valid', Recall_test], ['f1_score_valid', F1_score_test]])

with pd.ExcelWriter(4-test_Confusion_matrix.xlsx") as writer:
    model_test_evalute_test.to_excel(writer, sheet_name='model_test_evalute')
    Confusion_matrix_test.to_excel(writer, sheet_name='Confusion_matrix')
    pass

##############################################################################################################

## 2. Ternary classifiers with MLP
## example SPPT vs SNPT vs Ctrl
pos_neg_con_clinical_metabolism_train = clinical_metabolism_train
pos_neg_con_clinical_metabolism_test = clinical_metabolism_test
pos_neg_con_clinical_metabolism_train['y'] = pos_neg_con_clinical_metabolism_train['y'].map({'pos': 2, 'neg': 1, 'con': 0})
pos_neg_con_clinical_metabolism_test['y'] = pos_neg_con_clinical_metabolism_test['y'].map({'pos': 2, 'neg': 1, 'con': 0})

pos_neg_con_clinical_metabolism_x_train = pos_neg_con_clinical_metabolism_train.loc[:, ['Albumin','X9.OxoODE','DL.Norvaline','Enterostatin.human',
                                                                                        'Ethyl.3.hydroxybutyrate', 'Val.Ser', 'Eicosapentaenoic.acid',
                                                                                        'His.Pro','L.Pyroglutamic.acid','Methoxyacetic.acid']].values
pos_neg_con_clinical_metabolism_y_train = pos_neg_con_clinical_metabolism_train.loc[:, ['y']].values
pos_neg_con_clinical_metabolism_x_test = pos_neg_con_clinical_metabolism_test.loc[:, ['Albumin','X9.OxoODE','DL.Norvaline','Enterostatin.human',
                                                                                        'Ethyl.3.hydroxybutyrate', 'Val.Ser', 'Eicosapentaenoic.acid',
                                                                                        'His.Pro','L.Pyroglutamic.acid','Methoxyacetic.acid']].values
pos_neg_con_clinical_metabolism_y_test = pos_neg_con_clinical_metabolism_test.loc[:, ['y']].values

pos_neg_con_clinical_metabolism_x_train_scaled = pos_neg_con_clinical_metabolism_x_train
pos_neg_con_clinical_metabolism_x_test_scaled = pos_neg_con_clinical_metabolism_x_test

def pos_neg_con_clinical_metabolism_create_model_kfold():
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(15, activation='relu', input_shape=pos_neg_con_clinical_metabolism_x_train_scaled.shape[1:]),
         tf.keras.layers.Dense(15, activation='relu'),
         tf.keras.layers.Dense(3, activation='softmax')
         ]
    )
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

kfold = KFold(n_splits=10,shuffle=True, random_state=6666)
pos_neg_con_clinical_metabolism_accuracys = []
pos_neg_con_clinical_metabolism_losses = []
pos_neg_con_clinical_metabolism_accuracy_valid = []
pos_neg_con_clinical_metabolism_precision_valid = []
pos_neg_con_clinical_metabolism_recall_valid = []
pos_neg_con_clinical_metabolism_f1_score_valid = []
for train, valid in kfold.split(pos_neg_con_clinical_metabolism_x_train_scaled, pos_neg_con_clinical_metabolism_y_train):
    y_class = to_categorical(pos_neg_con_clinical_metabolism_y_train, num_classes=3)
    model_kfold = pos_neg_con_clinical_metabolism_create_model_kfold()
    early_stopping = EarlyStopping(monitor='loss', patience=3)
    history = model_kfold.fit(pos_neg_con_clinical_metabolism_x_train_scaled[train], y_class[train], validation_data=(pos_neg_con_clinical_metabolism_x_train_scaled[valid], y_class[valid]), epochs=250, verbose=0, shuffle=True, callbacks=[early_stopping])
    # evaluate the model
    scores = model_kfold.evaluate(pos_neg_con_clinical_metabolism_x_train_scaled[valid], y_class[valid], verbose=1)
    pos_neg_con_clinical_metabolism_accuracys.append(scores[1])
    pos_neg_con_clinical_metabolism_losses.append(scores[0])
    pos_neg_con_clinical_metabolism_y_valid_pred = model_kfold.predict(pos_neg_con_clinical_metabolism_x_train_scaled[valid])
    pos_neg_con_clinical_metabolism_y_valid_predn = np.argmax(pos_neg_con_clinical_metabolism_y_valid_pred,axis=1)
    c_matrix = confusion_matrix(np.argmax(y_class[valid],axis=1), pos_neg_con_clinical_metabolism_y_valid_predn)
    # accuracy，precision，recall_test，f1_score_test
    Accuracy_valid = accuracy_score(np.argmax(y_class[valid], axis=1), pos_neg_con_clinical_metabolism_y_valid_predn)
    Precision_valid = precision_score(np.argmax(y_class[valid], axis=1), pos_neg_con_clinical_metabolism_y_valid_predn, average='macro')
    Recall_valid = recall_score(np.argmax(y_class[valid], axis=1), pos_neg_con_clinical_metabolism_y_valid_predn, average='macro')
    F1_score_valid = f1_score(np.argmax(y_class[valid], axis=1), pos_neg_con_clinical_metabolism_y_valid_predn, average='macro')
    pos_neg_con_clinical_metabolism_accuracy_valid.append(Accuracy_valid)
    pos_neg_con_clinical_metabolism_precision_valid.append(Precision_valid)
    pos_neg_con_clinical_metabolism_recall_valid.append(Recall_valid)
    pos_neg_con_clinical_metabolism_f1_score_valid.append(F1_score_valid)
    if scores[1] >= pos_neg_con_clinical_metabolism_accuracys[np.argmax(pos_neg_con_clinical_metabolism_accuracys)]:
        if scores[0] <= pos_neg_con_clinical_metabolism_losses[np.argmin(pos_neg_con_clinical_metabolism_losses)]:
            Confusion_matrix_best = pd.DataFrame(c_matrix)
            Confusion_matrix_best.columns = ['pre_con', 'pre_neg', 'pre_pos']
            Confusion_matrix_best.index = ['Actual_con', 'Actual_neg', 'Actual_pos']
            model_test_evalute_best = pd.DataFrame([['Accuracy_valid', Accuracy_valid], ['Precision_valid', Precision_valid],
                                                    ['Recall_valid', Recall_valid], ['f1_score_valid', F1_score_valid]])
            acc_plot_graphs(history, 'accuracy')
            plt.savefig('1-best_valid_acc.pdf')
            plt.close()
            loss_plot_graphs(history, 'loss')
            plt.savefig('1-best_valid_loss.pdf')
            plt.close()
            plot_graphs(history, 'accuracy', 'loss')
            plt.savefig('1-best_valid_acc_loss.pdf')
            plt.close()
            test_acc_loss_best = pd.DataFrame({'accuracy': history.history['accuracy'], 'loss': history.history['loss']})
            with pd.ExcelWriter('2-best_model_valid_evalute_valid.xlsx') as writer:
                model_test_evalute_best.to_excel(writer, sheet_name='model_test_evalute')
                Confusion_matrix_best.to_excel(writer, sheet_name='Confusion_matrix')
                test_acc_loss_best.to_excel(writer, sheet_name='acc_loss')
                pass
            # serialize model to JSON
            model_json = model_kfold.to_json()
            with open("Historical.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model_kfold.save("4-ANN-pos-neg-con\\Historical.h5")

clinical_metabolism_k10fold = pd.DataFrame({'accuracy': pos_neg_con_clinical_metabolism_accuracys, 'loss': pos_neg_con_clinical_metabolism_losses,
                                  'accuracy_valid': pos_neg_con_clinical_metabolism_accuracy_valid, 'precision_valid': pos_neg_con_clinical_metabolism_precision_valid,
                                  'recall_valid': pos_neg_con_clinical_metabolism_recall_valid, 'f1_score': pos_neg_con_clinical_metabolism_f1_score_valid})
clinical_metabolism_k10fold.to_excel('3-10fold_acc_loss.xlsx')

model = load_model("Historical.h5")
model.summary()
pos_neg_con_clinical_metabolism_y_test_pred = model.predict(pos_neg_con_clinical_metabolism_x_test_scaled)
pos_neg_con_clinical_metabolism_y_test_predn = np.argmax(pos_neg_con_clinical_metabolism_y_test_pred,axis=1)
pos_neg_con_clinical_metabolism_y_test_class = to_categorical(pos_neg_con_clinical_metabolism_y_test, num_classes=3)
pos_neg_con_c_matrix_test = confusion_matrix(np.argmax(pos_neg_con_clinical_metabolism_y_test_class,axis=1), pos_neg_con_clinical_metabolism_y_test_predn)
# accuracy，precision，recall_test，f1_score_test
Accuracy_test = accuracy_score(np.argmax(pos_neg_con_clinical_metabolism_y_test_class,axis=1), pos_neg_con_clinical_metabolism_y_test_predn)
Precision_test = precision_score(np.argmax(pos_neg_con_clinical_metabolism_y_test_class,axis=1), pos_neg_con_clinical_metabolism_y_test_predn, average='macro')
Recall_test = recall_score(np.argmax(pos_neg_con_clinical_metabolism_y_test_class,axis=1), pos_neg_con_clinical_metabolism_y_test_predn, average='macro')
F1_score_test = f1_score(np.argmax(pos_neg_con_clinical_metabolism_y_test_class,axis=1), pos_neg_con_clinical_metabolism_y_test_predn, average='macro')
Confusion_matrix_test = pd.DataFrame(pos_neg_con_c_matrix_test)
Confusion_matrix_test.columns = ['pre_con', 'pre_neg', 'pre_pos']
Confusion_matrix_test.index = ['Actual_con', 'Actual_neg', 'Actual_pos']
model_test_evalute_test = pd.DataFrame([['Accuracy_valid', Accuracy_test], ['Precision_valid', Precision_test],
                                        ['Recall_valid', Recall_test], ['f1_score_valid', F1_score_test]])

with pd.ExcelWriter("4-test_Confusion_matrix.xlsx") as writer:
    model_test_evalute_test.to_excel(writer, sheet_name='model_test_evalute')
    Confusion_matrix_test.to_excel(writer, sheet_name='Confusion_matrix')
    pass

##############################################################################################################


