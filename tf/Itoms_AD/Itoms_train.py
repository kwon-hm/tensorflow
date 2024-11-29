import os
import numpy as np
import pandas as pd
import seaborn as sns
from time import sleep
from conf import config
import mongoDB as mongo
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from apscheduler.schedulers.background import BackgroundScheduler

print('=============Config start=============')
TIME_STEPS = 6
predictCount = TIME_STEPS * 2
rate = 0.001
shuffle = False
batch_size = 6
validation_split = 0.1
epochs = 1500
THRESHOLD = 0.6
THRESHOLD_CPU = 0.02
THRESHOLD_MEMORY = 0.03
THRESHOLD_NETWORK_R = 0.0001
THRESHOLD_NETWORK_T = 0.0002
THRESHOLD_DISKSIO_R = 0.0004
THRESHOLD_DISKSIO_W = 0.006
table = mongo.getMongoData({'type': 1})
table_name = table[0]['deviceId']
table_dir = table_name + '\\'
path = 'Itoms_AD\\'
save_dir = 'checkpoint_save_' + table_dir
checkpoint_path = path + save_dir + "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

print('TIME_STEPS: ', TIME_STEPS)
print('rate: ', rate)
print('shuffle: ', shuffle)
print('batch_size: ', batch_size)
print('validation_split: ', validation_split)
print('epochs: ', epochs)
print('THRESHOLD_CPU: ', THRESHOLD_CPU)
print('THRESHOLD_MEMORY: ', THRESHOLD_MEMORY)
print('THRESHOLD_NETWORK_R: ', THRESHOLD_NETWORK_R)
print('THRESHOLD_NETWORK_T: ', THRESHOLD_NETWORK_T)
print('THRESHOLD_DISKSIO_R: ', THRESHOLD_DISKSIO_R)
print('THRESHOLD_DISKSIO_W: ', THRESHOLD_DISKSIO_W)
print('checkpoint_path: ', checkpoint_path)
print('=============Config end=============')

scaler = MinMaxScaler()
model = keras.models.Sequential()
sched = BackgroundScheduler()

# 모델의 가중치를 저장하는 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

def train():
    """
    1. Get mongo data
    """
    time, mem, cpu, network_r, network_t, disk_r, disk_w= [], [], [], [], [], [], []

    data_list = mongo.getMongoData({'type': 2, 'table': table_name})
    for i in data_list:
        time.append(datetime.fromtimestamp(i['time']['current'] // 1000))
        cpu.append(i['currentLoad']['currentload'])
        # mem.append(i['mem']['used']/ 1024 / 1024 / 1024)
        mem.append(i['mem']['used'] / i['mem']['total'])
        network_r.append(i['networkStats'][0]['rx_sec'])
        network_t.append(i['networkStats'][0]['tx_sec'])
        disk_r.append(i['disksIO']['rIO_sec'])
        disk_w.append(i['disksIO']['wIO_sec'])

    df = pd.DataFrame()
    df['Date'] = time
    df['Cpu'] = cpu
    df['Mem'] = mem
    df['Network_r'] = network_r
    df['Network_t'] = network_t
    df['DisksIO_r'] = disk_r
    df['DisksIO_w'] = disk_w
    df.set_index('Date', inplace=True) 
    df = df.loc[:,['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']] 
    df = df.astype(float)
    print(df)
    print(df.shape)
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Used')
    plt.plot(df[['Cpu', 'Mem']])
    plt.legend(['Cpu', 'Mem'], loc = 'lower right')
    plt.show()

    """
    2. Model data
    """
    scaler.fit(df)
    df[['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']] = scaler.transform(df[['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']])

    df_train = df[:]

    x_train, y_train = create_dataset(
        df_train[['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']], TIME_STEPS
    )
    print('--------------------------')
    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('--------------------------')

    """
    3. Model & learning
    """
    if(os.path.isfile(path + save_dir + 'saved_model.pb')):
        model = keras.models.load_model(path + save_dir)
    else:
        model = create_model(x_train)
        if(os.path.isfile(checkpoint_path + '.index')):
            model.load_weights(checkpoint_path)
        history = model.fit(
            x_train,
            y_train,
            shuffle=shuffle,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[cp_callback],
            epochs=epochs
        )
        model.save(path + save_dir)
        print('loss: ',history.history['loss']);print()
        plt.figure(figsize = (10,5))
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        # plt.show()
        now = datetime.now()
        plt.savefig(config['imgDir'] + table_dir +  table_name + '_' +now.strftime('%Y-%m-%d %H.%M.%S') +'_loss.png')
        plt.close()

    """
    4. Reconstruction error
    """
    X_train_pred = model.predict(x_train)
    # print('1: ', np.square(X_train_pred - x_train)[0])
    # print('2: ', np.mean(np.square(X_train_pred - x_train), axis=1)[0])
    # print('3: ', np.max(np.mean(np.square(X_train_pred - x_train), axis=1), axis=1)[0])
    train_mae_loss = np.max(np.mean(np.square(X_train_pred - x_train), axis=1), axis=1)
    # train_mae_loss_min = np.min(np.mean(np.square(X_train_pred - x_train), axis=1), axis=1)
    sns.distplot(train_mae_loss, bins=100, kde=True)
    plt.show()

    """
    5. Threshold
    """
    test_score_df = pd.DataFrame(index=df_train[:-(TIME_STEPS)].index)
    test_score_df['loss'] = train_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = train_mae_loss > test_score_df.threshold
    test_score_df['memory'] = df_train[:-(TIME_STEPS)].Mem
    test_score_df['cpu'] = df_train[:-(TIME_STEPS)].Cpu
    test_score_df['n_r'] = df_train[:-(TIME_STEPS)].Network_r
    test_score_df['n_t'] = df_train[:-(TIME_STEPS)].Network_t
    test_score_df['d_r'] = df_train[:-(TIME_STEPS)].DisksIO_r
    test_score_df['d_w'] = df_train[:-(TIME_STEPS)].DisksIO_w

    print('--------------------------')
    print('test_score_df: ', test_score_df)
    print('--------------------------')
    plt.plot(test_score_df.index, test_score_df.loss, label='loss')
    plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
    plt.xticks(rotation=25)
    plt.show()

    anomalies = test_score_df[test_score_df.anomaly == True]
    anomalies.describe()
    print('--------------------------')
    print('anomalies.describe(): ', anomalies.describe())
    print('--------------------------')
    print('anomalies: ', anomalies)
    print('--------------------------')

    """
    6. Anomaly point
    """
    # plt.plot(df_train[:][['DisksIO_w']])
    # plt.plot(
    #     anomalies.index,
    #     anomalies.d_w,
    #     'o',
    #     color='green',
    #     label='anomaly'
    # )
    # plt.xticks(rotation=25)
    # plt.show()

    print('len(df): ', len(df))

    

    # print('len(df_test): ', df_test)

    # mem_, cpu_, network_r_, network_t_, disk_r_, disk_w_, time_ = [], [], [], [], [], [], []
    # test_predic_t, test_predic_c, test_predic_m, test_predic_n_r, test_predic_n_t, test_predic_d_r, test_predic_d_w = [], [], [], [], [], [], []
    # df_old_t, df_old_c, df_old_m, df_old_n_r, df_old_n_t, df_old_d_r, df_old_d_w = [], [], [], [], [], [], []

    # num = 0
    # while num < 1000:
    #     df_Cpu = df[num : num+1]['Cpu']
    #     df_Mem = df[num : num+1]['Mem']
    #     df_Network_r = df[num : num+1]['Network_r']
    #     df_Network_t = df[num : num+1]['Network_t']
    #     df_disksIO_r = df[num : num+1]['DisksIO_r']
    #     df_disksIO_w = df[num : num+1]['DisksIO_w']
    #     df_Date = df.index[num : num+1]

    #     time_.append(df_Date)
    #     cpu_.append(df_Cpu)
    #     mem_.append(df_Mem)
    #     network_r_.append(df_Network_r)
    #     network_t_.append(df_Network_t)
    #     disk_r_.append(df_disksIO_r)
    #     disk_w_.append(df_disksIO_w)
    #     if(len(time_) == TIME_STEPS):
    #         x_test_ = create_test_dataset(time_, cpu_, mem_, network_r_, network_t_, disk_r_, disk_w_, TIME_STEPS)
    #         tp = model.predict(x_test_)
    #         df_data = scaler.inverse_transform(df[num+1 : num+2]) # 실제
    #         predic = scaler.inverse_transform(tp[0]) # 예측
    #         print(num, " 실제 : ", df_data[0][0], df_data[0][1], df_data[0][2], df.index[num+1])
    #         print(num, " 예측 : ", predic[-1][0], predic[-1][1], predic[-1][2])
    #         print()

    #         test_predic_t.append(df.index[num+1])
    #         test_predic_c.append(predic[0][0])
    #         test_predic_m.append(predic[0][1])
    #         test_predic_n_r.append(predic[0][2])
    #         test_predic_n_t.append(predic[0][3])
    #         test_predic_d_r.append(predic[0][4])
    #         test_predic_d_w.append(predic[0][5])

    #         df_old_t.append(df.index[num+1])
    #         df_old_c.append(df_data[0][0])
    #         df_old_m.append(df_data[0][1])
    #         df_old_n_r.append(df_data[0][2])
    #         df_old_n_t.append(df_data[0][3])
    #         df_old_d_r.append(df_data[0][4])
    #         df_old_d_w.append(df_data[0][5])

    #         time_.pop(0)
    #         cpu_.pop(0)
    #         mem_.pop(0)
    #         network_r_.pop(0)
    #         network_t_.pop(0)
    #         disk_r_.pop(0)
    #         disk_w_.pop(0)
    #     num += 1

    # _df_pre = pd.DataFrame()
    # _df_pre['Date'] = test_predic_t
    # _df_pre['Cpu'] = test_predic_c
    # _df_pre['Mem'] = test_predic_m
    # _df_pre['Network_r'] = test_predic_n_r
    # _df_pre['Network_t'] = test_predic_n_t
    # _df_pre['DisksIO_r'] = test_predic_d_r
    # _df_pre['DisksIO_w'] = test_predic_d_w
    # _df_pre.set_index('Date', inplace=True) 
    # _df_pre = _df_pre.loc[:,['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']] 
    # _df_pre = _df_pre.astype(float)

    # _df_old = pd.DataFrame()
    # _df_old['Date'] = df_old_t
    # _df_old['Cpu'] = df_old_c
    # _df_old['Mem'] = df_old_m
    # _df_old['Network_r'] = df_old_n_r
    # _df_old['Network_t'] = df_old_n_t
    # _df_old['DisksIO_r'] = df_old_d_r
    # _df_old['DisksIO_w'] = df_old_d_w
    # _df_old.set_index('Date', inplace=True) 
    # _df_old = _df_old.loc[:,['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']] 
    # _df_old = _df_old.astype(float)

    # plt.title('Model')
    # plt.xlabel('Date')
    # plt.ylabel('Used')
    # plt.plot(_df_old[:]['Cpu'])
    # plt.plot(_df_pre[:]['Cpu'], color='red')
    # plt.legend(['old', 'pre', ], loc = 'lower right')
    # plt.show()

    # print('end.')
    if sched.running == False:
        sched.start()
        sched.add_job(job_predict, 'cron', second='05', id="predict_01")

def create_model(x_train):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(keras.layers.Dropout(rate=rate))
    model.add(keras.layers.RepeatVector(TIME_STEPS))
    model.add(keras.layers.LSTM(64, activation='relu', return_sequences=True))
    model.add(keras.layers.Dropout(rate=rate))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(x_train.shape[2])))
    # model.add(keras.layers.Dense(32))
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    return model

def create_dataset(X, time_steps):
    xs, ys = [], []
    for i in range(len(X) - time_steps):
        xs.append(X[i: (i + time_steps)].values.reshape(time_steps, X.shape[1])),
        ys.append(X[i + 1: (i + time_steps) + 1].values.reshape(time_steps, X.shape[1]))
    return np.array(xs), np.array(ys)

def create_test_dataset(time_data, cpu_data, mem_data, network_r, network_t, disk_r, disk_w, time_steps_data):
    df_ = pd.DataFrame()
    df_['Date'] = time_data
    df_['Cpu'] = cpu_data
    df_['Mem'] = mem_data
    df_['Network_r'] = network_r
    df_['Network_t'] = network_t
    df_['DisksIO_r'] = disk_r
    df_['DisksIO_w'] = disk_w
    df_.set_index('Date', inplace=True) 
    df_ = df_.loc[:,['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']] 
    df_ = df_.astype(float)
    df_test_ = df_[:]
    # df_test_[['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']] = scaler.transform(df_test_[['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']])
    xx = []
    xx.append(df_test_[:].values.reshape(time_steps_data,df_.shape[1]))
    return np.array(xx)

def create_dataset_new(X, time_steps):
    X[['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']] = scaler.transform(X[['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']])
    xx = []
    xx.append(X[:].values.reshape(time_steps, X.shape[1]))
    return np.array(xx)

def job_predict():
    now = datetime.now()
    print("job_predict Run=======: ", now.strftime('%Y-%m-%d %H:%M:%S'))

    mem_, cpu_, network_r_, network_t_, disk_r_, disk_w_, time_ = [], [], [], [], [], [], []
    test_predic_t, test_predic_c, test_predic_m, test_predic_n_r, test_predic_n_t, test_predic_d_r, test_predic_d_w = [], [], [], [], [], [], []
    df_old_t, df_old_c, df_old_m, df_old_n_r, df_old_n_t, df_old_d_r, df_old_d_w = [], [], [], [], [], [], []
    loss_c, loss_m, loss_n_r, loss_n_t, loss_d_r, loss_d_w = [], [], [], [], [], []

    data_count = mongo.getMongoData({'type': 3})
    if(data_count >= TIME_STEPS):
        data_list = mongo.getMongoData({'type': 4, 'time_steps': TIME_STEPS})
        for i, j in enumerate(data_list):
            time_.append(datetime.fromtimestamp(j['time']['current'] // 1000))
            cpu_.append(j['currentLoad']['currentload'])
            mem_.append(j['mem']['used'] / j['mem']['total'])
            network_r_.append(j['networkStats'][0]['rx_sec'])
            network_t_.append(j['networkStats'][0]['tx_sec'])
            disk_r_.append(j['disksIO']['rIO_sec'])
            disk_w_.append(j['disksIO']['wIO_sec'])

            df_old_t.append(datetime.fromtimestamp(j['time']['current'] // 1000))
            df_old_c.append(j['currentLoad']['currentload'])
            df_old_m.append(j['mem']['used'] / j['mem']['total'])
            df_old_n_r.append(j['networkStats'][0]['rx_sec'])
            df_old_n_t.append(j['networkStats'][0]['tx_sec'])
            df_old_d_r.append(j['disksIO']['rIO_sec'])
            df_old_d_w.append(j['disksIO']['wIO_sec'])

            if(i == data_list._Cursor__retrieved -1):
                loss_c.append(0)
                loss_m.append(0) 
                loss_n_r.append(0) 
                loss_n_t.append(0) 
                loss_d_r.append(0) 
                loss_d_w.append(0)
                test_predic_t.append(datetime.fromtimestamp(j['time']['current'] // 1000))
                test_predic_c.append(j['currentLoad']['currentload'])
                test_predic_m.append(j['mem']['used'] / j['mem']['total'])
                test_predic_n_r.append(j['networkStats'][0]['rx_sec'])
                test_predic_n_t.append(j['networkStats'][0]['tx_sec'])
                test_predic_d_r.append(j['disksIO']['rIO_sec'])
                test_predic_d_w.append(j['disksIO']['wIO_sec'])

        num = 0
        while num < predictCount:
            if(len(time_) == TIME_STEPS):
                df_ = pd.DataFrame()
                df_['Date'] = time_
                df_['Cpu'] = cpu_
                df_['Mem'] = mem_
                df_['Network_r'] = network_r_
                df_['Network_t'] = network_t_
                df_['DisksIO_r'] = disk_r_
                df_['DisksIO_w'] = disk_w_
                df_.set_index('Date', inplace=True) 
                df_ = df_.loc[:,['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']] 
                df_ = df_.astype(float)
                # df_[['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']] = scaler.transform(df_[['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']])
                
                x_data = create_dataset_new(df_[['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']], TIME_STEPS)
                pre = model.predict(x_data)

                mae_loss = np.mean(np.square(pre - x_data), axis=1)

                loss_c.append(pre[0][0][0])
                loss_m.append(pre[0][0][1])
                loss_n_r.append(pre[0][0][2])
                loss_n_t.append(pre[0][0][3])
                loss_d_r.append(pre[0][0][4])
                loss_d_w.append(pre[0][0][5])

                predic = scaler.inverse_transform(pre[0])

                test_predic_t.append(time_[-1] + timedelta(minutes=1))
                test_predic_c.append(predic[0][0])
                test_predic_m.append(predic[0][1])
                test_predic_n_r.append(predic[0][2])
                test_predic_n_t.append(predic[0][3])
                test_predic_d_r.append(predic[0][4])
                test_predic_d_w.append(predic[0][5])

                time_.append(time_[-1] + timedelta(minutes=1))
                cpu_.append(predic[0][0])
                mem_.append(predic[0][1])
                network_r_.append(predic[0][2])
                network_t_.append(predic[0][3])
                disk_r_.append(predic[0][4])
                disk_w_.append(predic[0][5])

                time_.pop(0)
                cpu_.pop(0)
                mem_.pop(0)
                network_r_.pop(0)
                network_t_.pop(0)
                disk_r_.pop(0)
                disk_w_.pop(0)

            num += 1

        _df_old = pd.DataFrame()
        _df_old['Date'] = df_old_t
        _df_old['Cpu'] = df_old_c
        _df_old['Mem'] = df_old_m
        _df_old['Network_r'] = df_old_n_r
        _df_old['Network_t'] = df_old_n_t
        _df_old['DisksIO_r'] = df_old_d_r
        _df_old['DisksIO_w'] = df_old_d_w
        _df_old.set_index('Date', inplace=True) 
        _df_old = _df_old.loc[:,['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']] 
        _df_old = _df_old.astype(float)

        _df_pre = pd.DataFrame()
        _df_pre['Date'] = test_predic_t
        _df_pre['Cpu'] = test_predic_c
        _df_pre['Mem'] = test_predic_m
        _df_pre['Network_r'] = test_predic_n_r
        _df_pre['Network_t'] = test_predic_n_t
        _df_pre['DisksIO_r'] = test_predic_d_r
        _df_pre['DisksIO_w'] = test_predic_d_w
        _df_pre.set_index('Date', inplace=True) 
        _df_pre = _df_pre.loc[:,['Cpu', 'Mem', 'Network_r', 'Network_t', 'DisksIO_r', 'DisksIO_w']] 
        _df_pre = _df_pre.astype(float)

        # df_pre_scaler = scaler.transform(_df_pre[:])

        # cpu image
        score = pd.DataFrame()
        score['date'] = test_predic_t
        score['loss'] = loss_c
        score['threshold'] = THRESHOLD
        score['anomaly'] = score.loss > score.threshold
        score['cpu'] = test_predic_c
        score.set_index('date', inplace=True)
        anomalies = score[score.anomaly == True]

        plt.close()
        plt.title('Cpu predict')
        plt.xlabel('Date')
        plt.ylabel('CurrentLoad(%)')
        plt.xticks(rotation=45)
        plt.plot(_df_old[:]['Cpu'], color='cornflowerblue')
        plt.plot(_df_pre[:]['Cpu'], color='orangered')
        plt.plot(
            anomalies.index,
            anomalies.cpu,
            'o',
            color='red',
            label='anomaly'
        )
        plt.legend(['Cpu', 'Predict', 'Anomaly'], loc = 'lower right')
        plt.savefig(config['imgDir'] + table_dir + table_name + '_predict_cpu.png')
        plt.close()

        # memory image
        score = pd.DataFrame()
        score['date'] = test_predic_t
        score['loss'] = loss_m
        score['threshold'] = THRESHOLD
        score['anomaly'] = score.loss > score.threshold
        score['memory'] = test_predic_m
        score.set_index('date', inplace=True)
        anomalies = score[score.anomaly == True]

        plt.title('Memory predict')
        plt.xlabel('Date')
        plt.ylabel('Memory Used(%)')
        plt.xticks(rotation=45)
        plt.plot(_df_old[:]['Mem'], color='orange')
        plt.plot(_df_pre[:]['Mem'], color='orangered')
        plt.plot(
            anomalies.index,
            anomalies.memory,
            'o',
            color='red',
            label='anomaly'
        )
        plt.legend(['Memory', 'Predict', 'Anomaly'], loc = 'lower right')
        plt.savefig(config['imgDir'] + table_dir + table_name + '_predict_memory.png')
        plt.close()

        # network image
        score = pd.DataFrame()
        score['date'] = test_predic_t
        score['loss_r'] = loss_n_r
        score['loss_t'] = loss_n_t
        score['threshold'] = THRESHOLD
        # score['threshold_network_t'] = THRESHOLD_NETWORK_T
        score['anomaly_r'] = score.loss_r > score.threshold
        score['anomaly_t'] = score.loss_t > score.threshold
        # score['anomaly_t'] = df_pre_scaler[:, [3][0]] > score.threshold_network_t
        score['network_r'] = test_predic_n_r
        score['network_t'] = test_predic_n_t
        score.set_index('date', inplace=True)
        anomalies_r = score[score.anomaly_r == True]
        anomalies_t = score[score.anomaly_t == True]

        plt.title('Network predict')
        plt.xlabel('Date')
        plt.ylabel('Network Used')
        plt.xticks(rotation=45)
        plt.plot(_df_old[:]['Network_r'], color='saddlebrown')
        plt.plot(_df_old[:]['Network_t'], color='greenyellow')
        plt.plot(_df_pre[:]['Network_r'], color='orangered')
        plt.plot(_df_pre[:]['Network_t'], color='green')
        plt.plot(
            anomalies_r.index,
            anomalies_r.network_r,
            'o',
            color='red',
            label='anomaly_r'
        )
        plt.plot(
            anomalies_t.index,
            anomalies_t.network_t,
            'o',
            color='black',
            label='anomaly_t'
        )
        plt.legend(['Network_receiv', 'Network_transfer', 'Predict_r', 'Predict_t', 'Anomaly_r', 'Anomaly_t'], loc = 'lower right')
        plt.savefig(config['imgDir'] + table_dir + table_name + '_predict_network.png')
        plt.close()

        # disksIO image
        score = pd.DataFrame()
        score['date'] = test_predic_t
        score['loss_r'] = loss_d_r
        score['loss_w'] = loss_d_w
        score['threshold'] = THRESHOLD
        # score['threshold_disksIO_w'] = THRESHOLD_DISKSIO_W
        score['anomaly_r'] = score.loss_r > score.threshold
        score['anomaly_w'] = score.loss_w > score.threshold
        # score['anomaly_w'] = df_pre_scaler[:, [5][0]] > score.threshold_disksIO_w
        score['disksIO_r'] = test_predic_d_r
        score['disksIO_w'] = test_predic_d_w
        score.set_index('date', inplace=True)
        anomalies_r = score[score.anomaly_r == True]
        anomalies_w = score[score.anomaly_w == True]

        plt.title('Disk predict')
        plt.xlabel('Date')
        plt.ylabel('Disk Used')
        plt.xticks(rotation=45)
        plt.plot(_df_old[:]['DisksIO_r'], color='blueviolet')
        plt.plot(_df_old[:]['DisksIO_w'], color='cyan')
        plt.plot(_df_pre[:]['DisksIO_r'], color='orangered')
        plt.plot(_df_pre[:]['DisksIO_w'], color='green')
        plt.plot(
            anomalies_r.index,
            anomalies_r.disksIO_r,
            'o',
            color='red',
            label='anomaly_r'
        )
        plt.plot(
            anomalies_w.index,
            anomalies_w.disksIO_w,
            'o',
            color='black',
            label='anomaly_w'
        )
        plt.legend(['DisksIO_read', 'DisksIO_write', 'Predict_r', 'Predict_w', 'Anomaly_r', 'Anomaly_w'], loc = 'lower right')
        plt.savefig(config['imgDir'] + table_dir + table_name + '_predict_disk.png')
        plt.close()

        print('end.')
    else:
        print("---- No data ----")
        print('TIME_STEPS: ', TIME_STEPS)
        print('Data_count: ', data_count)
        print("-----------------")
    

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def schedule():
    createFolder(config['imgDir'] + table_dir)
    train()
    # sched.start()
    # sched.add_job(job_predict, 'cron', second='0', id="predict_01")
    while True:
        if(os.path.isfile(path + save_dir + 'saved_model.pb')):
            print("Running main process...............")
        else:
            sched.remove_job("predict_01")
            train()
        sleep(5)

schedule()