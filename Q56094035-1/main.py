import os
import pandas as pd
import numpy as np
from datetime import timedelta
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--consumption', 
                        help='PATH TO YOUR CONSUMPTION DATA(.csv)')
parser.add_argument('--generation', 
                        help='PATH TO YOUR GENERATION DATA(.csv)')
parser.add_argument('--bidresult', 
                        help='PATH TO YOUR BIDRESULT DATA(.csv)')
parser.add_argument('--output', 
                        default='output.csv',
                        help='OUTPUT FILE NAME(.csv)')


def train():
    root_path = '../training_data'
    file_name = os.listdir(root_path)

    tmp = []

    for name in file_name:
        df = pd.read_csv(os.path.join(root_path, name), names=['time', 'generation', 'consumption'])
        df.drop(0 , inplace=True)
        df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S")
        tmp.append(df)




    test = pd.date_range(start="2018-01-01", end="2018-01-30", freq="D")

    # for i in test:
    start_time = "2018-01-01" # i.strftime("%Y-%m-%d")
    end_time = "2018-08-30" # (i + timedelta(days=1)).strftime("%Y-%m-%d")
    gt_start_time = "2018-01-02"
    gt_end_time = "2018-08-31"
    mask1 = tmp[0]['time'] >= start_time
    mask2 = tmp[0]['time'] < end_time
    day_df = tmp[0][(mask1 & mask2)]
    day_df = day_df.set_index('time')
    time = pd.date_range(start=start_time, end=end_time, freq="H", closed='left')
    all_df = day_df.reindex(time)

    mask1 = tmp[0]['time'] >= gt_start_time
    mask2 = tmp[0]['time'] < gt_end_time
    day_df = tmp[0][(mask1 & mask2)]
    day_df = day_df.set_index('time')
    time = pd.date_range(start=gt_start_time, end=gt_end_time, freq="H", closed='left')
    day_df = day_df.reindex(time)
    gt = day_df['generation'].to_numpy().astype(np.float) - day_df["consumption"].to_numpy().astype(np.float)
    gt = np.expand_dims(gt, axis=0)
    # print(gt_df)

    for idx in range(1, 50):
        mask1 = tmp[idx]['time'] >= start_time
        mask2 = tmp[idx]['time'] < end_time
        day_df = tmp[idx][(mask1 & mask2)]
        day_df = day_df.set_index('time')
        time = pd.date_range(start=start_time, end=end_time, freq="H", closed='left')
        day_df = day_df.reindex(time)
        all_df = pd.concat([all_df, day_df], ignore_index=True, axis=1)

        mask1 = tmp[idx]['time'] >= gt_start_time
        mask2 = tmp[idx]['time'] < gt_end_time
        day_df = tmp[idx][(mask1 & mask2)]
        day_df = day_df.set_index('time')
        time = pd.date_range(start=gt_start_time, end=gt_end_time, freq="H", closed='left')
        day_df = day_df.reindex(time)
        gt = np.concatenate((gt, np.expand_dims(day_df['generation'].to_numpy().astype(np.float) - day_df["consumption"].to_numpy().astype(np.float), axis=0)), axis=0)

    gt = np.sum(gt, axis=0)
    # 0 買 1 賣
    gt = np.where(gt <= 0, 0, gt)
    gt = np.where(gt > 0, 1, gt)
    print("finish preparing data!")

    print(gt)
    x_train, x_test, y_train, y_test = train_test_split(all_df.to_numpy(), gt, test_size=0.2)
    print(x_train)
    print(x_test)
    print(asdf)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    param = {'max_depth':12, 'eta':0.001, 'objective':'multi:softmax', 'num_class': 2, 'lambda': 2, 'gamma': 0.1, 'booster': 'gbtree'}
    num_round = 3000
    bst = xgb.train(param, dtrain, num_round)
    preds = bst.predict(dtest)

    bst.save_model('test.model')

    correct = 0
    for i in range(len(preds)):
        if preds[i] - y_test[i] == 0:
            correct = correct + 1
    print('Accuracy:', correct / len(preds))

def pred(con_path, gen_path, output_path):
    bst = xgb.Booster()
    bst.load_model("test.model")

    
    gen_csv = pd.read_csv(gen_path, names=['time', 'generation'])
    gen_csv.drop(0 , inplace=True)
    gen_csv['time'] = pd.to_datetime(gen_csv['time'], format="%Y-%m-%d %H:%M:%S")


    con_csv = pd.read_csv(con_path, names=['time', 'consumption'])
    con_csv.drop(0 , inplace=True)
    con_csv['time'] = pd.to_datetime(con_csv['time'], format="%Y-%m-%d %H:%M:%S")

    print(gen_csv)
    print(con_csv)
    
    df = pd.merge(gen_csv, con_csv, on="time")

    start_time = df['time'].head(1).tolist()[0].strftime("%Y-%m-%d")
    end_time = (df['time'].tail(1) + timedelta(days=1)).tolist()[0].strftime("%Y-%m-%d")
    time = pd.date_range(start=start_time, end=end_time, freq="H", closed='left')

    
    df = df.set_index('time')
    df = df.reindex(time)
    print(df)

    # root_path = './training_data'
    # file_name = os.listdir(root_path)

    # tmp = []

    # for name in file_name:
    #     df = pd.read_csv(os.path.join(root_path, name), names=['time', 'generation', 'consumption'])
    #     
    #     df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S")
    #     tmp.append(df)

    # test = pd.date_range(start="2018-08-01", end="2018-08-31", freq="D")

    # # for i in test:
    # start_time = "2018-08-01" # i.strftime("%Y-%m-%d")
    # end_time = "2018-08-30" # (i + timedelta(days=1)).strftime("%Y-%m-%d")
    # gt_start_time = "2018-08-02"
    # gt_end_time = "2018-08-31"
    # mask1 = tmp[0]['time'] >= start_time
    # mask2 = tmp[0]['time'] < end_time
    # day_df = tmp[0][(mask1 & mask2)]
    # day_df = day_df.set_index('time')
    # time = pd.date_range(start=start_time, end=end_time, freq="H", closed='left')
    # all_df = day_df.reindex(time)

    # mask1 = tmp[0]['time'] >= gt_start_time
    # mask2 = tmp[0]['time'] < gt_end_time
    # day_df = tmp[0][(mask1 & mask2)]
    # day_df = day_df.set_index('time')
    # time = pd.date_range(start=gt_start_time, end=gt_end_time, freq="H", closed='left')
    # day_df = day_df.reindex(time)
    # gt = day_df['generation'].to_numpy().astype(np.float) - day_df["consumption"].to_numpy().astype(np.float)
    # gt = np.expand_dims(gt, axis=0)

    # for idx in range(1, 40):
    #     mask1 = tmp[idx]['time'] >= start_time
    #     mask2 = tmp[idx]['time'] < end_time
    #     day_df = tmp[idx][(mask1 & mask2)]
    #     day_df = day_df.set_index('time')
    #     time = pd.date_range(start=start_time, end=end_time, freq="H", closed='left')
    #     day_df = day_df.reindex(time)
    #     all_df = pd.concat([all_df, day_df], ignore_index=True, axis=1)

    #     mask1 = tmp[idx]['time'] >= gt_start_time
    #     mask2 = tmp[idx]['time'] < gt_end_time
    #     day_df = tmp[idx][(mask1 & mask2)]
    #     day_df = day_df.set_index('time')
    #     time = pd.date_range(start=gt_start_time, end=gt_end_time, freq="H", closed='left')
    #     day_df = day_df.reindex(time)
    #     gt = np.concatenate((gt, np.expand_dims(day_df['generation'].to_numpy().astype(np.float) - day_df["consumption"].to_numpy().astype(np.float), axis=0)), axis=0)
    
    # gt = np.sum(gt, axis=0)
    # # 0 買 1 賣
    # gt = np.where(gt <= 0, 0, gt)
    # gt = np.where(gt > 0, 1, gt)

    dtest = xgb.DMatrix(df.to_numpy())
    preds = bst.predict(dtest)

    buy = preds == 0
    sell = preds == 1
    action = list(buy)
    for i in range(len(action)):
        if action[i] == True:
            action[i] = "buy"
        else:
            action[i] = "sell"
            
    output = pd.Series(pd.date_range(start=start_time, end=end_time, freq='H', closed="left")).to_frame(name='time')
    output = pd.concat((output, pd.DataFrame(action, columns=['action'])), axis=1, ignore_index=True)
    preds[buy] = 2
    preds[sell] = 10
    output = pd.concat((output, pd.DataFrame(preds, columns=['target_price'])), axis=1, ignore_index=True)
    preds[buy] = 3
    preds[sell] = 4
    output = pd.concat((output, pd.DataFrame(preds, columns=['target_volume'])), axis=1, ignore_index=True)
    output = output.rename(columns={0: 'time', 1: 'action', 2: 'target_price', 3: 'target_volume'})
    output.to_csv(output_path, index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    con = args.consumption
    gen = args.generation
    output = args.output
    train()
    pred(con, gen, output)