import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from lightgbm.sklearn import LGBMClassifier
from collections import defaultdict
import os
import gc
import time
import psutil

pd.set_option("display.max_columns", None)

## 打印当前内存占用
def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print("{} memory used: {} MB".format(hint, memory))

## 优化内存空间
def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in tqdm(cols):
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("{:.2f} Mb, {:.2f} Mb ({:.2f} %)".format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df

## importance plot
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:50].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('features importance')
    plt.tight_layout()
    plt.savefig('lgbm_importances04.png')

## 从官方baseline里面抽出来的评测函数
def uAUC(labels, preds, user_id_list):
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])

    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)

    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0

    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc
            size += 1.0

    user_auc = float(total_auc)/size
    return user_auc

## 定义全局变量
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward", "favorite", "comment", "follow"]
PLAY_COLS = ["is_finish", "play_times", "play", "stay"]
CATEGORY_COLS = ["feedid", "authorid", "userid", "device", "bgm_song_id", "bgm_singer_id"]
FEED_COLS = ["feedid", "authorid", "videoplayseconds", "machine_tag_list", "bgm_song_id", "bgm_singer_id", "machine_keyword_list"]
ACTION_SAMPLE_RATE = {"read_comment": 0.1, "like": 0.2, "click_avatar": 0.2, "forward": 0.1, "comment": 0.1, "follow": 0.1, "favorite": 0.1}
MAX_DAY = 15
VALIDATE_DAY = 14
PAST_DAYS = 7
TOWARD_DAYS = 3

## 读取训练集
train_raw = pd.read_csv("data/user_action.csv")
print(train_raw.shape)
for y in ACTION_LIST:
    print(y, train_raw[y].mean())

## 读取测试集
test_raw = pd.read_csv("data/test_a.csv")
test_raw["date_"] = MAX_DAY
print(test_raw.shape)

## 合并处理
df = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)
print(df.head(3))
del train_raw, test_raw
gc.collect()

## 下采样
# temp_df = df[(df["date_"] >= VALIDATE_DAY)]
# for action in ACTION_LIST[:1]:
#     action_df = df[(df["date_"] < VALIDATE_DAY)]
#     df_neg = action_df[action_df[action] == 0]
#     df_pos = action_df[action_df[action] == 1]
#     df_neg = df_neg.sample(frac=ACTION_SAMPLE_RATE[action], random_state=2021, replace=False)
#     temp_df = pd.concat([temp_df, df_neg, df_pos])
# df = temp_df
# del temp_df
# gc.collect()

## 读取视频信息表
feed_info = pd.read_csv("data/feed_info.csv")

## 此份baseline只保留这X列
feed_info = feed_info[FEED_COLS]
df = df.merge(feed_info, on="feedid", how="left")
del feed_info
gc.collect()

## 视频时长是秒，转换成毫秒，才能与play、stay做运算
df["videoplayseconds"] *= 1000

## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
df["is_finish"] = (df["play"] >= df["videoplayseconds"]).astype("int8")
df["play_times"] = df["play"] / df["videoplayseconds"]

## 统计最大概率tag keywords
df["tags"] = df["machine_tag_list"].apply(lambda x: max(map(lambda y: y.split(" "), str(x).split(";")),
                                                        key=lambda z: float(z[1]) if len(z) > 1 else "0")[0])
df["keywords"] = df["machine_keyword_list"].apply(lambda x: str(x).split(";")[0])

## 读取feed_embedding pca
feed_embedding = pd.read_csv("data/feed_embeddings.csv")
temp = list(feed_embedding["feed_embedding"].apply(lambda x: list(map(float, x.strip().split(" ")))))
pca = PCA(n_components=32)
X_r = pca.fit(temp).transform(temp).T
feed_embedding = feed_embedding.drop(columns="feed_embedding")

for i, j in enumerate(X_r):
    feed_embedding["pca{}".format(i)] = j

df = df.merge(feed_embedding, on="feedid", how="left")
del feed_embedding, temp
gc.collect()

df[df.select_dtypes("float").columns] = df.select_dtypes("float").apply(pd.to_numeric, downcast="float")

## 统计历史X天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
for stat_cols in tqdm([
    ["device"],
    ["userid"],
    ["feedid"],
    ["authorid"],
    ["tags"],
    ["keywords"],
    ["bgm_singer_id"],
    ["bgm_song_id"],
    ["userid", "authorid"],
    ["userid", "tags"],
    ["userid", "bgm_singer_id"],
    ["userid", "bgm_song_id"]
]):

    f = "_".join(stat_cols)
    stat_df = pd.DataFrame()

    for target_day in range(PAST_DAYS + 1, MAX_DAY + 1):
        left, right = target_day - PAST_DAYS, target_day - 1
        tmp = df[((df["date_"] >= left) & (df["date_"] <= right))].reset_index(drop=True)
        tmp["date_"] = target_day

        g = tmp.groupby(stat_cols)
        feats = []

        # if stat_cols not in [["device"]]:
        #     tmp = tmp.merge(g.size().reset_index(name="{}_count".format(f)), on=stat_cols, how="left")
        #     feats.append("{}_count".format(f))

        # if stat_cols in [["userid", "authorid"], ["userid", "tags"]]:
        #     tmp["{}_in_{}".format(f, stat_cols[0])] = tmp["{}_count".format(f)] / (tmp["{}_count".format(stat_cols[0])] + 1)
        #     tmp["{}_in_{}".format(f, stat_cols[1])] = tmp["{}_count".format(f)] / (tmp["{}_count".format(stat_cols[1])] + 1)
        #     feats.append("{}_in_{}".format(f, stat_cols[0]))
        #     feats.append("{}_in_{}".format(f, stat_cols[1]))

        if stat_cols not in [["device"], ["userid", "bgm_singer_id"], ["userid", "bgm_song_id"]]:
            tmp["{}_{}day_count".format(f, PAST_DAYS)] = g["date_"].transform("count")
            feats.append("{}_{}day_count".format(f, PAST_DAYS))

        # for x in PLAY_COLS[:3]:
        #     tmp["{}_{}day_{}_rate".format(f, PAST_DAYS, x)] = g[x].transform("mean")
        #     feats.append("{}_{}day_{}_rate".format(f, PAST_DAYS, x))


        # if stat_cols not in [["device"], ["feedid"]]:
        #     tmp["{}_{}day_videoplayseconds_mean".format(f, PAST_DAYS)] = g["videoplayseconds"].transform("mean")
        #     feats.append("{}_{}day_videoplayseconds_mean".format(f, PAST_DAYS))

        if stat_cols in [["userid"]]:
            for x in PLAY_COLS[1:]:
                for stat in ["max", "mean", "min", "std", "median"]:
                    tmp["{}_{}day_{}_{}".format(f, PAST_DAYS, x, stat)] = g[x].transform(stat)
                    feats.append("{}_{}day_{}_{}".format(f, PAST_DAYS, x, stat))

        for y in ACTION_LIST:
            tmp["{}_{}day_{}_sum".format(f, PAST_DAYS, y)] = g[y].transform("sum")
            tmp["{}_{}day_{}_mean".format(f, PAST_DAYS, y)] = g[y].transform("mean")
            feats.extend(["{}_{}day_{}_sum".format(f, PAST_DAYS, y), "{}_{}day_{}_mean".format(f, PAST_DAYS, y)])

        tmp = tmp[stat_cols + feats + ["date_"]].drop_duplicates(stat_cols + ["date_"]).reset_index(drop=True)
        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
        del g, tmp

    df = df.merge(stat_df, on=stat_cols + ["date_"], how="left")
    del stat_df
    gc.collect()

df = df.drop(columns=["play_times", "is_finish", "play", "stay", "tags", "keywords",
                      "machine_tag_list", "machine_keyword_list", "authorid", "device", "bgm_song_id", "bgm_singer_id"])

## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行
# for f in tqdm(["userid", "feedid", "authorid"]):
#     df[f + "_count"] = df[f].map(df[f].value_counts())
#
# for f1, f2 in tqdm([
#     ["userid", "feedid"],
#     ["userid", "authorid"]
# ]):
#
#     df["{}_in_{}_nunique".format(f1, f2)] = df.groupby(f2)[f1].transform("nunique")
#     df["{}_in_{}_nunique".format(f2, f1)] = df.groupby(f1)[f2].transform("nunique")
#
# for f1, f2 in tqdm([
#     ["userid", "authorid"]
# ]):
#
#     df["{}_{}_count".format(f1, f2)] = df.groupby([f1, f2])["feedid"].transform("count")
#     df["{}_in_{}_count_prop".format(f1, f2)] = df["{}_{}_count".format(f1, f2)] / (df[f2 + "_count"] + 1)
#     df["{}_in_{}_count_prop".format(f2, f1)] = df["{}_{}_count".format(f1, f2)] / (df[f1 + "_count"] + 1)
#
# df["videoplayseconds_in_userid_mean"] = df.groupby("userid")["videoplayseconds"].transform("mean")
# df["videoplayseconds_in_authorid_mean"] = df.groupby("authorid")["videoplayseconds"].transform("mean")
# df["feedid_in_authorid_nunique"] = df.groupby("authorid")["feedid"].transform("nunique")
#
# df = df.drop(columns=["feedid", "authorid"])

## 内存够用的不需要做这一步
show_memory_info("step9")
df = reduce_mem(df, [f for f in df.columns if f not in ["date_"] + ACTION_LIST])

## 定义训练集验证集
train = df[~df["read_comment"].isna()].reset_index(drop=True)
test = df[df["read_comment"].isna()].reset_index(drop=True)

cols = [f for f in df.columns if f not in ["date_", "userid", "feedid"] + ACTION_LIST]
del df
gc.collect()

print(train[cols].shape)
print(cols)

#trn_x = train[train["date_"] < VALIDATE_DAY].reset_index(drop=True)
trn_x = train[(train["date_"] >= PAST_DAYS + 1) & (train["date_"] < VALIDATE_DAY)].reset_index(drop=True)
val_x = train[train["date_"] == VALIDATE_DAY].reset_index(drop=True)

##################### 线下验证 #####################
uauc_list = []
r_list = []
feature_importance_df = pd.DataFrame()

for y in ACTION_LIST[:4]:
    print("=========", y, "=========")

    t = time.time()
    clf = LGBMClassifier(
        learning_rate=0.05,
        #class_weight="balanced",
        n_estimators=5000,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2021,
        metric="None"
    )

    clf.fit(
        trn_x[cols], trn_x[y],
        eval_set=[(val_x[cols], val_x[y])],
        eval_metric="auc",
        #categorical_feature=CATEGORY_COLS,
        early_stopping_rounds=100,
        verbose=50,
    )

    val_x[y + "_score"] = clf.predict_proba(val_x[cols])[:, 1]
    val_uauc = uAUC(val_x[y], val_x[y + "_score"], val_x["userid"])
    uauc_list.append(val_uauc)
    print(val_uauc)
    r_list.append(clf.best_iteration_)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = cols
    fold_importance_df["importance"] = clf.feature_importances_
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print("runtime: {}\n".format(time.time() - t))

display_importances(feature_importance_df)
weighted_uauc = 0.4 * uauc_list[0] + 0.3 * uauc_list[1] + 0.2 * uauc_list[2] + 0.1 * uauc_list[3]
del trn_x, val_x
gc.collect()
print(uauc_list)
print(weighted_uauc)

#################### 全量训练 #####################
trn_all = train[(train["date_"] >= PAST_DAYS + 1) & (train["date_"] < MAX_DAY)].reset_index(drop=True)
r_dict = dict(zip(ACTION_LIST[:4], r_list))

for y in ACTION_LIST[:4]:
    print("=========", y, "=========")

    t = time.time()
    clf = LGBMClassifier(
        learning_rate=0.05,
        n_estimators=r_dict[y],
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2021
    )

    clf.fit(
        trn_all[cols], trn_all[y],
        eval_set=[(trn_all[cols], trn_all[y])],
        early_stopping_rounds=r_dict[y],
        verbose=100
    )

    test[y] = clf.predict_proba(test[cols])[:, 1]
    print("runtime: {}\n".format(time.time() - t))

test[["userid", "feedid"] + ACTION_LIST[:4]].to_csv(
    "sub_%.6f_%.6f_%.6f_%.6f_%.6f.csv" % (weighted_uauc, uauc_list[0], uauc_list[1], uauc_list[2], uauc_list[3]),index=False
)