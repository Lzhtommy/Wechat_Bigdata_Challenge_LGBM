import pandas as pd

ACTION_LIST = ["read_comment", "like", "click_avatar", "forward", "favorite", "comment", "follow"]
lgb = pd.read_csv("data/test_a.csv")
deepfm = pd.read_csv("submit_base_deepfm.csv")

result = pd.DataFrame()
result[["userid", "feedid"]] = lgb[["userid", "feedid"]]
result[ACTION_LIST[:1]] = 0.3 * lgb[ACTION_LIST[:1]] + 0.7 * deepfm[ACTION_LIST[:1]]
result[ACTION_LIST[1:4]] = 0.7 * lgb[ACTION_LIST[1:4]] + 0.3 * deepfm[ACTION_LIST[1:4]]
result.to_csv("aggregate.csv", index=False)