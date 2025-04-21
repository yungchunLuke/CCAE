import numpy as np
def data_augmentation(df, time_steps, window_size, cols=None, random_seed=None):
    # 如果未指定 cols 參數，則預設使用資料框的所有欄位
    if cols is None:
        cols = df.columns

    # 初始化一個空的列表來存放提取出的樣本數據
    samples_list = []

    # 對指定的每一列進行滑動窗口操作
    for col in cols:
        # 根據窗口大小和時間步長，從每列中提取子序列樣本
        for i in range(0, len(df) - time_steps + 1, window_size):
            # 使用 iloc 根據索引提取從 i 到 i + time_steps 的時間段的數據
            # 並將其轉換為 NumPy 陣列，方便進行後續的數據處理
            samples_list.append(df.iloc[i:i + time_steps, col].to_numpy())

    # 將收集到的所有樣本轉換成 NumPy 多維陣列
    final_data = np.array(samples_list)

    # 如果指定了 random_seed，則設置隨機種子，確保數據打亂時的隨機性是可重現的
    if random_seed is not None:
        np.random.seed(random_seed)

    # 返回增強後的數據集，這是一個 NumPy 陣列
    return final_data

