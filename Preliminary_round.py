import pandas as pd
import os
import sys
from collections import defaultdict
import numpy as np


# This function is not allowed to be modified
def read_data(input_file_path):
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Error: File '{input_file_path}' does not exist")
    if not input_file_path.endswith('.csv'):
        raise ValueError("Error: Please provide a CSV file")

    df = pd.read_csv(input_file_path)
    return df


# This function is not allowed to be modified
def output_data(result_df, output_file_path):
    anomaly_fields = ['login_account', 'operation_time', 'anomaly_type']
    missing_anomaly_fields = [field for field in anomaly_fields if field not in result_df.columns]

    if missing_anomaly_fields:
        raise ValueError(f"Missing anomaly detection fields: {', '.join(missing_anomaly_fields)}")

    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_df = result_df[anomaly_fields]
    output_df.to_csv(output_file_path, index=False)
    return True


def process_competition_data(input_file_path, output_file_path):
    try:
        df = read_data(input_file_path)

        # Write your code in the area below.The final output result must be assigned to the variable 'result_df'
        #################################################################################

        # --- 1. 数据预处理与向量化准备 ---
        df['operation_time'] = pd.to_datetime(df['operation_time'], format='%Y/%m/%d %H:%M:%S')
        df.sort_values(['login_account', 'operation_time'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # 预计算所有需要的数组
        times_dt = df['operation_time'].values
        times = times_dt.astype('datetime64[s]').astype(np.int64)
        hours = pd.DatetimeIndex(times_dt).hour.values
        privilege = df['privilege_level'].values
        page_urls = df['page_url'].values.astype(str)
        contents = df['operation_content'].values.astype(str)
        accounts = df['login_account'].values

        # 预计算布尔掩码
        has_products = np.char.find(page_urls, '/products') >= 0
        has_asterisk = np.char.find(contents, '*') >= 0

        # === 规则5 - 向量化优化 ===
        rule5_violations = set()
        # 向量化筛选候选记录
        mask_r5 = (privilege == 1) & has_products & ((hours >= 19) | (hours <= 8))
        
        if mask_r5.any():
            r5_indices = np.where(mask_r5)[0]
            r5_accounts = accounts[r5_indices]
            r5_times = times[r5_indices]
            
            # 按账号分组
            unique_r5_accounts = np.unique(r5_accounts)
            for account in unique_r5_accounts:
                acc_mask = r5_accounts == account
                acc_times = r5_times[acc_mask]
                acc_indices = r5_indices[acc_mask]
                
                # 排序
                sort_order = np.argsort(acc_times)
                acc_times = acc_times[sort_order]
                acc_indices = acc_indices[sort_order]
                
                n = len(acc_times)
                left = 0
                for right in range(n):
                    # 使用秒级时间戳比较
                    while left < right and acc_times[right] - acc_times[left] >= 3600:
                        left += 1
                    if right - left + 1 >= 10:
                        rule5_violations.update(acc_indices[left:right + 1].tolist())

        # === 规则4 - 直接使用预计算的布尔数组 ===
        rule4_violations = set(np.where(has_asterisk)[0].tolist())

        # === 规则 1, 2, 3 - 使用numpy加速 ===
        rule1_violations = set()
        rule2_violations = set()
        rule3_violations = set()
        
        # 获取账号分组信息
        unique_accounts, inverse = np.unique(accounts, return_inverse=True)
        
        for acc_idx, account in enumerate(unique_accounts):
            # 获取该账号的所有记录索引
            mask = inverse == acc_idx
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                continue
            
            # 提取该账号的数据
            acc_times = times[indices]
            acc_pages = page_urls[indices]
            n = len(indices)
            
            left_r1, left_r2, left_r3 = 0, 0, 0
            
            for right in range(n):
                current_time = acc_times[right]
                
                # --- Rule 1 ---
                while left_r1 < right and current_time - acc_times[left_r1] >= 300:
                    left_r1 += 1
                if right - left_r1 + 1 > 7:
                    rule1_violations.update(indices[left_r1:right + 1].tolist())
                
                # --- Rule 2 ---
                while left_r2 < right and current_time - acc_times[left_r2] >= 86400:
                    left_r2 += 1
                if right - left_r2 + 1 >= 80:
                    rule2_violations.update(indices[left_r2:right + 1].tolist())
                
                # --- Rule 3 ---
                while left_r3 < right and current_time - acc_times[left_r3] >= 300:
                    left_r3 += 1
                # 使用numpy向量化计算非敏感页面数量
                window_pages = acc_pages[left_r3:right + 1]
                nonsensitive_count = np.sum(np.char.find(window_pages, '/products') < 0)
                if nonsensitive_count > 7:
                    rule3_violations.update(indices[left_r3:right + 1].tolist())

        # --- 3. 汇总结果 - 并发模型 ---
        all_anomaly_dfs = []

        if rule1_violations:
            viol_list = list(rule1_violations)
            df1 = pd.DataFrame({
                'login_account': accounts[viol_list],
                'operation_time': times_dt[viol_list],
                'anomaly_type': 1
            })
            all_anomaly_dfs.append(df1)

        if rule2_violations:
            viol_list = list(rule2_violations)
            df2 = pd.DataFrame({
                'login_account': accounts[viol_list],
                'operation_time': times_dt[viol_list],
                'anomaly_type': 2
            })
            all_anomaly_dfs.append(df2)

        if rule3_violations:
            viol_list = list(rule3_violations)
            df3 = pd.DataFrame({
                'login_account': accounts[viol_list],
                'operation_time': times_dt[viol_list],
                'anomaly_type': 3
            })
            all_anomaly_dfs.append(df3)

        if rule4_violations:
            viol_list = list(rule4_violations)
            df4 = pd.DataFrame({
                'login_account': accounts[viol_list],
                'operation_time': times_dt[viol_list],
                'anomaly_type': 4
            })
            all_anomaly_dfs.append(df4)

        if rule5_violations:
            viol_list = list(rule5_violations)
            df5 = pd.DataFrame({
                'login_account': accounts[viol_list],
                'operation_time': times_dt[viol_list],
                'anomaly_type': 5
            })
            all_anomaly_dfs.append(df5)

        if not all_anomaly_dfs:
            result_df = pd.DataFrame(columns=['login_account', 'operation_time', 'anomaly_type'])
        else:
            result_df = pd.concat(all_anomaly_dfs, ignore_index=True)
            result_df.sort_values(by=['login_account', 'operation_time', 'anomaly_type'], inplace=True)
            result_df.reset_index(drop=True, inplace=True)

        # 格式化时间输出
        result_df['operation_time'] = result_df['operation_time'].dt.strftime('%Y/%m/%d %H:%M:%S')

        #################################################################################

        output_data(result_df, output_file_path)
        return result_df

    except Exception as e:
        return f"Error during processing: {str(e)}"


if __name__ == "__main__":
    input_csv_path = "b.csv"
    output_csv_path = "2.csv"
    data = process_competition_data(input_csv_path, output_csv_path)