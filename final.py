from __future__ import annotations

import os
import sys
import json
import traceback
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO


# =========================
# 0) 全局：忽略 warnings（你要求“输出中忽略 warning”）
# =========================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# 如果你想彻底不显示任何 warning（更激进）：
# warnings.filterwarnings("ignore")


# =========================
# 0) 全局配置：你的项目目录 + 文件名
# =========================
PROJECT_DIR = "/Users/c1c/Desktop/梧桐杯/省级决赛/project"

STUDENT_FILE_TASK1 = "new_student_model.xlsx"
FRAUD_1_1 = "fraud_model_1_1.xlsx"
FRAUD_1_2 = "fraud_model_1_2.xlsx"
FRAUD_2   = "fraud_model_2.xlsx"
VALIDATE_FILE = "new_validata_data.xlsx"

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 更稳一点的默认


# =========================================================
# 1) 工具：自动读取 xlsx/csv
# =========================================================
def read_table_auto(path: str):
    import pandas as pd
    import pathlib

    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p.resolve()}")

    suf = p.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(p)

    with open(p, "rb") as f:
        head = f.read(4)
    if head[:2] == b"PK":
        return pd.read_excel(p)

    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(p)


# =========================================================
# 2) 分区输出捕获
# =========================================================
@dataclass
class TaskResult:
    name: str
    ok: bool
    stdout: str
    stderr: str
    exception: Optional[str] = None


def _section(title: str) -> None:
    bar = "=" * 18
    print(f"\n{bar} {title} {bar}\n")


def run_task(task_name: str, fn: Callable[[], None]) -> TaskResult:
    out_buf = StringIO()
    err_buf = StringIO()
    ok = True
    exc_txt = None

    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            fn()
    except Exception:
        ok = False
        exc_txt = traceback.format_exc()

    return TaskResult(
        name=task_name,
        ok=ok,
        stdout=out_buf.getvalue(),
        stderr=err_buf.getvalue(),
        exception=exc_txt
    )


def print_task_result(res: TaskResult) -> None:
    status = "OK" if res.ok else "FAILED"
    _section(f"[{status}] {res.name}")

    if res.exception:
        print("--- exception (captured) ---")
        print(res.exception.rstrip("\n"))
        print()

    print("--- stdout (captured) ---")
    sys.stdout.write(res.stdout)
    print("\n--- stderr (captured) ---")
    sys.stdout.write(res.stderr)
    print()


# =========================================================
# 3) Task 1（保持你原逻辑）
# =========================================================
def task_1():
    import pandas as pd
    import numpy as np

    df = read_table_auto(STUDENT_FILE_TASK1)

    df['is_risky'] = df.apply(
        lambda x: 1 if pd.notnull(x.get('fraud_msisdn')) or (x.get('voice_receive', 0) > 0) else 0,
        axis=1
    )

    print(f"总学生数: {len(df)}")
    print(f"高危/受害学生数: {df['is_risky'].sum()}")
    print(f"高危比例: {df['is_risky'].mean():.2%}")

    print("\n--- 猜想一：不同身份的中招率 ---")
    if 'hk_resident_type' in df.columns:
        risk_by_type = df.groupby('hk_resident_type')['is_risky'].agg(['count', 'mean']).sort_values(by='mean', ascending=False)
        risk_by_type.columns = ['人数', '高危比例']
        print(risk_by_type)

    print("\n--- 猜想二：大陆APP使用天数与风险的关系 ---")
    if 'app_max_cnt' in df.columns:
        df['app_max_cnt'] = pd.to_numeric(df['app_max_cnt'], errors='coerce').fillna(0)
        df['app_usage_level'] = pd.cut(df['app_max_cnt'], bins=[-1, 0, 7, 20, 100], labels=['不使用', '低频', '中频', '高频'])
        risk_by_app = df.groupby('app_usage_level')['is_risky'].agg(['count', 'mean'])
        print(risk_by_app)

    print("\n--- 猜想三：接听陌生境外电话次数与风险的关系 ---")
    if 'total_voice_cnt' in df.columns:
        df['total_voice_cnt'] = pd.to_numeric(df['total_voice_cnt'], errors='coerce').fillna(0)
        df['foreign_call_level'] = pd.cut(df['total_voice_cnt'], bins=[-1, 0, 5, 1000], labels=['无', '少', '多'])
        risk_by_call = df.groupby('foreign_call_level')['is_risky'].agg(['count', 'mean'])
        print(risk_by_call)

    print("\n--- 猜想四：不同年龄段的中招率 ---")
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        age_bins = [0, 18, 22, 26, 30, 100]
        age_labels = ['<18岁', '18-22岁', '23-26岁', '27-30岁', '>30岁']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
        risk_by_age = df.groupby('age_group')['is_risky'].agg(['count', 'mean'])
        print(risk_by_age)

    def get_id_score(identity):
        ident = str(identity)
        if '新来港港人' in ident:
            return 50
        elif '香港人' in ident:
            return 40
        elif '内地' in ident:
            return 5
        else:
            return 10

    def get_app_score(days):
        if days <= 0:
            return 0
        elif days <= 7:
            return 30
        elif days <= 20:
            return 20
        else:
            return 10

    def get_call_score(cnt):
        return 20 if cnt > 0 else 0

    if 'hk_resident_type' in df.columns:
        df['score_id'] = df['hk_resident_type'].apply(get_id_score)
    else:
        df['score_id'] = 10

    if 'app_max_cnt' in df.columns:
        df['app_max_cnt'] = pd.to_numeric(df['app_max_cnt'], errors='coerce').fillna(0)
        df['score_app'] = df['app_max_cnt'].apply(get_app_score)
    else:
        df['score_app'] = 0

    if 'total_voice_cnt' in df.columns:
        df['total_voice_cnt'] = pd.to_numeric(df['total_voice_cnt'], errors='coerce').fillna(0)
        df['score_call'] = df['total_voice_cnt'].apply(get_call_score)
    else:
        df['score_call'] = 0

    df['risk_score'] = df['score_id'] + df['score_app'] + df['score_call']

    target_count = 1000
    potential_victims = df.sort_values(by='risk_score', ascending=False).head(target_count)

    if 'voice_call' in df.columns:
        df['is_confirmed_victim'] = df['voice_call'] > 0
        confirmed_victims = df[df['is_confirmed_victim'] == True]
        remaining_count = target_count - len(confirmed_victims)

        if remaining_count > 0:
            high_score_candidates = df[~df['is_confirmed_victim']].sort_values(by='risk_score', ascending=False).head(remaining_count)
            potential_victims = pd.concat([confirmed_victims, high_score_candidates])
        else:
            potential_victims = confirmed_victims.head(target_count)

    print(f"\n基于评分模型筛选出的高危人数: {len(potential_victims)}")
    print(f"入选者最低分数: {potential_victims['risk_score'].min()}")
    if 'hk_resident_type' in potential_victims.columns:
        print(f"名单构成:\n{potential_victims['hk_resident_type'].value_counts().head()}")
    else:
        print("名单构成:\n(hk_resident_type 列不存在)")


# =========================================================
# 4) Task 2（保持你原逻辑：不新增 pred_score）
# =========================================================
def task_2():
    import numpy as np
    import pandas as pd
    from scipy import sparse
    import xgboost as xgb
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

    def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        today = pd.Timestamp.today().normalize()

        if "open_dt" in df.columns:
            s = pd.to_numeric(df["open_dt"], errors="coerce").round(0).astype("Int64")
            dt = pd.to_datetime(s.astype("string"), format="%Y%m%d", errors="coerce")
            df["account_tenure_days"] = (today - dt).dt.days
            df.loc[df["account_tenure_days"] < 0, "account_tenure_days"] = 0
            df = df.drop(columns=["open_dt"])

        if "proc_time" in df.columns:
            s_num = pd.to_numeric(df["proc_time"], errors="coerce").round(0).astype("Int64")
            dt = pd.to_datetime(s_num.astype("string"), format="%Y%m%d", errors="coerce")
            df["proc_time_has_value"] = dt.notna().astype(int)
            df["proc_age_days"] = (today - dt).dt.days
            df["proc_age_days"] = df["proc_age_days"].fillna(0)
            df.loc[df["proc_age_days"] < 0, "proc_age_days"] = 0
            df["proc_age_days"] = df["proc_age_days"].astype(int)
            df = df.drop(columns=["proc_time"])

        return df

    NUM_COLS = [
        "iden_type_num", "change_imei_times",
        "dispersion_rate", "call_cnt_day_2s", "avg_actv_dur",
        "call_cnt_day", "tot_msg_cnt",
        "face_val",
        "mth_fee", "opp_num_stu_cnt",
        "call_stu_cnt", "rechrg_for_ppd",
        "cellsite_duration", "vas_ofr_id_num",
        "account_tenure_days",
        "proc_age_days",
        "call_cnt_day_3m"
    ]
    NUM_COLS = list(dict.fromkeys(NUM_COLS))

    CAT_COLS = [
        "is_1cmn", "post_or_ppd",
        "chnl_class",
        "audit_status", "hit_student_model",
    ]

    print(f"DEBUG: 配置已重置。将在后续严格使用这 {len(NUM_COLS)} 个数值特征，不包含其他任何特征。")

    def fit_transform_preprocess(Xdf: pd.DataFrame):
        Xdf = add_date_features(Xdf)

        use_num = [c for c in NUM_COLS if c in Xdf.columns]
        use_cat = [c for c in CAT_COLS if c in Xdf.columns]

        X_num = Xdf[use_num].apply(pd.to_numeric, errors="coerce")
        num_median = X_num.median(numeric_only=True)
        X_num = X_num.fillna(num_median)

        num_mean = X_num.mean()
        num_std = X_num.std(ddof=0).replace(0, 1.0)
        X_num = (X_num - num_mean) / num_std
        X_num_sp = sparse.csr_matrix(X_num.to_numpy(dtype=np.float32))

        if use_cat:
            X_cat = Xdf[use_cat].astype("string").fillna("NA")
            X_cat_dum = pd.get_dummies(X_cat, columns=use_cat, dummy_na=False)
            cat_feature_names = list(X_cat_dum.columns)
            X_cat_sp = sparse.csr_matrix(X_cat_dum.to_numpy(dtype=np.float32))
        else:
            cat_feature_names = []
            X_cat_sp = sparse.csr_matrix((len(Xdf), 0))

        X_all = sparse.hstack([X_num_sp, X_cat_sp], format="csr")

        state = {
            "use_num": use_num,
            "num_median": num_median,
            "num_mean": num_mean,
            "num_std": num_std,
            "use_cat": use_cat,
            "cat_feature_names": cat_feature_names,
        }
        return X_all, state

    def transform_preprocess(Xdf: pd.DataFrame, state: dict):
        Xdf = add_date_features(Xdf)

        use_num = state["use_num"]
        use_cat = state["use_cat"]
        cat_feature_names = state["cat_feature_names"]

        X_num = Xdf.reindex(columns=use_num).apply(pd.to_numeric, errors="coerce")
        X_num = X_num.fillna(state["num_median"])
        X_num = (X_num - state["num_mean"]) / state["num_std"]
        X_num_sp = sparse.csr_matrix(X_num.to_numpy(dtype=np.float32))

        if use_cat:
            X_cat = Xdf.reindex(columns=use_cat).astype("string").fillna("NA")
            X_cat_dum = pd.get_dummies(X_cat, columns=use_cat, dummy_na=False)
            for c in cat_feature_names:
                if c not in X_cat_dum.columns:
                    X_cat_dum[c] = 0
            X_cat_dum = X_cat_dum.reindex(columns=cat_feature_names, fill_value=0)
            X_cat_sp = sparse.csr_matrix(X_cat_dum.to_numpy(dtype=np.float32))
        else:
            X_cat_sp = sparse.csr_matrix((len(Xdf), 0))

        X_all = sparse.hstack([X_num_sp, X_cat_sp], format="csr")
        return X_all

    def evaluate_binary(y_true, y_prob, threshold=0.01):
        y_pred = (y_prob >= threshold).astype(int)
        metrics = {
            "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
        return metrics

    print("正在读取数据...")
    df_1_1 = read_table_auto(FRAUD_1_1); df_1_1["target"] = 0
    df_1_2 = read_table_auto(FRAUD_1_2); df_1_2["target"] = 0
    df_2   = read_table_auto(FRAUD_2);   df_2["target"] = 1

    df = pd.concat([df_1_1, df_1_2, df_2], ignore_index=True)

    keep_cols = ["target", "open_dt", "proc_time"] + NUM_COLS + CAT_COLS
    keep_cols = list(dict.fromkeys(keep_cols))
    final_keep = [c for c in keep_cols if c in df.columns]
    df = df[final_keep].copy()

    print(f"数据清洗完毕，当前数据列: {list(df.columns)}")

    y = df["target"].astype(int).to_numpy()
    Xdf = df.drop(columns=["target"])

    if "user_id" in df.columns:
        groups = df["user_id"].astype(str).to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(gss.split(Xdf, y, groups=groups))
    else:
        tr_idx, te_idx = train_test_split(list(range(len(df))), test_size=0.2, random_state=42, stratify=y)

    X_train_df = Xdf.iloc[tr_idx].copy()
    y_train = y[tr_idx]
    X_test_df = Xdf.iloc[te_idx].copy()
    y_test = y[te_idx]

    print("正在特征工程...")
    X_train, state = fit_transform_preprocess(X_train_df)
    X_test = transform_preprocess(X_test_df, state)

    print(f"DEBUG: 最终进入模型的特征列表: {state['use_num'] + state['cat_feature_names']}")

    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    print("正在训练模型（XGBoost）...")
    clf = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary(y_test, y_prob, threshold=0.01)
    print("\n测试集指标:")
    print(metrics)

    print(f"\n正在读取待打标任务数据: {VALIDATE_FILE}")
    task_df = read_table_auto(VALIDATE_FILE)

    if not task_df.empty:
        X_task = transform_preprocess(task_df.copy(), state)
        task_prob = clf.predict_proba(X_task)[:, 1]
        THRESHOLD = 0.01
        task_df["label"] = (task_prob >= THRESHOLD).astype(int)

        print("\n待打标预测完成，label 分布:")
        print(task_df["label"].value_counts())

        task_df.to_excel("final_submit_xgb.xlsx", index=False)
        print("\n已保存 final_submit_xgb.xlsx（仅新增 label 一列）")
    else:
        print("未找到待打标任务数据，跳过。")


# =========================================================
# 5) Task 3 / Task 4 / Task 5（保持你原样）
# =========================================================
def task_3():
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import warnings
    import os
    warnings.filterwarnings('ignore')

    def student_fraud_analysis_final():
        print("Step 1: 正在自动搜索并加载本地数据文件 (.xlsx)...")

        def load_file(keyword):
            files = [f for f in os.listdir('.') if keyword.lower() in f.lower()]
            if not files:
                return None
            target_file = next((f for f in files if f.endswith('.xlsx')), files[0])
            print(f"   -> 发现并加载: {target_file}")
            if target_file.endswith('.xlsx'):
                return pd.read_excel(target_file)
            else:
                for enc in ['utf-8', 'gbk', 'utf-8-sig']:
                    try:
                        return pd.read_csv(target_file, encoding=enc)
                    except:
                        continue
            return None

        df_fraud = load_file('fraud_model_2')
        df_student = load_file('new_student_model')

        if df_fraud is None or df_student is None:
            print("❌ 错误：在当前目录下找不到匹配的数据文件。请检查文件名是否正确。")
            return

        df_fraud.columns = df_fraud.columns.str.strip()
        df_student.columns = df_student.columns.str.strip()

        hit_fraud_msisdns = df_student['fraud_msisdn'].dropna().unique()
        df_fraud['hit_target'] = df_fraud['msisdn'].isin(hit_fraud_msisdns).astype(int)

        print("Step 2: 正在运行全特征权重计算模型...")
        exclude = ['user_id', 'msisdn', 'hit_target', 'stat_dt', 'open_dt', 'proc_time', 'acct_id']
        features = [c for c in df_fraud.columns if c in df_fraud.select_dtypes(include=[np.number, 'object']).columns and c not in exclude]

        X = df_fraud[features].copy()
        y = df_fraud['hit_target']

        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].fillna('Unknown').astype(str))
            else:
                X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        weights = pd.DataFrame({'维度特征': features, '权重占比': rf.feature_importances_}).sort_values('权重占比', ascending=False)

        matched = df_student.merge(df_fraud, left_on='fraud_msisdn', right_on='msisdn', suffixes=('_stu', '_fraud'))

        print("\n" + "="*70)
        print("【一、关联匹配与重复性总结】")
        print(f"1. 诈骗库总号码基数: {df_fraud['msisdn'].nunique()}")
        print(f"2. 在学生库中成功关联到的‘活跃诈骗号’: {len(hit_fraud_msisdns[np.isin(hit_fraud_msisdns, df_fraud['msisdn'].unique())])}")
        print(f"3. 关联产生的攻击行为总样本: {len(matched)} 条")

        attack_counts = matched.groupby('fraud_msisdn').size()
        print(f"4. 关联号码的平均重复攻击频次: {attack_counts.mean():.2f} 人次/号")
        print(f"5. 高危重复号码 TOP 3 触达人数:")
        print(attack_counts.sort_values(ascending=False).head(3).to_string())

        print("\n" + "="*70)
        print("【二、诈骗号识别全特征权重排名 (Top 15)】")
        print(weights.head(15).to_string(index=False))

        print("\n" + "="*70)
        print("【三、基于关联路径的策略建议】")

        stu_res_col = [c for c in matched.columns if 'hk_resident_type' in c and '_stu' in c][0]
        print(f"1. 受扰最严重的学生画像 (TOP 3):")
        stu_stats = matched[stu_res_col].value_counts(normalize=True).head(3)
        for res_type, ratio in stu_stats.items():
            print(f"{res_type: <20} {ratio:.2%}")

        print("\n2. 策略核心支撑点:")
        top_f = weights.iloc[0]['维度特征']
        print(f"- [路径识别]: 特征 '{top_f}' 表现出极高判别权重，应作为运营侧配置拦截策略的首选维度。")
        print(f"- [模式总结]: 关联数据展示了诈骗号对学生群体存在‘点对多’的重复触达，建议对高重复攻击号 (如触达数>80) 实施全网拦截。")
        print(f"- [防范重点]: 针对受扰最高的本地学生（>60%），建议联动 ADCC 开启针对性语音提示。")
        print("="*70)

    student_fraud_analysis_final()


def task_4():
    import pandas as pd
    import numpy as np

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    data_dir = PROJECT_DIR

    def _read_any(base_no_ext: str):
        xlsx = os.path.join(data_dir, f"{base_no_ext}.xlsx")
        csv = os.path.join(data_dir, f"{base_no_ext}.csv")
        if os.path.exists(xlsx):
            return pd.read_excel(xlsx)
        return pd.read_csv(csv)

    print("Loading data...")
    df_fraud = _read_any("fraud_model_2")
    df_n1 = _read_any("fraud_model_1_1")
    df_n2 = _read_any("fraud_model_1_2")
    df_normal = pd.concat([df_n1, df_n2], ignore_index=True)

    print("\n=======================================================")
    print("  RULES VALIDATION REPORT (Recall & Precision Check)")
    print("=======================================================")
    print(f"Total Fraud Samples: {len(df_fraud)}")
    print(f"Total Normal Samples: {len(df_normal)}")

    def rule_1_lite_batch(df):
        part_a = (df['rt_plan_desc'] == 'HKD33 MySIM Lite') & (df['iden_type_num'] > 3)
        if 'dispersion_rate' in df.columns:
            part_b = (df['rt_plan_desc'] == 'HKD33 MySIM Lite') & \
                     (df['dispersion_rate'] == 0) & \
                     (df['iden_type_num'] <= 3) & \
                     (df['call_cnt_day'] > 15) & \
                     (df['called_cnt_day'] <= 2) & \
                     (df['tot_msg_cnt'] == 0) & \
                     (df['avg_actv_dur'] < 60)
            return part_a | part_b
        return part_a

    def rule_2_lite_data(df):
        return (df['rt_plan_desc'] == 'HKD33 MySIM Lite') & \
               (df['ofr_nm'].astype(str).str.contains('60GB Local Data|LTE Basic DATA Pass', na=False))

    def rule_4_dormant_activation(df):
        return (df['rt_plan_desc'] == '$48 5G MySIM') & \
               (df['avg_actv_dur'] > 90) & \
               (df['ofr_nm'].astype(str).str.contains('Call Forward|Roaming CallBack', na=False)) & \
               (df['call_cnt_day'] < 30)

    def rule_5_birthday_abuse(df):
        return (df['rt_plan_desc'] == '$80 4G/3G data & voice Prepaid Card') & \
               (df['ofr_nm'].astype(str).str.contains('Birthday Bonus', na=False)) & \
               (df['avg_actv_dur'] < 30)

    def rule_6_oneway_roaming(df):
        base_logic = (df['ofr_nm'].astype(str).str.contains('Roaming CallBack', na=False)) & \
                     (df['called_cnt_day'] == 0) & \
                     (df['tot_msg_cnt'] == 0) & \
                     (df['call_cnt_day'] > 0) & \
                     (df['iden_type_num'] >= 5)
        if 'dispersion_rate' in df.columns:
            return base_logic & (df['dispersion_rate'] == 0)
        return base_logic

    def rule_7_5g_asymmetric(df):
        base_logic = (df['rt_plan_desc'] == '$48 5G MySIM') & \
                     (df['called_cnt_day'] == 0) & \
                     (df['tot_msg_cnt'] == 0) & \
                     (df['call_cnt_day'] > 0) & \
                     (df['iden_type_num'] >= 5)
        if 'dispersion_rate' in df.columns:
            return base_logic & (df['dispersion_rate'] == 0)
        return base_logic

    def rule_7_universal_dispersion(df):
        if 'dispersion_rate' not in df.columns:
            return pd.Series([False] * len(df), index=df.index)

        common_filter = (df['rt_plan_desc'] != '$48 5G MySIM') & \
                        (df['rt_plan_desc'] != 'HKD33 MySIM Lite')

        part_a = (df['call_cnt_day'] > 20) & \
                 (df['called_cnt_day'] <= 2) & \
                 (df['dispersion_rate'] == 0) & \
                 (df['iden_type_num'] >= 4)

        part_b = (df['call_cnt_day'] > 45) & \
                 (df['called_cnt_day'] <= 2) & \
                 (df['dispersion_rate'] == 0) & \
                 (df['iden_type_num'] >= 2) & \
                 (df['iden_type_num'] < 4)

        super_talk_fp = (df['rt_plan_desc'] == '$48 4G/3G Super Talk') & (df['avg_actv_dur'] > 60)
        return (part_a | part_b) & common_filter & (~super_talk_fp)

    rules = {
        "Rule 1 (Lite Batch)": rule_1_lite_batch,
        "Rule 2 (Lite Data)": rule_2_lite_data,
        "Rule 3 (Dormant Activation)": rule_4_dormant_activation,
        "Rule 4 (Birthday Abuse)": rule_5_birthday_abuse,
        "Rule 5 (Roaming Anomaly)": rule_6_oneway_roaming,
        "Rule 6 (5G Asymmetric)": rule_7_5g_asymmetric,
        "Rule 7 (Universal Dispersion)": rule_7_universal_dispersion
    }

    summary_data = []
    total_fraud_caught = set()
    total_normal_caught = set()

    for name, rule_func in rules.items():
        f_hits = df_fraud[rule_func(df_fraud)]
        n_hits = df_normal[rule_func(df_normal)]
        tp = len(f_hits)
        fp = len(n_hits)
        total_fraud_caught.update(f_hits.index.tolist())
        total_normal_caught.update(n_hits.index.tolist())
        recall = tp / len(df_fraud)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        summary_data.append({
            "Rule Name": name,
            "Caught Fraud (TP)": tp,
            "Caught Normal (FP)": fp,
            "Recall (Coverage)": f"{recall:.1%}",
            "Precision (Accuracy)": f"{precision:.1%}"
        })

    metrics_df = pd.DataFrame(summary_data)
    print("\n[Individual Rule Performance]")
    print(metrics_df.to_string(index=False))

    print("\n[Combined Performance]")
    total_tp = len(total_fraud_caught)
    total_fp = len(total_normal_caught)
    total_recall = total_tp / len(df_fraud)
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

    print(f"Total Unique Fraud Caught: {total_tp} / {len(df_fraud)}")
    print(f"Total Unique Normal Caught (False Positives): {total_fp} / {len(df_normal)}")
    print(f"Overall Recall (Coverage): {total_recall:.1%}")
    print(f"Overall Precision (Accuracy): {total_precision:.1%}")

    print("\n[Optimization Suggestions]")
    for _, row in metrics_df.iterrows():
        prec = float(row['Precision (Accuracy)'].strip('%'))
        if prec < 80.0:
            print(f"- {row['Rule Name']} has low precision ({prec:.1f}%). Consider tightening conditions.")
        elif prec > 95.0:
            print(f"- {row['Rule Name']} is excellent ({prec:.1f}%). Good for blocking.")


def task_5():
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    def task_5_black_sample_analysis():
        print("="*80)
        print("【任务 5：黑样本识别规则与自动化采集方案 - 修复版】")
        print("="*80)

        try:
            df_student = pd.read_excel('new_student_model.xlsx')
            df_normal_1 = pd.read_excel('fraud_model_1_1.xlsx')
            df_normal_2 = pd.read_excel('fraud_model_1_2.xlsx')
            df_fraud = pd.read_excel('fraud_model_2.xlsx')
            df_normal = pd.concat([df_normal_1, df_normal_2], ignore_index=True)
            print("✅ 数据读取成功。")
        except Exception as e:
            print(f"❌ 文件读取失败: {e}。请确保文件在当前目录下。")
            return

        def get_col(df, possible_names):
            for name in possible_names:
                if name in df.columns:
                    return name
            return None

        opp_col = get_col(df_fraud, ['opp_num_cnt', 'opp_num_stu_cnt'])
        call_col = get_col(df_fraud, ['call_cnt_day', 'call_cnt_day_2s'])

        if not opp_col or not call_col:
            print(f"❌ 错误：无法在数据中定位核心特征列。")
            print(f"当前存在的列名为: {list(df_fraud.columns)[:10]} ...")
            return
        else:
            print(f"ℹ️ 字段匹配成功：对端数使用 '{opp_col}'，通话数使用 '{call_col}'")

        for d in [df_normal, df_fraud]:
            d['dispersion'] = d[opp_col] / d[call_col].replace(0, 1)
            d['dispersion'] = d['dispersion'].clip(0, 1)

        avg_call_cnt = df_fraud[call_col].mean()
        avg_dispersion = df_fraud['dispersion'].mean()
        avg_duration = df_fraud['avg_actv_dur'].mean() if 'avg_actv_dur' in df_fraud.columns else 15
        avg_iden_num = df_fraud['iden_type_num'].mean() if 'iden_type_num' in df_fraud.columns else 5

        print("\n[一、 基于数据洞察的识别规则汇总]")
        print("-" * 50)

        x_calls_hour = max(10, int(avg_call_cnt / 8))
        print(f"规则 1 (高频外呼): 1小时内拨打电话次数不少于 {x_calls_hour} 次")
        print(f"规则 2 (社交离散): 号码离散度大于 {max(0.7, round(avg_dispersion, 2))}")
        print(f"规则 3 (短时通信): 平均通话时长低于 {max(12, int(avg_duration))} 秒")
        print(f"规则 4 (证照异常): 同一证照关联号码数量大于 {max(5, int(avg_iden_num))} 个")
        print(f"规则 5 (高危产品): 订购产品包含 'MySIM Lite' 或 '$48 5G MySIM'")

        print("\n[二、 黑样本识别与扩展方案说明]")
        print("-" * 50)
        print("1. 识别机制：采用“业务规则+行为极值”双重判定。")
        print("2. 隐私合规：进入库前对 MSISDN 进行 SHA-256 加盐脱敏。")
        print("3. 扩展燃料：")
        print(f"   - 自动扫描存量数据，将满足上述规则的样本自动打标。")
        print(f"   - 本次扫描从好人库中发现候选黑样本: {len(df_normal[df_normal[call_col] > avg_call_cnt])} 条。")
        print(f"   - 结合任务 3：已将命中学生受扰名单的诈骗号自动转为高优先级训练样本。")

        print("\n" + "="*80)
        print("【任务 5 分析完成】")
        print("="*80)

    task_5_black_sample_analysis()


# =========================================================
# 6) LLM Agent（保留你原 prompt；增加更稳的 https 调用：certifi + retry）
# =========================================================
def _truncate(s: str, max_chars: int) -> str:
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[TRUNCATED]...\n"


def build_antifraud_prompt(results: List[TaskResult]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"### {r.name}\n"
            f"- status: {'OK' if r.ok else 'FAILED'}\n"
            f"- stdout:\n{_truncate(r.stdout.strip(), 6000)}\n\n"
            f"- stderr:\n{_truncate(r.stderr.strip(), 2000)}\n\n"
            f"- exception:\n{_truncate((r.exception or '').strip(), 2000)}\n"
        )

    joined = "\n\n".join(blocks)

    return f"""
假设你是一名资深反诈专家与风控策略负责人。
你目前得到了以下 5 个 task 的输出信息（包含 EDA、模型打标、诈骗号关联分析、规则召回精度评估、黑样本规则与自动化采集方案）。

请你完成以下工作：
1) 用“要点列表”提炼这些输出里最重要的证据（数据结论、关键特征、命中画像、风险模式、规则表现、误杀/漏放线索）。
2) 给出可落地的“策略防控建议”，至少覆盖：
   - 拦截/处置策略（例如规则组合、阈值、分层处置、限呼/限拨/二次验证）
   - 运营触达策略（例如对学生群体提示、分人群提示、分渠道提示）
   - 训练数据闭环（黑样本采集、去噪、标注、迭代节奏）
   - 指标监控与A/B（Recall/Precision、误杀、触达、申诉、漂移监控）
3) 明确指出：如果任务之间结论冲突或数据字段缺失，应如何补采/核验。

输出要求：
- 用中文回答
- 分章节：『证据摘要』『策略建议』『风险与缺口』『下一步行动清单』
- 尽量具体，给阈值/优先级/执行顺序
- 不要复述大段原始日志

以下是 5 个任务的原始输出（已做长度截断，不影响关键信息）：
{joined}
""".strip()


def call_openai_responses_stdlib(api_key: str, model: str, prompt: str) -> str:
    """
    纯标准库 urllib 调用 OpenAI Responses API（增强版）：
    - certifi CA（若安装）
    - 重试与退避
    - 代理环境变量兼容
    """
    import urllib.request
    import urllib.error
    import ssl
    import time

    url = "https://api.openai.com/v1/responses"
    payload = json.dumps({"model": model, "input": prompt}).encode("utf-8")

    try:
        import certifi  # pip install certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()

    proxy_handler = urllib.request.ProxyHandler()
    https_handler = urllib.request.HTTPSHandler(context=ctx)
    opener = urllib.request.build_opener(proxy_handler, https_handler)

    req = urllib.request.Request(
        url=url,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    last_err = None
    for attempt in range(1, 4):
        try:
            with opener.open(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw)

            if isinstance(data, dict) and data.get("output_text"):
                return str(data["output_text"]).strip()

            texts = []
            if isinstance(data, dict):
                for item in data.get("output", []) or []:
                    if isinstance(item, dict):
                        for c in item.get("content", []) or []:
                            if isinstance(c, dict) and c.get("type") in ("output_text", "text") and c.get("text"):
                                texts.append(c["text"])
            if texts:
                return "\n".join(texts).strip()

            return json.dumps(data, ensure_ascii=False, indent=2)[:12000]

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return f"[OpenAI API ERROR] status={e.code}\n{body}"
        except Exception as e:
            last_err = e
            time.sleep(2 ** (attempt - 1))

    return "[FATAL] OpenAI 调用失败（重试后仍失败）：\n" + "".join(
        traceback.format_exception(type(last_err), last_err, last_err.__traceback__)
    )


def llm_agent_analyze(results: List[TaskResult], api_key: str, model: str) -> str:
    prompt = build_antifraud_prompt(results)
    return call_openai_responses_stdlib(api_key=api_key, model=model, prompt=prompt)


# =========================================================
# 8) 给 UI 调用的入口：run_all()
# =========================================================
def run_all(
    project_dir: str = PROJECT_DIR,
    api_key: str = "",
    model: str = "",
    show_console: bool = False,
) -> Tuple[List[TaskResult], str]:
    """
    给 Streamlit UI 调用：
    - 返回每个 task 的 TaskResult（stdout/stderr/exception）
    - 返回 LLM 输出（没有 key 则为空字符串）
    """
    if not os.path.isdir(project_dir):
        raise FileNotFoundError(f"PROJECT_DIR 不存在：{project_dir}")

    cwd_old = os.getcwd()
    os.chdir(project_dir)

    tasks: List[tuple[str, Callable[[], None]]] = [
        ("Task 1 - student_model EDA + weighted scoring", task_1),
        ("Task 2 - XGB train + label (final_submit_xgb.xlsx)", task_2),
        ("Task 3 - student_fraud_analysis_final (RF weights + linkage)", task_3),
        ("Task 4 - rules validation report (recall & precision)", task_4),
        ("Task 5 - black sample rules & automation plan", task_5),
    ]

    results: List[TaskResult] = []
    try:
        for name, fn in tasks:
            res = run_task(name, fn)
            results.append(res)
            if show_console:
                print_task_result(res)

        llm_text = ""
        api_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        model = (model or os.getenv("OPENAI_MODEL", "") or DEFAULT_OPENAI_MODEL).strip()

        if api_key:
            llm_text = llm_agent_analyze(results, api_key=api_key, model=model)

        return results, llm_text

    finally:
        os.chdir(cwd_old)


# =========================================================
# 7) main：终端兼容（不影响 UI）
# =========================================================
def main():
    _section("MERGED SCRIPT START")

    if not os.path.isdir(PROJECT_DIR):
        print(f"[FATAL] PROJECT_DIR 不存在：{PROJECT_DIR}")
        return

    os.chdir(PROJECT_DIR)
    print(f"WORKDIR = {os.getcwd()}")

    results, llm_text = run_all(PROJECT_DIR, api_key=os.getenv("OPENAI_API_KEY", ""), model=os.getenv("OPENAI_MODEL", ""), show_console=True)

    _section("LLM Agent (OpenAI) / Anti-fraud Strategy Output")
    if llm_text:
        print("\n--- LLM RESPONSE START ---\n")
        print(llm_text)
        print("\n--- LLM RESPONSE END ---\n")
    else:
        print("未检测到 OPENAI_API_KEY，跳过 LLM 分析。")

    _section("MERGED SCRIPT END")


if __name__ == "__main__":
    main()