#!/usr/bin/env python3
# coding: utf-8
"""
通用抓取 + 备用本地数据生成与预处理脚本（不依赖 sklearn）。
生成的数据更贴近真实市场数据资产特征，包含额外字段与相关性结构，适合用于实验与复现。
"""
import time
import random
import json
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from urllib.parse import urljoin, urlparse
import urllib.robotparser
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ========== 配置 ==========
BASE_URL = "https://youe.example.com/"  # 替换为真实数据源（示例）
LIST_PAGE_PATH = "/data-service?page={}"   # 列表页路径模板（示例）
MAX_PAGES = 100
TARGET_RECORDS = 3782
REQUESTS_PER_SECOND = 1.0
OUTPUT_DIR = "youe_dataset_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUALITY_METRICS = ["completeness", "accuracy", "timeliness", "consistency",
                   "availability", "scarcity", "compliance"]
TEXT_FIELD = "description"

# ========== 实用函数 ==========
def can_fetch_url(base_url: str, user_agent: str = "*") -> bool:
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch(user_agent, base_url)
        logging.info("robots.txt %s -> can_fetch=%s", robots_url, allowed)
        return allowed
    except Exception as e:
        logging.warning("robots.txt 检查失败: %s. 将使用备用数据源或本地数据", e)
        return False

def polite_get(url: str, session: requests.Session, max_retries: int = 3, timeout: int = 10):
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, timeout=timeout, headers={"User-Agent": "DataAssetBot/1.0"})
            r.raise_for_status()
            return r
        except Exception as e:
            wait = 1.5 ** attempt
            logging.warning("请求失败 %s (attempt %d/%d): %s. 等待 %.1f s 后重试", url, attempt, max_retries, e, wait)
            time.sleep(wait)
    logging.error("请求多次失败，放弃: %s", url)
    return None

# 解析模板（可根据目标页面改写）
def parse_list_page(html_text: str) -> List[str]:
    soup = BeautifulSoup(html_text, "html.parser")
    links = []
    for a in soup.select("a.asset-link"):
        href = a.get("href")
        if href:
            links.append(href)
    return links

def parse_detail_page(html_text: str) -> Optional[Dict]:
    soup = BeautifulSoup(html_text, "html.parser")
    try:
        asset_id_el = soup.select_one("span.asset-id")
        ts_el = soup.select_one("time.tx-ts")
        if not asset_id_el or not ts_el:
            return None
        asset_id = asset_id_el.get_text(strip=True)
        ts_text = ts_el.get("datetime") or ts_el.get_text(strip=True)
        try:
            transaction_timestamp = datetime.fromisoformat(ts_text).isoformat()
        except Exception:
            transaction_timestamp = datetime.now().isoformat()
        data_type_el = soup.select_one("span.data-type")
        provider_el = soup.select_one("a.provider-name")
        price_el = soup.select_one("span.historical-price")
        freq_el = soup.select_one("span.tx-frequency")
        desc_el = soup.select_one("div.description")

        data_type = data_type_el.get_text(strip=True) if data_type_el else "unknown"
        provider = provider_el.get_text(strip=True) if provider_el else "unknown"
        try:
            historical_price = float(price_el.get_text(strip=True).replace("$", "")) if price_el else 0.0
        except Exception:
            historical_price = 0.0
        try:
            transaction_frequency = int(freq_el.get_text(strip=True)) if freq_el else 0
        except Exception:
            transaction_frequency = 0
        quality = {}
        for m in QUALITY_METRICS:
            el = soup.select_one(f"span.q-{m}")
            if el:
                try:
                    quality[m] = float(el.get_text(strip=True))
                except Exception:
                    quality[m] = None
            else:
                quality[m] = None
        description = desc_el.get_text(strip=True) if desc_el else ""
        record = {
            "asset_id": asset_id,
            "transaction_timestamp": transaction_timestamp,
            "data_type": data_type,
            "provider": provider,
            "historical_price": historical_price,
            "transaction_frequency": transaction_frequency,
            **quality,
            "description": description
        }
        # 占位字段
        record.setdefault("license", None)
        record.setdefault("privacy_level", None)
        record.setdefault("data_size_mb", None)
        record.setdefault("num_fields", None)
        record.setdefault("update_frequency", None)
        return record
    except Exception as e:
        logging.debug("解析详情页失败: %s", e)
        return None

# ========== 备用本地数据生成（更高真实性） ==========
def generate_dataset(n: int = TARGET_RECORDS, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed + 1)
    rows = []

    # provider 列表与信誉分（更少热门 provider 占少数）
    providers = []
    provider_reputation = {}
    for i in range(1, 101):
        name = f"provider_{i:03d}"
        providers.append(name)
        # 模拟信誉分分布：少数高信誉，大多数中低
        rep = min(0.99, max(0.01, np.random.beta(2, 5) + (0.5 if i <= 10 else 0)))
        provider_reputation[name] = round(rep, 3)

    data_types = [
        ("transaction", {"price_loc": 2.0, "price_scale": 1.0, "size_mb_loc": 50, "size_scale": 30}),
        ("user_profile", {"price_loc": 1.0, "price_scale": 0.7, "size_mb_loc": 200, "size_scale": 150}),
        ("location", {"price_loc": 0.5, "price_scale": 0.3, "size_mb_loc": 20, "size_scale": 15}),
        ("sensor", {"price_loc": 0.8, "price_scale": 0.6, "size_mb_loc": 500, "size_scale": 400}),
        ("image", {"price_loc": 5.0, "price_scale": 3.0, "size_mb_loc": 2000, "size_scale": 1500}),
        ("text", {"price_loc": 0.7, "price_scale": 0.5, "size_mb_loc": 100, "size_scale": 80}),
        ("medical", {"price_loc": 8.0, "price_scale": 4.0, "size_mb_loc": 300, "size_scale": 200}),
    ]

    licenses = ["open", "cc-by", "commercial", "restricted"]
    privacy_levels = ["public", "anonymized", "pseudonymized", "sensitive"]

    start_time = datetime.now() - timedelta(days=365 * 2)  # 最近两年数据
    # 季节/周模式：高峰在工作日、白天
    for i in range(n):
        asset_id = f"A{100000 + i}"
        # 时间戳：密度随时间增长，加入季节性与日周期
        days_offset = int((i / n) * 365 * 2)
        hour = int(np.clip(np.random.normal(14, 4), 0, 23))  # 白天偏多
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        ts = start_time + timedelta(days=days_offset, hours=hour, minutes=minute, seconds=second)
        transaction_timestamp = ts.isoformat()

        # provider 按Zipf-like分布（热门 provider 被更多使用）
        p_idx = np.random.zipf(1.5)
        p_idx = (p_idx % len(providers))
        provider = providers[p_idx]
        rep = provider_reputation[provider]

        # data_type 按权重选择
        dt_name, dt_meta = random.choices(data_types, weights=[0.20,0.18,0.12,0.15,0.08,0.20,0.07], k=1)[0]

        # 价格依据 data_type 与 provider reputation，有长尾分布（往上有高价）
        base_price = max(0.01, np.random.lognormal(mean=math.log(max(0.1, dt_meta["price_loc"])), sigma=dt_meta["price_scale"]/dt_meta["price_loc"]))
        # reputation 提升质量同时略微提升价格
        historical_price = round(base_price * (1.0 + rep * 0.6) * (1 + np.random.normal(0, 0.1)), 4)

        # data_size_mb 与 data_type 相关且带噪声
        data_size_mb = max(0.1, abs(np.random.normal(dt_meta["size_mb_loc"], dt_meta["size_scale"])))
        num_fields = max(1, int(np.round(np.random.poisson(lam=10 if dt_name in ["transaction","user_profile"] else 5))))

        # 更新频率（每天、每周、每月）
        update_candidates = ["real-time", "hourly", "daily", "weekly", "monthly"]
        update_probs = [0.05, 0.10, 0.35, 0.3, 0.2]
        update_frequency = random.choices(update_candidates, weights=update_probs, k=1)[0]

        # 交易频率与更新时间、data_type 相关
        tx_freq_base = {"real-time":50, "hourly":10, "daily":3, "weekly":1, "monthly":0.2}[update_frequency]
        transaction_frequency = max(1, int(np.round(np.random.poisson(lam=tx_freq_base))))

        # license 与 privacy_level 基于 data_type 与随机性
        license = random.choices(licenses, weights=[0.25,0.25,0.35,0.15], k=1)[0]
        privacy_level = random.choices(privacy_levels, weights=[0.15,0.5,0.25,0.1], k=1)[0]

        # 质量分：受 provider reputation、data_type 与一些相关性影响
        quality = {}
        # 构造一个基础质量分（0-1）受 rep 与 data_type 各自权重影响
        dt_quality_bias = {
            "transaction": 0.05, "user_profile": 0.00, "location": -0.02,
            "sensor": -0.05, "image": 0.02, "text": -0.01, "medical": 0.10
        }.get(dt_name, 0.0)
        base_quality = np.clip(rep + dt_quality_bias + np.random.normal(0, 0.08), 0.0, 1.0)
        # 各指标相关但有独立噪声
        for m in QUALITY_METRICS:
            # 不同 metric 有不同偏好与方差
            mv = base_quality + np.random.normal(0, 0.12)
            if m == "scarcity":
                # scarcity 与价格正相关
                mv = np.clip(0.2 + (historical_price / (historical_price + 1.0)) * 0.8 + np.random.normal(0, 0.1), 0, 1)
            if m == "timeliness":
                mv = mv + (0.1 if update_frequency in ["real-time","hourly"] else -0.05)
            quality[m] = round(float(np.clip(mv, 0.0, 1.0)), 4)

        # 描述模板，更自然，包含关键词
        desc_templates = {
            "transaction": [
                "包含用户支付与订单流水，适用于风控、信贷场景，字段包括金额、时间戳、商家ID等。来源: {}。",
                "交易日志序列，覆盖近三年历史，适合训练欺诈检测与用户画像。提供商：{}。"
            ],
            "user_profile": [
                "用户画像数据集，包含人口统计信息、兴趣标签、历史行为特征。来源: {}。",
                "聚合的用户属性与标签，用于推荐与个性化服务。提供者：{}。"
            ],
            "location": [
                "位置轨迹数据，经过匿名化处理，常用于出行分析与位置推荐。来源: {}。",
                "位置信息记录（经化名处理），包含经纬度、时间序列。提供者：{}。"
            ],
            "sensor": [
                "工业传感器采样数据，包含温度、振动、电流等时序信号。来源: {}。",
                "IoT 设备周期采样，适合故障预测与能耗优化。提供者：{}。"
            ],
            "image": [
                "图像数据集（已标注/未标注），可用于目标检测与分类任务。来源: {}。",
                "包含多类别标注的图片集合，适用于视觉模型训练。提供者：{}。"
            ],
            "text": [
                "文本语料与对话日志，已清理与分词处理，适合 NLP 任务。来源: {}。",
                "文章与评论集合，包含情感标签或主题标注。提供者：{}。"
            ],
            "medical": [
                "医疗检测与电子病历（已去标识化），用于临床研究与诊断模型。来源: {}（合规审查通过）。",
                "包含检查结果、诊断编码与匿名化病史，合规可用于科研。提供者：{}。"
            ]
        }
        tmpl = random.choice(desc_templates.get(dt_name, ["通用数据集，适用多种任务。来源: {}。"]))
        description = tmpl.format(provider)

        # 引入一些缺失值模式：医疗与某些大提供商数据可能缺少部分质量值或字段
        missing_prob = 0.01 if rep > 0.8 else 0.05
        if dt_name == "medical":
            missing_prob = 0.07
        # 随机将部分 quality 设为 None
        for m in list(quality.keys()):
            if random.random() < missing_prob:
                quality[m] = None

        row = {
            "asset_id": asset_id,
            "transaction_timestamp": transaction_timestamp,
            "data_type": dt_name,
            "provider": provider,
            "historical_price": historical_price,
            "transaction_frequency": transaction_frequency,
            "data_size_mb": round(float(data_size_mb), 2),
            "num_fields": int(num_fields),
            "update_frequency": update_frequency,
            "license": license,
            "privacy_level": privacy_level,
            **quality,
            "description": description
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # 对一些列引入少量异常值 / 手工噪声，模拟真实脏数据
    if "historical_price" in df.columns:
        idxs = df.sample(frac=0.002, random_state=seed).index
        df.loc[idxs, "historical_price"] = df.loc[idxs, "historical_price"] * np.random.uniform(5, 50, size=len(idxs))
    # 随机丢失少量 provider 值
    df.loc[df.sample(frac=0.003, random_state=seed+2).index, "provider"] = None

    return df

# ========== 文本场景映射（规则） ==========
def map_scenario_by_text(text: str) -> str:
    txt = (text or "").lower()
    if any(k in txt for k in ["信用", "欺诈", "信贷", "风控", "fraud", "credit"]):
        return "financial_risk_control"
    if any(k in txt for k in ["推荐", "推荐", "商品", "电商", "recommend", "购买"]):
        return "ecommerce_recommendation"
    if any(k in txt for k in ["患者", "诊断", "医疗", "病历", "clinical", "diagnosis", "medical"]):
        return "medical_diagnosis"
    return "other"

# ========== 预处理（手工 Min-Max） ==========
def preprocess_dataframe(df: pd.DataFrame, save_prefix: str = OUTPUT_DIR):
    df = df.copy()
    df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"])
    # 类别编码
    categorical_cols = ["data_type", "provider", "license", "privacy_level", "update_frequency"]
    cat_mappings = {}
    for col in categorical_cols:
        uniques = sorted(df[col].fillna("unknown").astype(str).unique())
        mapping = {v: i for i, v in enumerate(uniques)}
        cat_mappings[col] = mapping
        df[f"{col}_idx"] = df[col].fillna("unknown").astype(str).map(mapping).astype(int)
    with open(os.path.join(save_prefix, "cat_mappings.json"), "w", encoding="utf8") as f:
        json.dump(cat_mappings, f, ensure_ascii=False, indent=2)
    logging.info("类别映射已保存：%s", {k: len(v) for k, v in cat_mappings.items()})

    # 数值列归一化（手工 Min-Max）
    numeric_cols = ["historical_price", "transaction_frequency", "data_size_mb", "num_fields"] + QUALITY_METRICS
    # 确保存在这些列
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
    df_numeric = df[numeric_cols].fillna(0).astype(float)
    mins = df_numeric.min(axis=0)
    maxs = df_numeric.max(axis=0)
    scaler_meta = {"feature_names": numeric_cols, "min": [], "max": []}
    for col in numeric_cols:
        col_min = float(mins[col])
        col_max = float(maxs[col])
        scaler_meta["min"].append(col_min)
        scaler_meta["max"].append(col_max)
        if col_max == col_min:
            df[f"{col}_norm"] = 0.0
        else:
            df[f"{col}_norm"] = (df[col].astype(float) - col_min) / (col_max - col_min)
    with open(os.path.join(save_prefix, "scaler_meta.json"), "w", encoding="utf8") as f:
        json.dump(scaler_meta, f, ensure_ascii=False, indent=2)
    logging.info("数值归一化完成，元数据已保存。")

    # 场景映射
    df["mapped_scenario"] = df[TEXT_FIELD].apply(map_scenario_by_text)
    df.loc[df["mapped_scenario"] == "other", "manual_annotation_needed"] = True
    df["manual_annotation_needed"] = df["manual_annotation_needed"].fillna(False)

    # 按时间排序并划分（时间连贯）
    df = df.sort_values("transaction_timestamp").reset_index(drop=True)
    n = len(df)
    n_train = int(math.floor(0.70 * n))
    n_val = int(math.floor(0.15 * n))
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    logging.info("分割完成：total=%d, train=%d, val=%d, test=%d", n, len(train_df), len(val_df), len(test_df))

    # 保存
    df.to_csv(os.path.join(save_prefix, "raw_processed_all.csv"), index=False, encoding="utf-8-sig")
    train_df.to_csv(os.path.join(save_prefix, "train.csv"), index=False, encoding="utf-8-sig")
    val_df.to_csv(os.path.join(save_prefix, "val.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(save_prefix, "test.csv"), index=False, encoding="utf-8-sig")
    logging.info("文件已保存目录：%s", save_prefix)

    return {"all": df, "train": train_df, "val": val_df, "test": test_df, "mappings": cat_mappings, "scaler_meta": scaler_meta}

# ========== 主流程 ==========
def crawl_dataset(base_url: str = BASE_URL, max_pages: int = MAX_PAGES, target_records: int = TARGET_RECORDS):
    session = requests.Session()
    if not can_fetch_url(base_url):
        logging.info("使用本地备用数据集以继续处理流程。")
        return generate_dataset(n=target_records)

    records = []
    seen_asset_ids = set()
    page = 1
    last_request_time = 0.0
    while page <= max_pages and len(records) < target_records:
        list_url = urljoin(base_url, LIST_PAGE_PATH.format(page))
        wait = max(0, (1.0 / REQUESTS_PER_SECOND) - (time.time() - last_request_time))
        if wait > 0:
            time.sleep(wait)
        last_request_time = time.time()
        r = polite_get(list_url, session)
        if r is None:
            logging.warning("列表页请求失败 %s", list_url)
            break
        detail_paths = parse_list_page(r.text)
        if not detail_paths:
            logging.info("未找到详情链接，停止爬取。")
            break
        for p in detail_paths:
            if len(records) >= target_records:
                break
            detail_url = urljoin(base_url, p)
            wait = max(0, (1.0 / REQUESTS_PER_SECOND) - (time.time() - last_request_time))
            if wait > 0:
                time.sleep(wait)
            last_request_time = time.time()
            rd = polite_get(detail_url, session)
            if rd is None:
                continue
            rec = parse_detail_page(rd.text)
            if rec is None:
                continue
            if rec.get("asset_id") in seen_asset_ids:
                continue
            seen_asset_ids.add(rec.get("asset_id"))
            records.append(rec)
        logging.info("已抓取 %d 条记录，页面 %d 完成", len(records), page)
        page += 1

    if len(records) == 0:
        logging.info("未抓取到数据，使用本地备用数据集。")
        return generate_dataset(n=target_records)
    df = pd.DataFrame(records)
    if len(df) < target_records:
        deficit = target_records - len(df)
        logging.info("抓取数量不足，将补齐 %d 条备用数据。", deficit)
        supplement = generate_dataset(n=deficit, seed=random.randint(0, 100000))
        df = pd.concat([df, supplement], ignore_index=True)
    return df

if __name__ == "__main__":
    logging.info("开始数据获取流程。")
    raw_df = None
    try:
        raw_df = crawl_dataset()
        raw_df.to_csv(os.path.join(OUTPUT_DIR, "youe_raw.csv"), index=False, encoding="utf-8-sig")
        logging.info("获取到 %d 条原始记录。", len(raw_df))
    except Exception as e:
        logging.exception("数据获取流程出错，使用本地备用数据集继续：%s", e)
        raw_df = generate_dataset(n=TARGET_RECORDS)

    # 预处理与划分
    res = preprocess_dataframe(raw_df, save_prefix=OUTPUT_DIR)
    logging.info("处理完成，输出位于：%s", OUTPUT_DIR)
