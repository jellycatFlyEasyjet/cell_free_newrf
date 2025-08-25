#!/usr/bin/env python3
"""
检查AP间CFR数据的配对关系和维度问题
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter

def check_ap_pairing(dataset_path):
    """检查AP间的配对关系"""
    
    print("="*80)
    print("🔍 AP配对关系分析")
    print("="*80)
    
    dataset = pd.read_pickle(dataset_path)
    dataset = dataset.dropna()
    
    print(f"数据集大小: {len(dataset)} 条记录")
    print(f"总AP数: {len(dataset['TxID'].unique())}")
    print(f"总STA数: {len(dataset['RxID'].unique())}")
    
    # 分析AP-STA配对
    print(f"\n📊 AP-STA配对分析:")
    
    # 按STA分组，看每个STA与多少个AP有连接
    sta_ap_counts = defaultdict(list)
    for _, row in dataset.iterrows():
        sta_id = int(row['RxID'])
        ap_id = int(row['TxID'])
        sta_ap_counts[sta_id].append(ap_id)
    
    # 统计每个STA连接的AP数量
    ap_count_distribution = Counter([len(aps) for aps in sta_ap_counts.values()])
    print(f"  STA连接AP数量分布: {dict(ap_count_distribution)}")
    
    # 检查是否所有STA都连接到相同数量的AP
    ap_counts_per_sta = [len(aps) for aps in sta_ap_counts.values()]
    if len(set(ap_counts_per_sta)) == 1:
        print(f"  ✅ 所有STA都连接到 {ap_counts_per_sta[0]} 个AP")
    else:
        print(f"  ⚠️ STA连接的AP数量不一致: {set(ap_counts_per_sta)}")
    
    # 分析特定场景：base_AP=[1], target_AP=[2]
    print(f"\n🎯 特定配置分析 (base_AP=[1], target_AP=[2]):")
    
    base_ap = 1
    target_ap = 2
    
    # 检查有多少STA同时连接到AP1和AP2
    sta_with_ap1 = set(dataset[dataset['TxID'] == base_ap]['RxID'])
    sta_with_ap2 = set(dataset[dataset['TxID'] == target_ap]['RxID'])
    
    common_stas = sta_with_ap1 & sta_with_ap2
    
    print(f"  连接到AP{base_ap}的STA数量: {len(sta_with_ap1)}")
    print(f"  连接到AP{target_ap}的STA数量: {len(sta_with_ap2)}")
    print(f"  同时连接到AP{base_ap}和AP{target_ap}的STA数量: {len(common_stas)}")
    
    if len(common_stas) != len(sta_with_ap1) or len(common_stas) != len(sta_with_ap2):
        print(f"  ⚠️ 警告：不是所有STA都同时连接到这两个AP")
        only_ap1 = sta_with_ap1 - sta_with_ap2
        only_ap2 = sta_with_ap2 - sta_with_ap1
        if only_ap1:
            print(f"    只连接到AP{base_ap}的STA: {sorted(list(only_ap1))[:10]}...")
        if only_ap2:
            print(f"    只连接到AP{target_ap}的STA: {sorted(list(only_ap2))[:10]}...")
    else:
        print(f"  ✅ 所有STA都同时连接到AP{base_ap}和AP{target_ap}")
    
    # 模拟数据加载过程
    print(f"\n🧪 模拟数据加载过程:")
    
    def simulate_get_cfr_batch(ap_ids, sta_ids):
        """模拟get_cfr_batch函数"""
        cfr_list = []
        found_pairs = []
        
        for sta_id in sta_ids:
            for ap_id in ap_ids:
                pair = dataset[(dataset['TxID'] == ap_id) & (dataset['RxID'] == sta_id)]
                if not pair.empty:
                    cfr = pair['CSI'].iloc[0]
                    if isinstance(cfr, np.ndarray) and len(cfr) > 0:
                        cfr_list.append(cfr.flatten())
                        found_pairs.append((ap_id, sta_id))
                else:
                    print(f"    ⚠️ 未找到 AP{ap_id}-STA{sta_id} 的CFR数据")
        
        return np.array(cfr_list), found_pairs
    
    # 模拟batch_size=5的情况
    test_batch_size = 5
    available_stas = sorted(list(common_stas))[:test_batch_size]
    
    print(f"  测试batch: STAs {available_stas}")
    
    # 获取base_AP的CFR
    base_cfr, base_pairs = simulate_get_cfr_batch([base_ap], available_stas)
    print(f"  Base AP{base_ap} CFR形状: {base_cfr.shape}")
    print(f"  Base AP配对: {base_pairs}")
    
    # 获取target_AP的CFR
    target_cfr, target_pairs = simulate_get_cfr_batch([target_ap], available_stas)
    print(f"  Target AP{target_ap} CFR形状: {target_cfr.shape}")
    print(f"  Target AP配对: {target_pairs}")
    
    # 检查维度匹配
    if base_cfr.shape[0] == target_cfr.shape[0] == len(available_stas):
        print(f"  ✅ 维度匹配：{base_cfr.shape[0]} == {target_cfr.shape[0]} == {len(available_stas)}")
    else:
        print(f"  ❌ 维度不匹配：base_cfr={base_cfr.shape[0]}, target_cfr={target_cfr.shape[0]}, batch_size={len(available_stas)}")
    
    # 分析位置数据
    print(f"\n📍 位置数据分析:")
    
    def simulate_get_loc_batch(dev_type, sta_ids):
        """模拟get_loc_batch函数"""
        if dev_type == "STA":
            loc_list = []
            for sta_id in sta_ids:
                sta_data = dataset[dataset['RxID'] == sta_id]
                if not sta_data.empty:
                    loc = sta_data['RxPos'].iloc[0]
                    loc_list.append(loc)
                else:
                    print(f"    ⚠️ 未找到STA{sta_id}的位置数据")
            return np.array(loc_list)
        
        elif dev_type == "AP":
            # 这里模拟原始代码的逻辑
            if sta_ids:
                sta_data = dataset[dataset['RxID'] == sta_ids[0]]
                if not sta_data.empty:
                    # 获取与第一个STA相关的所有AP位置
                    ap_positions = []
                    for _, row in sta_data.iterrows():
                        ap_pos = row['TxPos']
                        ap_positions.append(ap_pos)
                    return np.array(ap_positions)
            return np.array([])
    
    sta_locs = simulate_get_loc_batch("STA", available_stas)
    ap_locs = simulate_get_loc_batch("AP", available_stas)
    
    print(f"  STA位置数据形状: {sta_locs.shape}")
    print(f"  AP位置数据形状: {ap_locs.shape}")
    
    # 检查最终维度匹配
    print(f"\n🔧 最终维度检查:")
    print(f"  STA位置: {sta_locs.shape}")
    print(f"  Base CFR: {base_cfr.shape}")
    print(f"  Target CFR: {target_cfr.shape}")
    
    if sta_locs.shape[0] == base_cfr.shape[0]:
        print(f"  ✅ STA位置与Base CFR维度匹配")
    else:
        print(f"  ❌ STA位置与Base CFR维度不匹配: {sta_locs.shape[0]} vs {base_cfr.shape[0]}")
        print(f"      这就是导致 RuntimeError 的原因！")
    
    # 给出解决建议
    print(f"\n💡 解决建议:")
    print(f"  1. 确保批次中的所有STA都有对应的AP CFR数据")
    print(f"  2. 检查get_loc_batch函数是否正确返回对应数量的位置")
    print(f"  3. 考虑在数据加载时进行维度验证")
    print(f"  4. 过滤掉不完整的STA-AP配对")

if __name__ == "__main__":
    dataset_path = '/home/byang/BoYang/NeWRF-main/simulator/datasets/conference_500STA_4APs.pkl'
    check_ap_pairing(dataset_path)
