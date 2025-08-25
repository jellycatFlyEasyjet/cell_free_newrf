#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速运行权重初始化测试套件
"""

import subprocess
import sys
import os

def run_test_suite():
    """运行所有权重初始化测试"""
    print("="*80)
    print("🧪 运行权重初始化测试套件")
    print("="*80)
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    tests = [
        ("test_updated_model.py", "测试更新后的主模型"),
        ("test_weight_init.py", "权重初始化方法对比测试"),
        ("visualize_init_results.py", "生成可视化分析图表")
    ]
    
    for test_file, description in tests:
        print(f"\n🔧 {description}...")
        print("-" * 60)
        
        try:
            result = subprocess.run([sys.executable, test_file], 
                                  cwd=test_dir, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=120)
            
            if result.returncode == 0:
                print(f"✅ {test_file} 运行成功")
                if result.stdout:
                    # 只显示关键输出行
                    lines = result.stdout.split('\n')
                    key_lines = [line for line in lines if any(key in line for key in 
                                ['✅', '🏆', '📊', '💡', '最终损失', '收敛率', '综合评分'])]
                    if key_lines:
                        print("  关键结果:")
                        for line in key_lines[:5]:  # 只显示前5行关键结果
                            print(f"    {line}")
            else:
                print(f"❌ {test_file} 运行失败")
                if result.stderr:
                    print(f"  错误: {result.stderr[:200]}...")
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file} 运行超时 (>120秒)")
        except Exception as e:
            print(f"💥 {test_file} 运行异常: {e}")
    
    print("\n" + "="*80)
    print("🎯 测试套件运行完成")
    print("="*80)
    print("📁 所有测试文件已整理到 weight_init_tests/ 文件夹")
    print("📊 可视化结果已保存为 PNG 图片")
    print("📖 详细说明请查看 README.md")

if __name__ == "__main__":
    run_test_suite()
