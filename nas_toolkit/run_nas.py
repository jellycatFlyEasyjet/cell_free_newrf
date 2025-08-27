#!/usr/bin/env python3
"""
NAS Toolkit Launcher
===================
Interactive script to run different NAS tests
"""

import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def main():
    print("🚀 NAS Toolkit for NeWRF")
    print("=" * 40)
    print()
    print("Choose a test to run:")
    print("1. Quick Test (5 trials each)")
    print("   - Quick functionality verification")
    print("   - 5 architecture + 5 hyperparameter trials")
    print()
    print("2. Standard Test (10 + 15 trials)")
    print("   - Balanced search with good coverage")
    print("   - 10 architecture + 15 hyperparameter trials")
    print()
    print("3. Comprehensive Test (15 + 20 trials)")
    print("   - Thorough search for best results")
    print("   - 15 architecture + 20 hyperparameter trials")
    print()
    print("4. Custom Test")
    print("   - Specify your own parameters")
    print()
    print("5. View Toolkit Info")
    print("   - Display toolkit information")
    print()
    
    while True:
        choice = input("Enter your choice (1-5, or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            print("👋 Goodbye!")
            break
            
        elif choice == '1':
            print("\n🚀 Running Quick Test...")
            from optuna_nas import run_two_stage_nas
            run_two_stage_nas(5, 5)
            break
            
        elif choice == '2':
            print("\n🏗️ Running Standard Test...")
            from optuna_nas import run_two_stage_nas
            run_two_stage_nas(10, 15)
            break
            
        elif choice == '3':
            print("\n📊 Running Comprehensive Test...")
            from optuna_nas import run_two_stage_nas
            run_two_stage_nas(15, 20)
            break
            
        elif choice == '4':
            print("\n⚙️ Custom Test Configuration")
            try:
                arch_trials = int(input("Architecture trials (default 10): ") or "10")
                hp_trials = int(input("Hyperparameter trials (default 15): ") or "15")
                
                print(f"\n🔧 Running Custom Test: {arch_trials} arch + {hp_trials} hp trials")
                from optuna_nas import run_two_stage_nas
                run_two_stage_nas(arch_trials, hp_trials)
                break
                
            except ValueError:
                print("❌ Please enter valid numbers")
                continue
                
        elif choice == '5':
            from __init__ import get_info
            info = get_info()
            print(f"\n📋 NAS Toolkit Information")
            print(f"Version: {info['version']}")
            print(f"Author: {info['author']}")
            print(f"Description: {info['description']}")
            print("\nFeatures:")
            for feature in info['features']:
                print(f"  ✅ {feature}")
            print()
            
        else:
            print("❌ Invalid choice. Please enter 1-5 or 'q'.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check the requirements and try again.")
