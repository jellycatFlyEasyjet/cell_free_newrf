#!/usr/bin/env python3
"""
Intelligent NAS Test - Standard search with pruning
==================================================
Runs a comprehensive NAS search with intelligent pruning.
Estimated time: 15 minutes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_intelligent_nas import test_intelligent_nas

if __name__ == "__main__":
    print("🏗️ Starting Intelligent NAS Test...")
    print("This is a comprehensive search test (15 minutes)")
    print("=" * 50)
    test_intelligent_nas()
