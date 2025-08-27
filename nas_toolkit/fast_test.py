#!/usr/bin/env python3
"""
Fast NAS Test - Quick functionality validation
==============================================
Runs a minimal NAS search to verify everything works correctly.
Estimated time: 3-5 minutes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_fast_nas import test_fast_nas

if __name__ == "__main__":
    print("🚀 Starting Fast NAS Test...")
    print("This is a quick validation test (3-5 minutes)")
    print("=" * 50)
    test_fast_nas()
