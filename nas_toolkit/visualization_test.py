#!/usr/bin/env python3
"""
Visualization NAS Test - Complete test with full visualization
=============================================================
Runs NAS with complete visualization and analysis features.
Estimated time: 8-10 minutes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_nas_visualization import test_visualization_nas

if __name__ == "__main__":
    print("📊 Starting Visualization NAS Test...")
    print("This includes complete visualization features (8-10 minutes)")
    print("=" * 50)
    test_visualization_nas()
