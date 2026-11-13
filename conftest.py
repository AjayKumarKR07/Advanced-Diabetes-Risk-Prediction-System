"""
conftest.py
===========
Pytest configuration for Advanced Diabetes Risk Prediction System.
Adds src/ to sys.path so all test modules can import project modules
without manual sys.path manipulation.
"""
import sys
from pathlib import Path

# Project root is the directory containing this file
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

