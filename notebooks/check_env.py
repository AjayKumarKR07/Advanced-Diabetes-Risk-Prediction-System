"""Environment checker for Phase 4."""
import sys
print(f"Python: {sys.version}")

packages = {
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'imbalanced-learn': 'imblearn',
    'scikit-learn': 'sklearn',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'joblib': 'joblib',
    'PIL': 'PIL',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
}

for display, mod in packages.items():
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', 'unknown')
        print(f"  ✅ {display}: {ver}")
    except ImportError:
        print(f"  ❌ {display}: NOT AVAILABLE")

