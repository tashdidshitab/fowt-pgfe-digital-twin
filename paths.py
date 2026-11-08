"""Project locations. Override with environment variables:
  FOWT_ROOT       repository root (default: the folder containing this file)
  OPENFAST_RTEST  clone of https://github.com/OpenFAST/r-test at tag v5.0.0 (default: <root>/r-test)
"""
import os
ROOT = os.environ.get('FOWT_ROOT', os.path.dirname(os.path.abspath(__file__)))
RTEST = os.environ.get('OPENFAST_RTEST', os.path.join(ROOT, 'r-test'))
