import py_compile
import sys
import os

os.chdir(r'C:\Users\ACER\Desktop\cv_analysis\cv-matching-data-mining')

files = [
    "main.py",
    "src/models/learned_fusion.py",
    "src/models/cross_encoder_rerank.py",
    "src/extraction/skill_extractor.py",
    "api/main.py",
    "api/routers/cv.py",
    "api/routers/job.py",
    "api/routers/ranking.py",
    "tests/test_api_routers.py"
]

has_errors = False

for file in files:
    print(f"Checking: {file}")
    try:
        py_compile.compile(file, doraise=True)
        print(f"  ✓ OK")
    except py_compile.PyCompileError as e:
        print(f"  ✗ ERROR:")
        print(f"    {e}")
        has_errors = True

if not has_errors:
    print("\nAll OK")
else:
    print("\nSome files have syntax errors")
    sys.exit(1)
