"""Allow running as: python -m xgb

Usage:
    python -m xgb --model_name xgb_0 ...          # baseline
    python -m xgb tune --study_name tune_0 ...     # tuning (requires: pip install -e ".[tuning]")
"""
import sys

if len(sys.argv) > 1 and sys.argv[1] == "tune":
    sys.argv.pop(1)  # remove 'tune' from args before click parses
    from xgb.tuning import main
else:
    from xgb.run import main

main()
