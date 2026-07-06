"""Driver: apply fabric's VLMSpec injections + tokenizer fix, then run coreai.vlm.export's main().
Usage: python models/vlm/run_export.py <short-name> --output-dir <dir> --overwrite [--max-context-length N]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import models.vlm.register_specs  # noqa: F401,E402 — self-registers specs + patches on import
from coreai_models.vlm.export import main  # noqa: E402

if __name__ == "__main__":
    main()
