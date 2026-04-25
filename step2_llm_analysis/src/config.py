"""Path constants for the Step 2 LLM annotation pipeline.

All step2-specific data lives under step2_llm_analysis/data/.
Chunks (produced by Step 1) remain in the shared project data/ folder.
"""

from pathlib import Path

STEP2_DIR = Path(__file__).resolve().parents[1]   # step2_llm_analysis/
PROJECT_ROOT = STEP2_DIR.parent

# Step2-owned data
DATA_DIR = STEP2_DIR / "data"
LLM_ANNOTATIONS_DIR = DATA_DIR / "llm_annotations"
FEATURES_DIR = DATA_DIR / "features"

# Chunks are produced by Step 1 and live in step1's data folder
CHUNKS_DIR = PROJECT_ROOT / "step1_data_collection" / "data" / "chunks"
