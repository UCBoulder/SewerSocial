#!/bin/bash
# ============================================================
# PharmShed Pipeline - Run All Models + Ensemble
# Ubuntu machine runner script
#
# Usage:
#   chmod +x run_script.sh
#   ./run_script.sh
#
#   To run in background (keeps running after terminal closes):
#   nohup ./run_script.sh > logs/pipeline.log 2>&1 &
#
# Logs: each model writes its own log file in logs/
# Order: XGBoost -> RealMLP -> KNN -> SVM -> TabICL -> Threshold -> Ensemble
# ============================================================

set -e  # Exit immediately if any command fails

# -- Configuration --------------------------------------------
NOTEBOOK_DIR="."              # All notebooks are in the same folder as this script
LOG_DIR="./logs"
PYTHON="python3"              # Change to full path if needed e.g. /usr/bin/python3

# -- Setup ----------------------------------------------------
mkdir -p "$LOG_DIR"

echo "============================================================"
echo " PharmShed Pipeline Starting"
echo " $(date)"
echo " Notebooks : $NOTEBOOK_DIR"
echo " Logs      : $LOG_DIR"
echo "============================================================"

# -- Helper function for notebooks ----------------------------
run_notebook() {
    local name=$1
    local notebook=$2
    local logfile="$LOG_DIR/${name}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo " Starting  : $name"
    echo " Time      : $(date)"
    echo " Log       : $logfile"
    echo "------------------------------------------------------------"

    jupyter nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=python3 \
        --inplace \
        "$NOTEBOOK_DIR/$notebook" \
        > "$logfile" 2>&1

    if [ $? -eq 0 ]; then
        echo " COMPLETE  : $name - $(date)"
    else
        echo " FAILED    : $name - check $logfile"
        echo "------------------------------------------------------------"
        echo " Last 20 lines of log:"
        tail -20 "$logfile"
        echo "------------------------------------------------------------"
        exit 1
    fi
}

# -- Helper function for Python scripts -----------------------
run_python() {
    local name=$1
    local script=$2
    local logfile="$LOG_DIR/${name}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo " Starting  : $name"
    echo " Time      : $(date)"
    echo " Log       : $logfile"
    echo "------------------------------------------------------------"

    cd "$NOTEBOOK_DIR"
    $PYTHON "$script" > "$logfile" 2>&1
    local exit_code=$?
    cd - > /dev/null

    if [ $exit_code -eq 0 ]; then
        echo " COMPLETE  : $name - $(date)"
    else
        echo " FAILED    : $name - check $logfile"
        echo "------------------------------------------------------------"
        echo " Last 20 lines of log:"
        tail -20 "$logfile"
        echo "------------------------------------------------------------"
        exit 1
    fi
}

# -- Check dependencies ---------------------------------------
echo ""
echo "Checking dependencies..."

if ! command -v jupyter &> /dev/null; then
    echo "ERROR: jupyter not found. Install with: pip install jupyter nbconvert"
    exit 1
fi

if ! $PYTHON -c "import xgboost" &> /dev/null; then
    echo "ERROR: xgboost not found. Install with: pip install 'xgboost>=2.1.0'"
    exit 1
fi

if ! $PYTHON -c "import pytabkit" &> /dev/null; then
    echo "ERROR: pytabkit not found. Install with: pip install pytabkit"
    exit 1
fi

if ! $PYTHON -c "import sklearn" &> /dev/null; then
    echo "ERROR: scikit-learn not found. Install with: pip install scikit-learn"
    exit 1
fi

if ! $PYTHON -c "import permetrics" &> /dev/null; then
    echo "ERROR: permetrics not found. Install with: pip install permetrics"
    exit 1
fi

if ! $PYTHON -c "import tabicl" &> /dev/null; then
    echo "ERROR: tabicl not found. Install with: pip install tabicl"
    exit 1
fi

echo "All dependencies found."

# -- Check required data files --------------------------------
echo ""
echo "Checking required data files..."

REQUIRED_FILES=(
    "$NOTEBOOK_DIR/super_integrated_data.csv"
    "$NOTEBOOK_DIR/super_data_2022.csv"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  FOUND   : $f"
    else
        echo "  MISSING : $f"
        exit 1
    fi
done

echo "All required data files found."

# -- Run models -----------------------------------------------

# 1. XGBoost - must run first, saves model for threshold tuning
run_notebook "XGBoost" "xgboost_super_dataset.ipynb"

# 2. RealMLP - resource intensive, run second
run_notebook "RealMLP" "realmlp_super_dataset.ipynb"

# 3. KNN - slow with Hassanat distance, run third
run_notebook "KNN" "knn_super_dataset.ipynb"

# 4. SVM - slow on large data, run fourth
run_notebook "SVM" "svm_super_dataset.ipynb"

# 5. TabICL - Vanessa's model, Python script not notebook
run_python "TabICL" "tabicl_model.py"

# 6. XGBoost threshold tuning - depends on model saved in step 1
run_notebook "XGBoost_Threshold" "xgboost_super_threshold_tuning.ipynb"

# 7. Ensemble - depends on all proba files from steps 1-6
echo ""
echo "------------------------------------------------------------"
echo " Checking for all probability output files before ensemble..."
echo "------------------------------------------------------------"

PROBA_FILES=(
    "$NOTEBOOK_DIR/xgboost_super_proba_2022.csv"
    "$NOTEBOOK_DIR/realmlp_super_proba_2022.csv"
    "$NOTEBOOK_DIR/knn_super_proba_2022.csv"
    "$NOTEBOOK_DIR/svm_super_proba_2022.csv"
    "$NOTEBOOK_DIR/tabicl_super_proba_2022.csv"
)

ALL_PROBA_READY=true
for f in "${PROBA_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  FOUND   : $(basename $f)"
    else
        echo "  MISSING : $(basename $f)"
        ALL_PROBA_READY=false
    fi
done

if [ "$ALL_PROBA_READY" = false ]; then
    echo ""
    echo "ERROR: One or more probability files are missing. Ensemble cannot run."
    exit 1
fi

run_notebook "Ensemble" "ensemble_pharmshed.ipynb"

# -- Summary --------------------------------------------------
echo ""
echo "============================================================"
echo " PharmShed Pipeline COMPLETE"
echo " $(date)"
echo "============================================================"
echo ""
echo " Output files:"
echo "   $NOTEBOOK_DIR/xgboost_super_cv_results.csv"
echo "   $NOTEBOOK_DIR/xgboost_super_proba_2022.csv"
echo "   $NOTEBOOK_DIR/xgboost_super_threshold_summary.csv"
echo "   $NOTEBOOK_DIR/realmlp_super_cv_results.csv"
echo "   $NOTEBOOK_DIR/realmlp_super_proba_2022.csv"
echo "   $NOTEBOOK_DIR/knn_super_cv_results.csv"
echo "   $NOTEBOOK_DIR/knn_super_proba_2022.csv"
echo "   $NOTEBOOK_DIR/svm_super_cv_results.csv"
echo "   $NOTEBOOK_DIR/svm_super_proba_2022.csv"
echo "   $NOTEBOOK_DIR/tabicl_super_cv_results.csv"
echo "   $NOTEBOOK_DIR/tabicl_super_proba_2022.csv"
echo "   $NOTEBOOK_DIR/ensemble_validation_summary.csv"
echo "   $NOTEBOOK_DIR/ensemble_per_drug_recall.csv"
echo "   $NOTEBOOK_DIR/ensemble_vs_base_models.csv"
echo "   $NOTEBOOK_DIR/ensemble_per_drug_comparison.csv"
echo ""
echo " Logs saved to: $LOG_DIR/"
echo "============================================================"