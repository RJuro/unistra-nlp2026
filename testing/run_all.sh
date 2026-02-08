#!/bin/bash
# Run all NB06-NB11 tests
# Usage: ./run_all.sh [nb06|nb07|nb08|nb09|nb10|nb11]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin/python"

if [ ! -f "$VENV" ]; then
    echo "ERROR: venv not found at $VENV"
    echo "Run: python3 -m venv $SCRIPT_DIR/venv && $SCRIPT_DIR/venv/bin/pip install ..."
    exit 1
fi

run_test() {
    local name="$1"
    local script="$2"
    echo ""
    echo "=========================================="
    echo "Running $name..."
    echo "=========================================="
    "$VENV" "$SCRIPT_DIR/$script" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo ">>> $name: PASSED"
    else
        echo ">>> $name: FAILED (exit code $status)"
    fi
    return $status
}

if [ -n "$1" ]; then
    case "$1" in
        nb06) run_test "NB06" "test_nb06_faiss.py" ;;
        nb07) run_test "NB07" "test_nb07_reranking.py" ;;
        nb08) run_test "NB08" "test_nb08_distillation.py" ;;
        nb09) run_test "NB09" "test_nb09_finetuning.py" ;;
        nb10) run_test "NB10" "test_nb10_evaluation.py" ;;
        nb11) run_test "NB11" "test_nb11_annotation.py" ;;
        *) echo "Usage: $0 [nb06|nb07|nb08|nb09|nb10|nb11]"; exit 1 ;;
    esac
else
    # Run all
    PASSED=0
    FAILED=0
    for test in \
        "NB06:test_nb06_faiss.py" \
        "NB07:test_nb07_reranking.py" \
        "NB08:test_nb08_distillation.py" \
        "NB09:test_nb09_finetuning.py" \
        "NB10:test_nb10_evaluation.py" \
        "NB11:test_nb11_annotation.py"; do

        name="${test%%:*}"
        script="${test##*:}"
        run_test "$name" "$script"
        if [ $? -eq 0 ]; then
            PASSED=$((PASSED + 1))
        else
            FAILED=$((FAILED + 1))
        fi
    done

    echo ""
    echo "=========================================="
    echo "SUMMARY: $PASSED passed, $FAILED failed"
    echo "=========================================="
fi
