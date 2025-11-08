#!/bin/bash

# Full-Page Jazz Leadsheet Recognition - Experiment Runner
# This script runs all three experiments in sequence

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Full-Page Jazz Leadsheet Recognition - Experiment Runner      ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
EXP_NUM=${1:-"all"}  # Default to running all experiments
DEBUG=${2:-"false"}

# Helper function to run an experiment
run_experiment() {
    local exp_num=$1
    local exp_name=$2
    local exp_config=$3
    local exp_pretrained=$4
    local exp_freeze=$5
    local exp_epochs=$6

    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  EXPERIMENT $exp_num: $exp_name"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Config:      $exp_config"
    echo "Pretrained:  $exp_pretrained"
    echo "Freeze:      $exp_freeze"
    echo "Epochs:      $exp_epochs"
    echo ""

    read -p "Continue with Experiment $exp_num? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Starting Experiment $exp_num...${NC}"

        if [ "$DEBUG" = "true" ]; then
            python train_full_page.py \
                --config "$exp_config" \
                --load_pretrained "$exp_pretrained" \
                --freeze_encoder "$exp_freeze" \
                --epochs "$exp_epochs" \
                --batch_size 1 \
                --patience 5 \
                --output_name "exp${exp_num}_${exp_name}" \
                --debug True
        else
            python train_full_page.py \
                --config "$exp_config" \
                --load_pretrained "$exp_pretrained" \
                --freeze_encoder "$exp_freeze" \
                --epochs "$exp_epochs" \
                --output_name "exp${exp_num}_${exp_name}"
        fi

        echo -e "${GREEN}✓ Experiment $exp_num complete!${NC}"
        echo "Results saved to: weights/exp${exp_num}_${exp_name}/"
    else
        echo -e "${YELLOW}Skipping Experiment $exp_num${NC}"
    fi
}

# Check if data exists
if [ ! -d "data/handwritten" ]; then
    echo -e "${YELLOW}⚠ Warning: data/handwritten not found!${NC}"
    echo "Run: python data_prep.py --output_name handwritten"
    exit 1
fi

# Run experiments based on selection
case $EXP_NUM in
    1)
        run_experiment "1" "Baseline-System-Level" "config/full_page_baseline.gin" "True" "False" "10"
        ;;
    2)
        run_experiment "2" "From-Scratch" "config/full_page_no_pretrained.gin" "False" "False" "200"
        ;;
    3a)
        run_experiment "3a" "Pretrained-Frozen" "config/full_page_pretrained.gin" "True" "True" "100"
        ;;
    3b)
        run_experiment "3b" "Pretrained-FineTuned" "config/full_page_pretrained.gin" "True" "False" "150"
        ;;
    all)
        echo "Running all experiments in sequence..."

        run_experiment "1" "Baseline-System-Level" "config/full_page_baseline.gin" "True" "False" "10"

        run_experiment "2" "From-Scratch" "config/full_page_no_pretrained.gin" "False" "False" "200"

        run_experiment "3a" "Pretrained-Frozen" "config/full_page_pretrained.gin" "True" "True" "100"

        run_experiment "3b" "Pretrained-FineTuned" "config/full_page_pretrained.gin" "True" "False" "150"

        echo ""
        echo "╔════════════════════════════════════════════════════════════════╗"
        echo "║  ALL EXPERIMENTS COMPLETE!                                     ║"
        echo "╚════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Results locations:"
        echo "  Exp 1: weights/exp1_Baseline-System-Level/"
        echo "  Exp 2: weights/exp2_From-Scratch/"
        echo "  Exp 3a: weights/exp3a_Pretrained-Frozen/"
        echo "  Exp 3b: weights/exp3b_Pretrained-FineTuned/"
        echo ""
        echo "Next step: Compare results using EXPERIMENT_GUIDE.md"
        ;;
    help)
        echo ""
        echo "Usage: bash run_experiments.sh [EXP_NUM] [DEBUG]"
        echo ""
        echo "EXP_NUM options:"
        echo "  1      Run Experiment 1 (Baseline - System-Level Checkpoint)"
        echo "  2      Run Experiment 2 (From Scratch)"
        echo "  3a     Run Experiment 3a (Pretrained - Frozen Encoder)"
        echo "  3b     Run Experiment 3b (Pretrained - Fine-Tuned Encoder)"
        echo "  all    Run all experiments sequentially (default)"
        echo ""
        echo "DEBUG options:"
        echo "  true   Run in debug mode (smaller batches, fewer epochs)"
        echo "  false  Run normally (default)"
        echo ""
        echo "Examples:"
        echo "  bash run_experiments.sh 1              # Run only Experiment 1"
        echo "  bash run_experiments.sh all            # Run all experiments"
        echo "  bash run_experiments.sh 2 true         # Run Exp 2 in debug mode"
        echo ""
        ;;
    *)
        echo "Unknown experiment: $EXP_NUM"
        echo "Use 'bash run_experiments.sh help' for usage"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
