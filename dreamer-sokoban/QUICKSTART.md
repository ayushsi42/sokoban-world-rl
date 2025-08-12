# DreamerV3-Sokoban Quick Start Guide

## Installation (5 minutes)

```bash
# Clone and enter directory
cd dreamer-sokoban

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Test Installation (2 minutes)

```bash
# Run implementation tests
python test_implementation.py
```

You should see all tests passing with checkmarks (✓).

## First Training Run (10 minutes)

### Option 1: Quick Test (Small Puzzle)
```bash
python run_experiment.py train --experiment small_puzzle --steps 10000
```

### Option 2: Standard Training
```bash
python run_experiment.py train --experiment medium_puzzle
```

### Option 3: Custom Configuration
```bash
python run_experiment.py train \
    --steps 100000 \
    --seed 42 \
    --run-name my_first_run
```

## Monitor Training

### Real-time Monitoring
Training progress is displayed in the terminal with:
- Episode rewards
- Success rates
- Training losses

### TensorBoard
```bash
tensorboard --logdir logs/
```
Then open http://localhost:6006 in your browser.

## Evaluate Trained Agent

```bash
# Basic evaluation
python run_experiment.py evaluate checkpoints/my_first_run/final_checkpoint.pt

# Detailed evaluation with visualization
python run_experiment.py evaluate checkpoints/my_first_run/final_checkpoint.pt \
    --eval-planning \
    --save-report \
    --render
```

## Visualize Agent Behavior

```bash
# Generate imagination GIF
python run_experiment.py visualize checkpoints/my_first_run/final_checkpoint.pt

# Visualize value function
python run_experiment.py visualize checkpoints/my_first_run/final_checkpoint.pt \
    --visualize-values
```

## Common Commands

### Training Variants
```bash
# No curriculum learning
python run_experiment.py train --experiment no_curriculum

# Fast curriculum progression
python run_experiment.py train --experiment fast_curriculum

# Discrete-optimized model
python run_experiment.py train --experiment discrete_optimized

# Enhanced planning
python run_experiment.py train --experiment planning_focused
```

### Resume Training
```bash
# Continue from checkpoint
python run_experiment.py train \
    --config configs/train.yaml \
    --checkpoint checkpoints/my_run/checkpoint_50000.pt
```

### Batch Evaluation
```bash
# Evaluate multiple checkpoints
for ckpt in checkpoints/my_run/checkpoint_*.pt; do
    python run_experiment.py evaluate $ckpt --num-episodes 50
done
```

## Troubleshooting

### GPU Memory Issues
```bash
# Reduce batch size
python run_experiment.py train --experiment baseline
```

### Slow Training
```bash
# Use CPU for testing
python run_experiment.py train --device cpu --steps 1000
```

### Installation Issues
```bash
# Install specific dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install gym-sokoban
```

## Next Steps

1. **Experiment with Configurations**: Try different experiment presets
2. **Analyze Results**: Use the analysis tools to understand agent behavior
3. **Modify Architecture**: Experiment with world model modifications in `src/dreamer/world_model.py`
4. **Create Custom Environments**: Modify `src/environments/sokoban_wrapper.py`

## Getting Help

- Check `README.md` for detailed documentation
- Review test outputs: `python test_implementation.py`
- Examine logs in `logs/` directory
- Check configuration options in `configs/`

Happy experimenting with DreamerV3-Sokoban! 🎮🤖
