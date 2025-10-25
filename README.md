# DreamerV3-Sokoban: World Model-Based RL for Puzzle Solving

This repository implements DreamerV3 with discrete world model adaptations for solving Sokoban puzzles. The project investigates whether learned world models can effectively simulate and plan in discrete, logic-intensive environments.

## Features

- **Discrete World Model**: CategoricalRSSM designed for discrete state spaces
- **Hierarchical Planning**: Multi-level state abstraction for complex puzzles  
- **Curriculum Learning**: Progressive difficulty scaling from simple to complex puzzles
- **Value-Guided Imagination**: Enhanced planning using learned value functions
- **Comprehensive Evaluation**: Metrics for planning quality, sample efficiency, and generalization

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
cd dreamer-sokoban

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
# or
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train with default configuration
python run_experiment.py train

# Train with specific experiment configuration
python run_experiment.py train --experiment medium_puzzle --steps 500000

# Train with curriculum learning
python run_experiment.py train --experiment fast_curriculum

# Train discrete-optimized model
python run_experiment.py train --experiment discrete_optimized
```

### Evaluation

```bash
# Evaluate trained agent
python run_experiment.py evaluate checkpoints/final_checkpoint.pt --num-episodes 100

# Evaluate with planning quality analysis
python run_experiment.py evaluate checkpoints/final_checkpoint.pt --eval-planning --save-report
```

### Visualization

```bash
# Visualize imagined trajectories
python run_experiment.py visualize checkpoints/final_checkpoint.pt --trajectory-length 20

# Visualize value function landscape
python run_experiment.py visualize checkpoints/final_checkpoint.pt --visualize-values
```

### Analysis

```bash
# Analyze training logs
python run_experiment.py analyze logs/run_name/

# Analyze curriculum progression
python run_experiment.py analyze logs/run_name/ --analyze-curriculum
```

## Project Structure

```
dreamer-sokoban/
├── dreamer/
│   ├── agent.py           # DreamerV3 agent implementation
│   └── world_model.py     # Discrete world model (CategoricalRSSM)
├── environments/
│   └── sokoban_wrapper.py # Sokoban environment wrapper
├── analysis/
│   ├── evaluation.py      # Evaluation framework
│   └── visualization.py   # Visualization tools
├── train.py              # Main training script
├── configs/
│   ├── train.yaml            # Default training configuration
│   └── experiment.yaml       # Experiment configurations
├── experiments/              # Experiment logs and results
├── notebooks/               # Analysis notebooks
└── docs/                   # Documentation
```

## Configuration

### Key Configuration Options

```yaml
# Environment settings
environment:
  use_curriculum: true      # Enable curriculum learning
  dim_room: [7, 7]         # Puzzle dimensions
  num_boxes: 2             # Number of boxes
  observation_size: [64, 64] # Observation resolution

# World model settings
agent:
  world_model:
    rssm:
      categories: 32       # Discrete categories per latent
      stochastic_size: 32  # Number of categorical latents
  training:
    imagination_horizon: 15 # Planning horizon
    kl_weight: 1.0         # KL divergence weight
```

### Experiment Configurations

- `small_puzzle`: 5x5 rooms with 1 box (quick testing)
- `medium_puzzle`: 7x7 rooms with 2 boxes (standard)
- `large_puzzle`: 10x10 rooms with 4 boxes (challenging)
- `discrete_optimized`: Optimized for discrete dynamics
- `planning_focused`: Enhanced planning capabilities

## Algorithm Details

### CategoricalRSSM

The Categorical Recurrent State Space Model replaces continuous latents with discrete categorical distributions:

- **Prior**: p(z_t | h_{t-1}, a_{t-1})
- **Posterior**: q(z_t | h_{t-1}, a_{t-1}, o_t)
- **Dynamics**: Deterministic GRU + stochastic categorical states

### Value-Guided Planning

Instead of uniform sampling during imagination:
1. Generate multiple candidate action sequences
2. Evaluate each using the learned value function
3. Select actions with highest expected returns

### Curriculum Learning

Progressive difficulty stages:
1. **Stage 0**: 5x5 room, 1 box
2. **Stage 1**: 7x7 room, 2 boxes
3. **Stage 2**: 10x10 room, 3 boxes
4. **Stage 3**: 13x13 room, 4 boxes

Advancement criteria: 80% success rate over 100 episodes

## Results & Metrics

### Performance Metrics
- **Success Rate**: Percentage of solved puzzles
- **Sample Efficiency**: Steps to reach performance thresholds
- **Planning Quality**: Correlation between imagined and actual trajectories
- **Generalization**: Performance on unseen puzzle configurations

### Expected Performance
- Baseline (7x7, 2 boxes): >60% success rate
- Scaled (10x10, 4 boxes): >40% success rate
- Sample efficiency: <500K environment steps

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in configuration
   - Decrease imagination horizon
   - Use smaller world model architecture

2. **Slow training**
   - Ensure GPU is being used: `--device cuda`
   - Reduce training frequency: increase `train_every`
   - Use smaller observation size

3. **Poor performance**
   - Check reward shaping is enabled
   - Increase KL weight for better world model
   - Try curriculum learning if not using

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{dreamersokoban2024,
  title={World Model-Based Reinforcement Learning for Complex Puzzle Solving},
  author={DreamerSokoban Team},
  year={2024},
  url={https://github.com/yourusername/dreamer-sokoban}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DreamerV3 paper and implementation by Danijar Hafner et al.
- gym-sokoban environment by mpSchrader
- Inspired by research in model-based RL and discrete planning
