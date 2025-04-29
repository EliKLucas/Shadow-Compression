# Relational Melting: Visualizing Structural Resilience in Complex Networks

This project implements a novel method for visualizing and analyzing how different types of networks respond to progressive noise injection, termed "Relational Melting". By projecting small subgraphs (termed "snowflakes") from diverse systems into 2D "relational shadows", we can track how structural resilience manifests differently across domains.

## Overview

The project analyzes five types of networks:
- Worm Brain (modular small-world network)
- Random Graph (Erdős–Rényi random network)
- Social Graph (Barabási–Albert scale-free network)
- Knowledge Graph (synthetic modular semantic network)
- Protein Graph (folded highly modular structure)

For each network type, the code:
1. Extracts "snowflakes" (small subgraphs)
2. Injects progressive noise
3. Projects the structures into 2D space using UMAP
4. Creates visualizations showing how the structures "melt" under noise
5. Computes trustworthiness metrics to quantify structural resilience

## Installation

1. Clone this repository:
```bash
git clone https://github.com/EliKLucas/Shadow-Compression.git
cd Shadow-Compression
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main animation generation code is in `Melting_Animations/better_melting_animation.py`. To run it:

```bash
python Melting_Animations/better_melting_animation.py
```

This will:
1. Generate a base worm brain graph
2. Extract snowflakes
3. Create frames showing progressive noise injection
4. Save the animation as 'better_snowflake_melting.gif'

## Project Structure

- `Melting_Animations/`: Contains the main animation generation code
- `Paper/`: Contains the LaTeX source for the accompanying paper
- `frames_better/`: Directory where animation frames are saved

## Key Features

- Snowflake extraction from complex networks
- Progressive noise injection simulation
- UMAP-based dimensionality reduction
- Trustworthiness metric computation
- Animated visualization of structural collapse
- Comparative analysis across network types

## Citation

If you use this code in your research, please cite:

```
@article{lucas2023relational,
  title={Relational Shadows: A Comparative Analysis of Structural Resilience Across Complex Systems},
  author={Lucas, Elijah},
  year={2023}
}
```

## Contact

For questions or feedback, please contact: elijah.lucas.research@gmail.com 