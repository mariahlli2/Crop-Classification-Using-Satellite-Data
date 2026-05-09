# Crop Classification Using Satellite Data

A deep learning project for automated crop type classification using Sentinel-2 satellite imagery combined with auxiliary environmental data. The project explores how different covariate sources (climate, soil, topography) improve classification performance.

## Project Overview

This repository documents the progression of a crop classification system across three parts:

- **Part 1 (MCTNet Baseline)**: Implements the core MCTNet architecture using only Sentinel-2 temporal data
- **Part 2 (MCTNet + Covariates)**: Extends Part 1 by integrating auxiliary environmental variables
- **Part 3 (AttentionCNN)**: Builds on Part 2 with advanced attention mechanisms and regularization

Currently trained and evaluated on two regions: Arkansas and California.

### Progression Overview

```
Part 1: Sentinel-2 only
  ↓
Part 2: Sentinel-2 + Covariates (topography, climate, soil)
  ↓
Part 3: Sentinel-2 + Covariates + Enhanced Attention + Regularization
```

---

## Part 1: MCTNet Baseline — Temporal Classification from Satellite Data

### What It Does

Part 1 establishes the foundational MCTNet architecture for crop classification. It demonstrates how temporal patterns in Sentinel-2 imagery alone can effectively classify crops throughout the growing season:

- **Temporal Processing**: Ingests 36 timesteps of 10-band Sentinel-2 data
- **CNN Feature Extraction**: Learns temporal patterns through 1D convolutions
- **Transformer Attention**: Captures long-range temporal dependencies
- **Adaptive Positional Encoding (ALPE)**: Handles irregular observation gaps inherent to satellite data
- **Baseline Performance**: Establishes metrics without any auxiliary environmental data

### Architecture

```
Input: [B, T, 10] (batch, timesteps, Sentinel-2 bands)
  ↓
Conv1D Layers (10 → 64 channels)
  ↓
ALPE Positional Encoding
  ↓
Multi-Head Attention (5 heads)
  ↓
Transformer Blocks (Feed-forward + Layer Norm)
  ↓
Global Pooling
  ↓
Fully Connected → Classification
```

### Key Components

- **ALPE (Adaptive Location-based Positional Encoding)**: Unlike standard positional encodings, ALPE adjusts for missing observations (clouds, orbit gaps) by masking and learning adaptive positions
- **Temporal CNN**: Exploits the sequential nature of growing season data (dormancy → growth → harvest)
- **Lightweight Design**: Minimal parameters while capturing meaningful temporal dynamics

### Results

Part 1 establishes baseline performance on both states with only spectral information. Typical accuracies:
- Arkansas: 0.958% (5 crops)
- California: 0.867% (6 crops)

This baseline demonstrates that temporal satellite data alone carries sufficient signal, and improvement opportunities exist through additional data sources.

---

## Part 2: MCTNet + Covariates — Enhanced with Environmental Data

### What It Does

Part 2 builds directly on Part 1's proven architecture by augmenting it with supplementary environmental variables. The question: **Can topography, climate, and soil data improve upon the Part 1 baseline?**

Instead of just exploiting temporal spectral patterns, Part 2 fuses multiple data sources:

- **Temporal Sentinel-2**: Same 36-timestep, 10-band input from Part 1
- **Topographic Variables**: Elevation, landforms (affect water retention, microclimates)
- **Climate Data**: Temperature, precipitation, solar radiation (drive crop growth)
- **Soil Properties**: pH, organic carbon, texture (determine crop suitability, nutrient availability)
- **Dedicated Covariate Paths**: Each data source gets its own MLP branch before fusion

### Architecture

```
Part 1 Feature Path:
  Sentinel-2 [B, T, 10]
    ↓
  Conv1D + ALPE + Transformer
    ↓
  Spectral Features [B, 64]

Part 2 Covariate Paths (in parallel):
  Topography [B, 2] → MLP → [B, 32]
  Climate    [B, 3] → MLP → [B, 32]
  Soil       [B, 3] → MLP → [B, 32]

Fusion Layer:
  [Spectral Features] + [Topo Features] + [Climate Features] + [Soil Features]
    ↓
  Concatenate → [B, 128]
    ↓
  Dense Layers → Classification [B, N_CLASSES]
```
### Running Part 1 & 2

```bash

python main.py 


```

## Part 3: AttentionCNN Enhanced Architecture

### What It Does

Part 3 takes the proven Part 2 architecture and strengthens it with:

- **Better Attention**: CBAM (Convolutional Block Attention Module) for joint channel and spatial attention mechanisms
- **Modern Regularization**: DropPath, temporal masking, mixup, and label smoothing to prevent overfitting
- **Improved Training**: Warm-restart cosine scheduling, gradient clipping, better convergence
- **Maintained Covariate Support**: Keeps all Part 2 covariate integration but with improved scaling

### Running Part 3

```bash
python main_part3.py 
```

## Data Format

All of Part1 Part 2 and Part 3 expect CSV files in the `data/` directory with structure:

```
time_index, Blue, Green, Red, RE1, RE2, RE3, RE4, NIR, SWIR1, SWIR2, [covariates...], label
```

### Sentinel-2 Bands
- Blue, Green, Red: RGB channels
- RE1, RE2, RE3, RE4: Red Edge bands
- NIR: Near Infrared
- SWIR1, SWIR2: Shortwave Infrared

### Covariates
- `topo_elevation`, `topo_landforms`: Topographic features
- `clim_temp_mean`, `clim_precip_annual`, `clim_srad_annual`: Climate data
- `soil_ph`, `soil_oc`, `soil_texture`: Soil properties


## Getting Started

### Requirements
- Python 3.8+
- PyTorch 1.12+
- pandas, numpy, scikit-learn, matplotlib

### Install
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

### Quick Run
```bash
# Run Part 1 & 2 
python main.py 

# Run Part 3 
python main_part3.py
```

## License

This project is provided as-is for research and educational purposes.
