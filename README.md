# Re-Imaging Price Trends

Implementation of "Re-Imaging Price Trends" (Journal of Finance, 2023) - Converting stock price time series into candlestick chart images for CNN-based return prediction.

## 🚀 Quick Start

### 0. Colab Setup (Recommended)
For Google Colab environment:
```bash
# 1. Upload project to Google Drive
# 2. Run notebooks in order:
#    - 1_image_generation.ipynb (생성 완료)
#    - 2_model_training.ipynb (학습 & 평가)
```

### 1. Data Preparation
Generate candlestick images from stock data:
```bash
# Generate images for all time windows
python datageneration.py --image_days 5 --mode train
python datageneration.py --image_days 5 --mode test
python datageneration.py --image_days 20 --mode train  
python datageneration.py --image_days 20 --mode test
python datageneration.py --image_days 60 --mode train
python datageneration.py --image_days 60 --mode test
```

### 2. Model Training
Train models (single or ensemble):
```bash
# Single model training
python train.py --model CNN20d --image_days 20 --pred_days 20 --use_original_format

# Ensemble training (paper method: 5 independent runs)
python train.py --model CNN20d --image_days 20 --pred_days 20 --ensemble --ensemble_runs 5 --use_original_format
### 3. Model Evaluation
Evaluate with decile portfolio backtesting:
```bash
# Single model evaluation  
python test.py --model CNN20d --image_days 20 --pred_days 20 --use_original_format

# Ensemble evaluation
python test.py --model CNN20d --image_days 20 --pred_days 20 --ensemble --use_original_format

# 20-day models  
python main.py --model CNN20d --image_days 20 --pred_days 20

# 60-day models
python main.py --model CNN60d --image_days 60 --pred_days 60
```

### 3. Model Evaluation
```bash
# Test trained models
python test.py --model CNN5d --image_days 5 --pred_days 5
python test.py --model CNN20d --image_days 20 --pred_days 20
python test.py --model CNN60d --image_days 60 --pred_days 60
```

### 4. Generate Predictions
```bash
# Generate prediction factors for backtesting
python inference.py --model CNN5d --image_days 5 --pred_days 5
python inference.py --model CNN20d --image_days 20 --pred_days 20
python inference.py --model CNN60d --image_days 60 --pred_days 60
```

## 📊 Core Methodology

### Image Conversion
- **Representation**: Each trading day = 3 pixels (Open|High-Low bar|Close)
- **Image Sizes**: 
  - 5-day: 32×15 pixels
  - 20-day: 64×60 pixels  
  - 60-day: 96×180 pixels
- **Scaling**: Min-max normalization within each image
- **Colors**: Black background (0), white price data (255)

### Training Approach
- **Data Split**: Fixed split (1993-2000 train, 2001-2019 test)
- **No Rolling Windows**: Single model for entire test period
- **Class Balance**: Natural distribution (no SMOTE resampling)
- **Architecture**: CNN with BatchNorm, Dropout, Early Stopping

### Data Features
- **Source**: WRDS CRSP database
- **Period**: 1993-2019 (27 years)
- **Filtering**: IPO/delisting filter (>20% non-trading days excluded)
- **NA Handling**: Skip image generation if NA values in lookback window

## 🏗️ Architecture

### CNN Models
```
CNN5d:  Input(32×15)  → 2 conv blocks → 155K params → Binary output
CNN20d: Input(64×60)  → 3 conv blocks → 708K params → Binary output  
CNN60d: Input(96×180) → 4 conv blocks → 2.9M params → Binary output
```

### Training Configuration
- **Optimizer**: Adam (lr=1e-5)
- **Loss**: Binary Cross Entropy
- **Batch Size**: 128
- **Early Stopping**: 2 epochs
- **Train/Val Split**: 70:30 random split

## 📁 Project Structure

```
ReImaging_Price_Trends/
├── CLAUDE.md                              # Implementation guide
├── README.md                              # This file
├── data/
│   ├── data_1993_2000_train_val.parquet  # Training data
│   ├── data_2001_2019_test.parquet       # Test data
│   └── datageneration.ipynb              # WRDS data generation
├── main.py                                # Training script
├── train.py                               # Training pipeline
├── test.py                                # Model evaluation  
├── inference.py                           # Prediction generation
├── dataset.py                             # Image conversion core
├── model.py                               # CNN architectures
└── utils.py                               # Utilities
```

## 🎯 Expected Results

Based on the paper's findings:

### Weekly Strategy (5d→5d)
- **Sharpe Ratio**: 7.15 (Equal-Weight H-L portfolio)
- **Best Performance**: Short-term patterns most predictive

### Monthly Strategy (20d→20d)  
- **Sharpe Ratio**: 2.16 (Equal-Weight H-L portfolio)
- **Balanced**: Good risk-return trade-off

### Quarterly Strategy (60d→60d)
- **Sharpe Ratio**: 0.37 (Equal-Weight H-L portfolio)
- **Long-term**: Lower but still positive returns

## 🔬 Key Insights

1. **Visual Patterns Work**: CNNs extract meaningful patterns from price images
2. **Short-term is Best**: 5-day images provide strongest signals
3. **Cross-horizon Prediction**: Short images can predict long-term returns
4. **Outperforms Traditional**: Beats 7,846 technical indicators

## 📚 Citation

```bibtex
@article{jiang2023reimaging,
  title={Re-Imaging Price Trends},
  author={Jiang, Jingwen and Kelly, Bryan T and Xiu, Dacheng},
  journal={The Journal of Finance},
  year={2023},
  publisher={Wiley Online Library}
}
```

## 🔧 Requirements

### For Local Development:
```bash
pip install -r requirements.txt
```

### For Google Colab:
```python
!python setup_colab.py  # Automatic setup
```

### Core Requirements:
- Python 3.8+
- PyTorch 2.0+ (CUDA compatible)
- pandas 2.0+, numpy 1.24+, matplotlib 3.7+
- pyarrow 14.0+ (for parquet files)
- WRDS account (for data generation)

## 📄 License

This implementation is for research and educational purposes. Please cite the original paper when using this code.