# NAS Toolkit for NeWRF

A comprehensive Neural Architecture Search toolkit specifically designed for NeWRF (Neural Wireless Radio Field) models.

## 🎯 Features

- **Two-stage NAS**: Architecture search followed by hyperparameter optimization
- **Real-time Visualization**: Live progress tracking with charts and graphs
- **Intelligent Pruning**: Baseline comparison with smart trial pruning
- **Multiple Test Modes**: Fast, standard, and complete testing options
- **English Interface**: All charts and logs in English
- **Automatic Analysis**: Detailed result analysis and reporting

## 📁 Files Overview

### Jupyter Notebooks
- `Quick_NAS_Test.ipynb` - Fast test notebook (3-5 minutes)
- `NAS_Complete_Workflow.ipynb` - Complete workflow with detailed analysis

### Python Scripts
- `optuna_nas.py` - Core NAS implementation with English visualization
- `run_nas.py` - Interactive launcher script
- `__init__.py` - Package interface and convenience functions

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Recommended)
```bash
cd nas_toolkit
jupyter notebook Quick_NAS_Test.ipynb    # Fast test (3-5 minutes)
# OR
jupyter notebook NAS_Complete_Workflow.ipynb  # Full workflow with detailed analysis
```

### Option 2: Interactive Launcher
```bash
cd nas_toolkit
python run_nas.py
```

### Option 3: Python API
```python
from nas_toolkit import run_nas

# Run with custom parameters
results = run_nas(architecture_trials=10, hyperparameter_trials=15)
```

## 📊 Generated Outputs

The toolkit generates several files:

### Charts (PNG format)
- `nas_progress.png` - Real-time progress monitoring
- `final_nas_results.png` - Final results summary  
- `detailed_nas_analysis.png` - Comprehensive analysis
- `test_english_chart.png` - Test visualization

### Configuration Files
- `best_architecture.pkl` - Best architecture parameters
- `final_best_params.pkl` - Complete best configuration
- `*.db` - Optuna study databases

### Log Files
- `fast_nas_test.log` - Fast test logs
- `intelligent_nas_test.log` - Standard test logs
- `nas_visualization_test.log` - Visualization test logs

## 📈 Visualization Features

### Real-time Progress Display
- **tqdm progress bars**: Detailed training metrics
- **Console status**: Every 30 seconds progress update
- **Stage transitions**: From baseline to architecture to hyperparameters

### Automatic Chart Generation
Four-panel charts showing:
1. **Architecture Search Progress**: Loss curves over trials
2. **Hyperparameter Search Progress**: Optimization trajectory
3. **Overall Progress Bars**: Stage completion percentages
4. **Performance Comparison**: Baseline vs best results

### Progress Bar Information
- **Loss**: Current validation loss
- **Best**: Current best loss found
- **SNR**: Signal-to-noise ratio (dB)
- **Δ**: Improvement percentage vs baseline
- **Params**: Model parameter count
- **LR**: Current learning rate

## 🔧 Advanced Usage

### Custom NAS Run
```python
from nas_toolkit import run_nas

# Custom configuration
results = run_nas(
    architecture_trials=30,
    hyperparams_trials=50,
    storage_url="sqlite:///my_nas.db"
)

arch_study, hp_study, best_params = results
```

### Load and Test Best Model
```python
from nas_toolkit import load_and_test_best_model

model, params = load_and_test_best_model('final_best_params.pkl')
print(f"Best model has {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Progress Monitoring
```python
from nas_toolkit import progress_tracker, create_progress_plot

# Check current progress
progress_tracker.print_status()

# Generate progress chart
create_progress_plot('my_progress.png')
```

## ⚙️ Configuration Options

### Architecture Search Parameters
- `n_layers`: Network depth (3-8)
- `hidden_dim`: Hidden layer size (64, 128, 256, 512)
- `dropout_rate`: Dropout probability (0.0-0.5)
- `activation`: Activation function (relu, leaky_relu, tanh, elu)
- `use_batch_norm`: Batch normalization (True/False)
- `init_method`: Weight initialization (he, xavier, orthogonal, normal)
- `skip_connections`: Skip connections (True/False)

### Hyperparameter Search Parameters  
- `batch_size`: Training batch size (256, 512, 1024, 2048, 4096)
- `learning_rate`: Learning rate (1e-5 to 1e-2)
- `weight_decay`: L2 regularization (1e-6 to 1e-2)
- `scheduler_patience`: LR scheduler patience (5-20)
- `scheduler_factor`: LR reduction factor (0.1-0.9)

## 📋 Requirements

### Dependencies
- `torch` - PyTorch deep learning framework
- `optuna` - Hyperparameter optimization
- `matplotlib` - Plotting and visualization
- `tqdm` - Progress bars
- `numpy` - Numerical computing
- `pandas` - Data manipulation

### Installation
```bash
pip install torch optuna matplotlib tqdm numpy pandas
```

## 🎯 Performance Tips

1. **Start with fast test**: Verify everything works before longer runs
2. **Monitor progress**: Watch for early convergence or issues  
3. **Use pruning**: Let intelligent pruning save time on poor trials
4. **Check visualizations**: Charts help diagnose search problems
5. **Save intermediate results**: Studies are saved to database

## 🐛 Troubleshooting

### Common Issues
- **Import errors**: Check if parent directory modules are accessible
- **CUDA memory**: Reduce batch size or use CPU if GPU memory insufficient  
- **Chinese font warnings**: Matplotlib warnings about missing Chinese fonts (safe to ignore)
- **Permission errors**: Ensure write permissions for output files

### Debug Mode
Set logging level for more details:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 License

This toolkit is part of the NeWRF project and follows the same license terms.

## 👨‍💻 Author

Created by GitHub Copilot for the NeWRF project team.

---

For detailed Chinese documentation, see:
- `NAS_使用说明.md` - Complete usage guide
- `可视化功能说明.md` - Visualization features 
- `可视化实现成功.md` - Implementation report
