# Gradient and Steepest Descent Methods for Unconstrained Optimization

A comprehensive educational notebook on gradient descent optimization with practical applications to linear regression.

---

## 📋 Overview

This project provides an in-depth exploration of gradient descent and optimization methods, featuring:

- **Theoretical foundations** of unconstrained optimization
- **Mathematical proofs** and derivations
- **Algorithm implementations** from scratch
- **Visualizations** of optimization processes
- **Real-world applications** including linear regression
- **Advanced techniques** like Stochastic Gradient Descent (SGD) and mini-batch gradient descent
- **Performance comparisons** with scikit-learn implementations

---

## 📚 Contents

| Section | Topics |
|---------|--------|
| **1-5** | Mathematical theory (gradient, descent direction, convergence) |
| **6-7** | Gradient descent implementation from scratch |
| **8-10** | Linear regression applications and comparisons |
| **11-12** | Advanced techniques (SGD, mini-batch, learning rates) |
| **13-14** | Conclusion and references |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Required Python packages (listed below)

### Quick Setup

1. **Navigate to project directory:**
   ```bash
   cd /path/to/gradient_descent_optimization
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib scikit-learn
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter notebook gradient_descent_optimization.ipynb
   ```

4. **Run cells sequentially** using `Shift + Enter`

---

## 📦 Requirements

```
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

Install all at once:
```bash
pip install numpy matplotlib scikit-learn jupyter
```

---

## 💻 Usage

### Running the Notebook

1. **Open in Jupyter:**
   ```bash
   jupyter notebook gradient_descent_optimization.ipynb
   ```

2. **Execute sequentially:** Press `Shift + Enter` for each cell

3. **Important:** Run cells in order as later sections depend on earlier computations

### Key Execution Sections

- **Sections 1-5**: Theory review (informational, no execution needed)
- **Sections 6-7**: Gradient descent implementation and testing
- **Sections 8-10**: Linear regression applications
- **Sections 11-12**: Advanced techniques and comparisons

---

## 📁 Project Structure

```
gradient-descent-optimization/
├── gradient_descent_optimization.ipynb  # Main interactive notebook
├── GRADIENT_DESCENT_README.md          # This file
└── results/                             # Generated visualizations
    ├── convergence_analysis.png
    ├── linear_regression_fit.png
    ├── learning_rate_effects.png
    └── optimization_trajectories.png
```

---

## ✨ Key Features

### 1. Comprehensive Theory
- Gradient computation and geometric interpretation
- Descent direction analysis  
- Convergence criteria and theoretical foundations
- Learning rate selection strategies

### 2. From-Scratch Implementations
```python
# General gradient descent
def gradient_descent(func, grad_func, x0, alpha=0.01, max_iter=1000, tol=1e-6)

# Linear regression with gradient descent
class LinearRegressionGD:
    def fit(self, X, y, alpha=0.001, max_iter=1000, tol=1e-6)
    def predict(self, X)

# Stochastic gradient descent
class LinearRegressionSGD:
    def fit(self, X, y, batch_size=1, alpha=0.001, max_iter=100)
```

### 3. Rich Visualizations
- 3D surface plots of optimization landscape
- Contour plots with descent trajectories
- Convergence curves (function value & gradient norm)
- Learning rate comparison analysis
- Batch size impact visualization

### 4. Real-World Applications

#### 1D Quadratic Function
```
Function: f(x) = x²
Starting point: x₀ = 10.0
Learning rate: α = 0.1
Result: x_opt ≈ 0.00014, f(x) ≈ 2.04e-08
```

#### Housing Price Prediction
```
Dataset: 200 properties with 3 features
Features: Area (sq ft), Bedrooms, Age (years)
Train-Test: 80-20 split
Training R²: 0.9617
Test R²: 0.9613
Convergence: 1779 iterations
```

#### Advanced Comparisons
- Batch Gradient Descent vs. SGD vs. Mini-batch
- Learning rate effect on convergence
- Comparison with scikit-learn LinearRegression

---

## 📊 Results Summary

### 1D Quadratic Function
| Metric | Value |
|--------|-------|
| Optimal point | x ≈ 0.00014 |
| Final function value | 2.04e-08 |
| Iterations to convergence | ~50 |
| Learning rate | 0.1 |

### Housing Price Prediction
| Metric | Value |
|--------|-------|
| Training R² | 0.9617 |
| Test R² | 0.9613 |
| Iterations | 1779 |
| Learning rate | 0.001 |
| Features | 3 (normalized) |

### Learning Rate Comparison
| Learning Rate | Convergence | Final Cost | R² Score | Status |
|---|---|---|---|---|
| 0.001 | Failed | 3.09e+10 | -1.93 | Diverged |
| 0.01 | Slow | 3.02e+08 | 0.963 | Works |
| 0.1 | 183 iter | 2.98e+08 | 0.961 | ✓ Optimal |
| 0.5 | 31 iter | 2.98e+08 | 0.961 | ✓ Optimal |

### Batch Size Effects
| Batch Size | Type | Final Cost | Notes |
|---|---|---|---|
| 1 | SGD | 2.99e+08 | Noisy, fast updates |
| 16 | Mini-batch | 3.02e+08 | Balanced approach |
| 160 | Batch | 3.07e+10 | Diverges, slow |

---

## 🔬 Mathematical Foundations

### Gradient Definition
For a function $f: \mathbb{R}^n \rightarrow \mathbb{R}$:

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### Gradient Descent Update
$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \alpha \nabla f(\mathbf{x}^{(k)})$$

Where:
- $\mathbf{x}^{(k)}$ = current point at iteration $k$
- $\alpha$ = learning rate (step size)
- $\nabla f(\mathbf{x}^{(k)})$ = gradient at current point

### Linear Regression Cost (MSE)
$$J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^{m} (h_\mathbf{w}(\mathbf{x}^{(i)}) - y^{(i)})^2$$

Where:
- $h_\mathbf{w}(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b$ = prediction
- $m$ = number of samples

---

## 🎯 Learning Objectives

After studying this notebook, you will understand:

✅ Mathematical foundations of gradient descent  
✅ How to implement optimization algorithms from scratch  
✅ Impact of learning rate on convergence  
✅ Application to real-world regression problems  
✅ Differences between batch, SGD, and mini-batch approaches  
✅ How to evaluate and compare optimization methods  
✅ Proper data preprocessing and feature normalization  
✅ Model validation and performance metrics  

---

## 📈 Key Visualizations

### Optimization Trajectory
- 3D surface plot of cost function landscape
- Contour plot with colored trajectory showing descent path
- Gradient vectors at selected points along the path
- Clear markers for start and end points

### Convergence Analysis
- Log-scale plot of function value vs. iterations
- Gradient norm decay over iterations
- Effect of learning rate on convergence speed
- Comparison across different optimization methods

### Linear Regression Fit
- Original data points with fitted regression line
- Residual plots showing prediction errors
- Train vs. test performance comparison
- R² score visualization

---

## 🔗 References

### Foundational Texts
1. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
2. Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

### Online Resources
- Andrew Ng's Machine Learning Specialization (Coursera)
- Stanford CS229: Machine Learning
- scikit-learn Documentation: https://scikit-learn.org

---

## 👤 Author Information

**Student**: Harsh Tripathi  
**Roll Number**: AD23B1021  
**Course**: Advanced Non-Linear Optimization  
**Instructor**: Dr. Arnav Ghosh  
**Institution**: Indian Institute of Information Technology (IIIT) Raichur  
**Date**: May 2026  

---

## 📝 License

This educational material is provided as-is for learning purposes. Use and modification are permitted with proper attribution.

---

## ⚠️ Important Notes

1. **Sequential Execution**: Always run cells in order from top to bottom
2. **Dependencies**: Later cells depend on variables created in earlier cells
3. **Display**: Matplotlib visualizations display inline in Jupyter
4. **Runtime**: Complete notebook execution takes approximately 5 minutes
5. **Data**: Uses synthetic data and built-in datasets (no downloads needed)

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'numpy'`
**Solution:**
```bash
pip install numpy matplotlib scikit-learn
```

### Issue: Plots not displaying
**Solution:** Ensure you're in Jupyter Notebook (not standard Python), or add:
```python
%matplotlib inline
```
at the beginning of the notebook.

### Issue: Slow convergence or divergence
**Solution:** Adjust the learning rate:
- If diverging: decrease α (try 0.001 or 0.01)
- If too slow: increase α (try 0.1 or 0.5)
- Recommended range: **0.001 to 0.5**

### Issue: Memory error with large datasets
**Solution:** Use mini-batch gradient descent with appropriate batch size:
```python
model.fit(X, y, batch_size=32, max_iter=1000)
```

---

## ✅ Verification Checklist

Use this to verify successful setup:

- [ ] All packages installed (`pip list | grep numpy`)
- [ ] Jupyter notebook opens without errors
- [ ] First code cell (imports) executes successfully
- [ ] Visualizations display with colors and labels
- [ ] All cells complete without exceptions
- [ ] Housing model R² > 0.96
- [ ] Housing model converges in < 2000 iterations
- [ ] SGD shows faster convergence than batch GD

---

## 🚀 Next Steps / Extensions

Possible improvements and extensions:

### Beginner Level
- [ ] Try different learning rates on synthetic data
- [ ] Plot individual cost function terms
- [ ] Visualize feature importance from weights

### Intermediate Level
- [ ] Implement momentum-based gradient descent
- [ ] Add L1/L2 regularization
- [ ] Implement adaptive learning rate (decrease over time)
- [ ] Add cross-validation for hyperparameter tuning

### Advanced Level
- [ ] Implement Adam optimizer
- [ ] Add constraint handling for constrained optimization
- [ ] Apply to non-linear regression (polynomial features)
- [ ] Implement accelerated gradient descent methods

---

## 📞 Support & Questions

For help with specific concepts:

1. **Theory Questions**: Review Sections 1-5 (Mathematical Background)
2. **Implementation Help**: Check Section 7 (Implementation Details)
3. **Application Issues**: See Section 8-10 (Real-World Applications)
4. **Advanced Topics**: Study Section 11-12 (Advanced Methods)

---

## 📖 Recommended Reading Order

1. Read Sections 1-5 (theory first)
2. Execute Sections 6-7 (implementation)
3. Study Sections 8-10 (applications)
4. Experiment with Sections 11-12 (advanced)
5. Review conclusions and references

---

**Happy Learning! 🎓**

*Last Updated: May 2026*
