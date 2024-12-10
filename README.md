# Project: Prediction of Loan Default Risk is Critical to Financial Institution Solvency: Comparing Markov Chain Monte Carlo vs. Machine Learning for Credit Risk Estimation
### Course: DSC 210: Numerical Linear Algebra
### Instructor: Dr. Lily Weng

#### Instructions:
* Ensure that the following libraries are installed in the Python 3 environment:
  - [**`ucimlrepo`**](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients): Connection to the UC Irvine Machine Learning Repository, data acquisition
  - [**`torch`**](https://pypi.org/project/torch/): Python package for tensor computation with NumPy and deep neural networks.
  - [**`numpy`**](https://numpy.org/install/): Numerical Python for scientific and engineering computing for multidimensional array data structures.
  - [**`pandas`**](https://pandas.pydata.org/docs/getting_started/install.html): Python package for data analysis, manipulation, and data structures.
  - [**`matplotlib`**](https://matplotlib.org/stable/users/getting_started/): A comprehensive library for creating static, animated, and interactive visualizations in Python.
  - [**`seaborn`**](https://seaborn.pydata.org/installing.html): Python statistical data visualization library based on MatPlotLib.
  - [**`sci-kit-learn`**](https://scikit-learn.org/stable/install.html#installation-instructions) (sklearn): Python machine learning and statistical modeling library for pre-processing, dimensionality reduction, classification, regression, clustering, and model selection techniques.
  - [**`tqdm`**](https://pypi.org/project/tqdm/): Python library displays the smart progress meter for iterable functions.
  - [**`pymc`**](https://www.pymc.io/projects/docs/en/latest/installation.html): A probabilistic programming library for Python that allows users to build Bayesian models with a simple Python API and fit them using Markov chain Monte Carlo (MCMC) methods.
  - [**`arviz`**](https://python.arviz.org/en/stable/getting_started/Installation.html): Python package for exploratory analysis of Bayesian models. It serves as a backend-agnostic tool for diagnosing and visualizing Bayesian inference.
  Typing defines a standard notation for Python functions and variable type annotations. This notation can document code in a concise, standard format.
  - [**`pytensor`**](https://pytensor.readthedocs.io/en/latest/install.html): Python library that allows you to define, optimize/rewrite, and evaluate mathematical expressions involving multi-dimensional arrays efficiently.
  - [**`scipy`**](https://scipy.org/install/): SciPy contains modules for optimization, linear algebra, integration, interpolation, special functions, FFT, signal and image processing, ODE solvers, and other tasks common in science and engineering.
  - [**`imblearn`**](https://imbalanced-learn.org/dev/install.html#:~:text=maintenance-,Install,-%23): Imbalanced-learn is an open source, MIT-licensed library relying on `sci-kit-learn`providing tools for classification with imbalanced classes.

* To perform the Markov-Chain Monte Carlo Analysis, open file [`MCMC.ipynb`](https://github.com/laketalkemp/DSC210Final_Project/blob/a4739e8a464b8b641587fceccae3490d3bbb36d4/MCMC.ipynb) and run all the notebook cells on a GPU for faster performance.

* To perform the Fully Connected Neural Network Analysis, open file [`FCNN.ipynb`](https://github.com/laketalkemp/DSC210Final_Project/blob/a4739e8a464b8b641587fceccae3490d3bbb36d4/FCNN.ipynb) and run all the notebook cells on a GPU for faster performance.

#### Data Description:
This research aimed to examine the case of customers' default payments in Taiwan and compare the predictive accuracy of the probability of default among six data mining methods. From the risk management perspective, the predictive accuracy of the estimated probability of default will be more valuable than the binary result of classification - credible or not credible clients. The actual probability of default as the response variable (Y) and the predictive probability of default as the independent variable (X). We have 30,000 credit customers, each described by 23 different features covering their payment history, demographic information, and billing patterns. This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
* Yeh, I. (2009). [Default of Credit Card Clients. UCI Machine Learning Repository](https://doi.org/10.24432/C55S3H).

### Feature Selection
Mutual Information (MI) scores are used to determine feature importance by measuring the degree of dependence between each feature and the target variable; a higher MI score indicates a stronger relationship between the feature and the target, signifying that the feature is more important for prediction purposes.

<p align="center">
<img width="516" alt="Feature Importance" src="https://github.com/laketalkemp/DSC210Final_Project/blob/d51a57002ff9c3f77c9a67e75b14badf261664a0/Feature%20Importance(Matplotlib).png">
</p>

### Results:
Our goal is to predict default risk - a binary outcome. <br>

#### **Markov-Chain Monte Carlo (MCMC) Analysis**
The MCMC model gives insight into the uncertainty in our $\beta$ coefficients. These are not just point estimates, but full distributions showing how each feature affects default risk. A Markov Chain is a stochastic process wherein the next state depends only on the current state: <br>
<p align="center">
  $p(X_{t+1} = \frac{x}{X_t} , X_{t-1} ,...) = p(X_{t+1} = \frac{x}{X_t})$
</p>
 This 'memory-less' property makes MCMC computationally feasible. Each step depends on where we are, not how we got here.

<p align="center">
  <img width="816" alt="MCMC Model" src="https://github.com/laketalkemp/DSC210Final_Project/blob/d3256b6aebb92afd9209d6f256df89c4328552be/MCMC%20Image.png">
</p>

Confusion Matrix:
The logR model, on the other hand, has a very high TN rate. However, the other three quadrants are roughly around the same fractions as one another, which doesn’t bode too well for differentiating defaulted customers from those who did not.

<p align="center">
<img width="516" alt="MCMC Confusion Matrix" src="https://github.com/laketalkemp/DSC210Final_Project/blob/d3256b6aebb92afd9209d6f256df89c4328552be/Confusion_matrix_MCMC.png">
</p>

Receiver Operating Characteristics (ROC) Curve:
The MCMC logR model performs with an AUC of 0.73. The ROC curve is a visualization of the performance of the binary classification model. It plots the true positive rate (TPR) and false positive rate (FPR) at different classification thresholds.
<p align="center">
<img width="516" alt="MCMC ROC Curve" src="https://github.com/laketalkemp/DSC210Final_Project/blob/d3256b6aebb92afd9209d6f256df89c4328552be/ROC_Curve_MCMC.png">
</p>

Classification Report:
<p align="center">
  <img width="516" alt="MCMC Classification Report" src="https://github.com/laketalkemp/DSC210Final_Project/blob/d3256b6aebb92afd9209d6f256df89c4328552be/Class_report_MCMC.png">
</p>

#### **Fully Connected Neural Network (FCNN) Analysis**
Artificial neural networks are increasing in prevalence for these kinds of tasks due to their efficacy, broad use cases, and ability to modulate their complexity; for this supervised learning binary classification task, we built a simple five-layer fully connected neural network. In the context of credit risk assessment, neural networks can effectively model the complex, nonlinear relationships between various financial and demographic factors that influence a borrower's creditworthiness. They can handle large and diverse datasets, identify subtle patterns, and adapt to changing economic conditions (Khashman, 2010). As a result, neural networks have the potential to improve the accuracy and reliability of credit risk predictions significantly.<br>

<p align="center">
  <img width="516" alt="Fully Connected Neural Network" src="https://github.com/laketalkemp/DSC210Final_Project/blob/72c1e6529cb37267d8f7e4ff2481417067b897a0/FCNN%20Image.png">
</p>

Confusion Matrix:
The FCNN has high True Negative and True Positive ratios, indicating its classification is reasonably accurate. <br>
<p align="center">
<img width="516" alt="FCNN Confusion Matrix" src="https://github.com/laketalkemp/DSC210Final_Project/blob/7998514d92b352f7d75c3bc105a37a908783c189/SOTA_confusion.JPG">
</p>

Classification Report:
Although the FCNN data has been weighted during training, model validation is performed using the original, unbalanced test data.

<p align="center">
  <img width="516" alt="FCNN Classification Report" src="https://github.com/laketalkemp/DSC210Final_Project/blob/d4bb55d966760e9d3444bbc0de16757bed031505/SOTA_report.JPG"> 
</p>

Receiver Operating Characteristics (ROC) Curve:
A ROC curve is another important metric of classification accuracy; the area under the curve, or AUC, represents the probability of correctly classifying a given data point, and the higher the AUC, the better; here, we see that the FCNN has AUC of 0.04 points higher than the MCMC logR, meaning that it classifies correctly more often. <br>

<p align="center">
  <img width="516" alt="FCNN Receiver Operator Curve" src="https://github.com/laketalkemp/DSC210Final_Project/blob/fe49e0fc7e6f51c0e1b84ed4874bffcea14defe4/SOTA_ROC.JPG">
</p>
