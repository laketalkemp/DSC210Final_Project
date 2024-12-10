# Project: Prediction of Loan Default Risk is Critical to Financial Institution Solvency: Comparing Markov Chain Monte Carlo vs. Machine Learning for Credit Risk Estimation
### Course: DSC 210: Numerical Linear Algebra
### Instructor: Dr. Lily Weng

#### Instructions:
* Ensure that the following libraries are installed in the Python 3 environment:
  - **`ucimlrepo`**: Connection to the UC Irvine Machine Learning Repository, data acquisition
  - **`torch`**: Python package for tensor computation with NumPy and deep neural networks.
  - **`numpy`**: Numerical Python for scientific and engineering computing for multidimensional array data structures.
  - **`pandas`**: Python package for data analysis, manipulation, and data structures.
  - **`matplotlib`**: A comprehensive library for creating static, animated, and interactive visualizations in Python.
  - **`seaborn`**: Python statistical data visualization library based on MatPlotLib.
  - **`scikit-learn`** (sklearn): Python machine learning and statistical modeling library for pre-processing, dimensionality reduction, classification, regression, clustering, and model selection techniques.
  - **`tqdm`**: Python library displays the smart progress meter for iterable functions.
  - **`plotly`**: Interactive visualization library for Python
  - **`pymc`**: A probabilistic programming library for Python that allows users to build Bayesian models with a simple Python API and fit them using Markov chain Monte Carlo (MCMC) methods.
  - **`arviz`**: Python package for exploratory analysis of Bayesian models. It serves as a backend-agnostic tool for diagnosing and visualizing Bayesian inference.
  Typing defines a standard notation for Python functions and variable type annotations. This notation can document code in a concise, standard format.
  - **`pytensor`**: Python library that allows you to define, optimize/rewrite, and evaluate mathematical expressions involving multi-dimensional arrays efficiently.
  - **`manim`**: Python library for creating mathematical animations.
  - **`scipy`**: SciPy contains modules for optimization, linear algebra, integration, interpolation, special functions, FFT, signal and image processing, ODE solvers, and other tasks common in science and engineering.

* To perform the Markov-Chain Monte Carlo Analysis, open file `MCMC.ipynb` and run all the notebook cells.

* To perform the Fully Connected Neural Network Analysis, open file `FILENAME.ipynb` and run all the notebook cells. This file is preferred to be run on a GPU for faster performance.

#### Results:

* **Markov-Chain Monte Carlo Analysis (MCMC)**

Top 5 words for each of the topics:
<img width="828" alt="Screen Shot 2022-04-15 at 9 20 17 AM" src="https://user-images.githubusercontent.com/18485647/163595256-eaa19d1b-9c52-4cd2-9a63-b1dbcd728d23.png">

Distributions of top 3 topics to each document:
<p align="center">
<img width="33%" alt="Screen Shot 2022-04-15 at 9 19 11 AM" src="https://user-images.githubusercontent.com/18485647/163595137-f9ea7e72-1f20-417f-bad4-4161cfcbe2f3.png">
 <img width="33%" alt="Screen Shot 2022-04-15 at 9 20 03 AM" src="https://user-images.githubusercontent.com/18485647/163595234-1951d9bf-0a54-40ec-8002-50d0042be260.png">
 <img width="33%" alt="Screen Shot 2022-04-15 at 9 19 35 AM" src="https://user-images.githubusercontent.com/18485647/163595185-18ca8fcc-ef7d-48fe-b7a5-76bb485a80a2.png">
</p>
Visualized embedding of documents in 2-D space:
<img width="816" alt="Screen Shot 2022-04-15 at 9 23 07 AM" src="https://user-images.githubusercontent.com/18485647/163595568-e9d9fd26-986a-4c06-8bf8-3bfbb95dbc79.png">

* **Fully Connected Neural Network Analysis (FCNN)**
<img width="844" alt="Fully Connected Neural Network" src="https://github.com/laketalkemp/DSC210Final_Project/blob/72c1e6529cb37267d8f7e4ff2481417067b897a0/FCNN%20Image.png">

Confusion Matrix:
<img width="844" alt="FCNN Confusion Matrix" src="https://user-images.githubusercontent.com/18485647/163595579-638c74c8-27f4-4ee2-ade2-e62406422600.png">

Classification Report:
<p align="center">
  <img width="844" alt="FCNN Classification Report" src="https://user-images.githubusercontent.com/18485647/163595666-06612d57-978b-4245-b89f-72fdf4bc1f7e.png"> 
</p>

Receiver Operator Curve (ROC):
<p align="center">
  <img width="844" alt="FCNN Receiver Operator Curve" src="https://user-images.githubusercontent.com/18485647/163595676-8fb7feb8-7471-440a-a19f-8c2735f987cb.png">
</p>

Neural Network (FCNN) Validation Results:
<img width="844" alt="Loan Default Prediction Classification Validation Results" src="https://github.com/laketalkemp/DSC210Final_Project/blob/72c1e6529cb37267d8f7e4ff2481417067b897a0/FCNN%20Validation%20History.png">
