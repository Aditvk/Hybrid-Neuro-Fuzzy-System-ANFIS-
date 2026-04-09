# Hybrid Neuro-Fuzzy System for Student Performance Prediction

## Project Overview
This project implements an Adaptive Neuro-Fuzzy Inference System (ANFIS) to predict student performance. It combines the human-readable rule structure of fuzzy logic with the adaptive learning capabilities of artificial neural networks.

## System Architecture
* **Inputs (3):** * Attendance (0-100%)
  * Assignment Marks (0-100)
  * Test Marks (0-100)
* **Output (1):** * Performance Score (0-100), mapped to levels: **Poor** (0-40), **Average** (40-70), **Good** (70-100).
* **Membership Functions:** Gaussian curves (differentiable for backpropagation).

## Integration of Fuzzy Rules and Neural Networks (How it works)
This system operates as a 5-layer feed-forward neural network that functions as a Takagi-Sugeno fuzzy inference system:

1. **The Fuzzy Logic Component (Interpretability):** Instead of a "black box" neural network, the system uses expert-defined linguistic variables (Poor, Average, Good). The network is structured so that its nodes represent fuzzy sets and IF-THEN rules.
2. **The Neural Network Component (Learning/Adaptation):** A pure fuzzy system relies on a human to perfectly guess the shape of the membership curves. This hybrid system uses **Backpropagation (Gradient Descent)** and **Least Squares Estimation**. 
   * As the system processes historical student data, the neural network calculates the error between its prediction and the actual student score.
   * It propagates this error backward, automatically tuning the parameters (mean and variance) of the Gaussian membership curves. 
   * This means the system "learns" the actual data distribution of the students, shifting the definitions of "Good" or "Poor" attendance to match reality rather than human guesswork.

## How to Run
1. Open MATLAB and navigate to the project directory.
2. Run the `student_performance_anfis.m` script.
3. The script will generate a synthetic student dataset, build the initial fuzzy system, train it using the ANFIS hybrid learning algorithm, and output test predictions to the console.
