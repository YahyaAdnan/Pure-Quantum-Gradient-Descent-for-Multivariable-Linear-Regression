# Pure Quantum Gradient Descent (PQGD) for 2D Linear Regression

This notebook shows utilizing Qiskit a hybrid application of Pure Quantum Gradient Descent (PQGD) for a multivariate linear regression issue. By means of a proprietary oracle gate, the method builds a quantum circuit encoding the loss function into quantum phases. Classical post-processing and the Inverse Quantum Fourier Transform (IQFT) then help to retrieve the gradient information. Using classical techniques, sign detection, parameter updates, and convergence evaluation.

> **Note:** This implementation is based on the hybrid quantum-classical framework presented in Alsaedi (2025) and builds on the Pure Quantum Gradient Descent algorithm proposed by Chen et al. (2024).

--- 

## 1. Setup and Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import UnitaryGate, QFT
from math import pi
import time
```

First we import the necessary libraries. Whereas `matplotlib.pyplot` is used to show convergence and parameter evolution, the `numpy` and `math` modules manage numerical operations. Including particular libraries like `UnitaryGate` and `QFT` for building the oracle and inverse Fourier transform, `qiskit` offers the tools to develop and simulate quantum circuits. Data normalisation using `MinMaxScaler` from `sklearn` fits the encoding criteria of quantum systems.

---

## 2. Data Generation

```python
np.random.seed(42)
num_samples = 50
x1 = np.random.randint(0, 5, size=(num_samples, 1))
x2 = np.random.randint(0, 5, size=(num_samples, 1))
X = np.hstack([x1, x2])
y = 5.5 * X[:, 1] + 2.9 * X[:, 0]
```

We create a synthetic dataset with a target variable `y`, two independent variables `x1` and `x2`, and a linear connection between them modeled. Arbitrarily chosen coefficients (2.9 for `x1` and 5.5 for `x2`) build a ground-truth model. This controlled setting lets us assess if the PQGD method can recover these parameters by means of optimization.

## 3. Normalization Functions

```python
def normalize_X(X):
    scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
    X_norm = scaler.fit_transform(X)
    return X_norm, scaler

def normalize_y(y):
    y_min, y_max = np.min(y), np.max(y)
    y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
    return y_norm, y_min, y_max

def rescale_y(y_norm, y_min, y_max):
    return (y_norm + 1) / 2 * (y_max - y_min) + y_min
```

Because of limits in phase encoding and amplitude scaling, quantum circuits usually run best inside limited numerical ranges. We convert feature values into the range \[-0.5, 0.5] by means of `normalize_X`, which fits rather nicely with quantum states generated using Hadamard gates. Designed to map output labels into the range \[-1, 1], the function `normalize_y` qualifies for phase-based encoding into the oracle. At last, `rescale_y` returns the original scale for interpretation from the normalized forecasts.

---

## 4. PQGD Components

### 4.1 Superposition Preparation

```python
def prepare_superposition(n, k):
    qc = QuantumCircuit(n * k)
    for i in range(k):
        qc.h(range(i * n, (i + 1) * n))
    return qc
```

Hadamard gates applied to every qubit allows the quantum register to be first set. Every set of `n` qubits stands for a single dimension of a parameter. Placing them in a consistent superposition helps the circuit to assess a superposition of perturbations (deltas) around the present weight vector. Realizing quantum parallelism in the gradient estimating process depends on this arrangement.

---

### 4.2 Classical Loss Function

```python
def loss_function(w_vec, x_data, y_data):
    errors = [
        sum(w * x for w, x in zip(w_vec, x_row)) - y
        for x_row, y in zip(x_data, y_data)
    ]
    return 0.5 * sum(e ** 2 for e in errors)
```

We derive the traditional loss function by means of the mean squared error (MSE) formulation. Within the oracle construction and sign detection elements is this purpose. It offers a scalar cost number that shows the degree of fit of the current parameter vector for the training data.

### 4.3 Oracle Gate with Phase Encoding

```python
def apply_oracle_gate_multidim(qc, w_vec, x_data, y_data, n, m, l):
    # function body elided for brevity
```

This function creates a customized diagonal unitary gate that translates the scaled loss function into the phase of every computational basis state. Evaluating the loss function at perturbed weights helps one build the oracle gate by transferring this cost into a phase with formula exp(2πi·φ). PQGD's "pure" quantum gradient estimating character comes from this process.

---

### 4.4 Inverse Quantum Fourier Transform (IQFT)

```python
def apply_multi_iqft(qc, n, k):
    # function body elided
```

Reverse the encoding process and transform phase-encoded gradient information into measured amplitudes using the IQFT. Extraction of gradient magnitude estimates from quantum observations depends on this stage.

---

### 4.5 Gradient Extraction

```python
def measure_multi_gradient(statevector, n, m, k):
    # function body elided
```

We compute expectation values and measure the output statevector once the quantum circuit is simulated. These expectation values scale to project gradient magnitudes. This forms the hybrid update loop's quantum component.

---

### 4.6 Classical Sign Detection

```python
def detect_gradient_sign_component(w_vec, x_data, y_data, i, epsilon=0.01):
    # function body elided
```

Positive and negative gradients cannot be separated by quantum phase encoding by itself. We investigate the loss function at somewhat shifted weights on the classical side to ascertain the sign of every gradient component in order to solve this.

---

### 4.7 Classical Weight Update

```python
def classical_update_vector(w_vec, grad_vec, alpha):
    # function body elided
```

We update the weight vector with the signed gradient vector and given learning rate. Considered the classical part of the hybrid method, this stage reflects conventional gradient descent.


---

## 5. Quantum Gradient Descent Training Loop

```python
def quantum_gradient_descent_loop_multi(...):
    # function body elided
```

This operates to capture the whole hybrid training program. It starts the weights, gets the quantum circuit ready, codes the loss function, computes the gradient, implements classical sign correction, changes the parameters, and searches for early convergence with a patience-based criterion.

---

## 6. Execution and Training

```python
X_norm, scaler_X = normalize_X(X)
y_norm, y_min, y_max = normalize_y(y)

start_time = time.time()
final_weights, history, losses = quantum_gradient_descent_loop_multi(
    X_norm, y_norm, n=3, m=8, l=0.01, alpha=0.01, steps=1000
)
end_time = time.time()
```

We normalize the input and output data then run the hybrid PQGD training algorithm. Measurement of execution time helps to evaluate performance.

---

## 7. Rescaling and Interpretation

```python
original_weights = []
for i in range(len(final_weights)):
    scale = scaler_X.data_range_[i] / 1.0
    w_rescaled = final_weights[i] / scale
    original_weights.append(w_rescaled)
```

Following optimization, the learned weights are rescaled back to their original units applying the same scaling factor as the normalisation procedure. This lets us match the actual coefficients applied in the data generating phase with the optimal weights.


---

## 8. Plotting Loss and Weight Trajectories

### Loss Curve

```python
mse_history = [2 * loss for loss in losses]

plt.figure(figsize=(8, 5))
plt.plot(mse_history, label="MSE")
plt.title("Loss Convergence (PQGD)")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

Plotting the MSE across every iteration lets us see the convergence behavior. This enables us to evaluate the model's seamless and fast convergence to a solution.

### Weight Trajectories

```python
weights_history = np.array(history)

plt.figure(figsize=(8, 5))
for i in range(weights_history.shape[1]):
    plt.plot(weights_history[:, i], label=f"Theta {i}")
plt.title("Weights Over Iterations (PQGD)")
plt.xlabel("Iteration")
plt.ylabel("Weight Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

Over training, this graph shows the development of every parameter. It provides understanding of the direction of convergence and the consistency of the updates.

---


## References

\[1] Alsaedi, Y. A. G. (2025). *Quantum Gradient Descent for Linear Regression: A Hybrid Approach* (Master’s Thesis, Universiti Teknologi Malaysia).

\[2] Chen, R., Guang, Z., Guo, C., Feng, G., & Hou, S.-Y. (2024). Pure Quantum Gradient Descent Algorithm and Full Quantum Variational Eigensolver. *Frontiers of Physics*, 19(2), 21202. [https://doi.org/10.1007/s11467-024-1243-z](https://doi.org/10.1007/s11467-024-1243-z)
