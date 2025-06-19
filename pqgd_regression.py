import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import UnitaryGate, QFT
from math import pi
import time

# =================== Set X & Y ========================

np.random.seed(42)
num_samples = 50
x1 = np.random.randint(0, 5, size=(num_samples, 1))
x2 = np.random.randint(0, 5, size=(num_samples, 1))
X = np.hstack([x1, x2])  
y = 5.5 * X[:, 1] + 2.9 * X[:, 0]

# =================== Normalizing Functions ========================

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


# =================== PQGD Logic ========================

def prepare_superposition(n, k):
    qc = QuantumCircuit(n * k)
    for i in range(k):
        qc.h(range(i * n, (i + 1) * n))
    return qc

def loss_function(w_vec, x_data, y_data):
    errors = [
        sum(w * x for w, x in zip(w_vec, x_row)) - y
        for x_row, y in zip(x_data, y_data)
    ]
    return 0.5 * sum(e ** 2 for e in errors)

def apply_oracle_gate_multidim(qc, w_vec, x_data, y_data, n, m, l):
    k = len(w_vec)
    N = 2 ** n
    total_states = 2 ** (n * k)
    oracle_diag = []

    for index in range(total_states):
        delta_bits = format(index, f'0{n * k}b')
        deltas = [
            int(delta_bits[i * n: (i + 1) * n], 2) - (N // 2)
            for i in range(k)
        ]

        delta_real = [l * delta / N for delta in deltas]
        w_shifted = [w + dx for w, dx in zip(w_vec, delta_real)]

        f_val = loss_function(w_shifted, x_data, y_data)
        phase = (N / (m * l)) * f_val
        oracle_diag.append(np.exp(2j * np.pi * phase))

    oracle_gate = UnitaryGate(np.diag(oracle_diag), label="O_f_multidim")
    qc.append(oracle_gate, qc.qubits)
    return qc

def apply_multi_iqft(qc, n, k):
    iqft_gate = QFT(n, inverse=True, do_swaps=True).to_gate()
    iqft_gate.name = "IQFT"
    for i in range(k):
        qc.append(iqft_gate, qc.qubits[i * n: (i + 1) * n])
    return qc

def measure_multi_gradient(statevector, n, m, k):
    N = 2 ** n
    probs = statevector.probabilities_dict()
    grad_sums = np.zeros(k)
    total_prob = 0

    for bitstring, prob in probs.items():
        total_prob += prob
        for i in range(k):
            offset = i * n
            bits = bitstring[offset:offset + n]
            index = int(bits, 2)
            grad_sums[i] += (m / N) * index * prob

    grad = grad_sums / total_prob
    return grad

def detect_gradient_sign_component(w_vec, x_data, y_data, i, epsilon=0.01):
    w_plus = w_vec.copy()
    w_minus = w_vec.copy()
    w_plus[i] += epsilon
    w_minus[i] -= epsilon
    f_plus = loss_function(w_plus, x_data, y_data)
    f_minus = loss_function(w_minus, x_data, y_data)
    return +1 if f_plus > f_minus else -1

def classical_update_vector(w_vec, grad_vec, alpha):
    updated_weights = [w - alpha * g for w, g in zip(w_vec, grad_vec)]
    return updated_weights

def quantum_gradient_descent_loop_multi(
    x_data, y_data, n=4, m=16, l=0.05, alpha=0.05, w_init=None, steps=300,
    tolerance=0.0001, patience=20
):
    k = len(x_data[0])
    w_vec = w_init if w_init is not None else [0.0] * k
    weights_history = []
    losses = []

    best_loss = None
    patience_counter = 0

    for step in range(steps):
        qc = prepare_superposition(n, k)
        apply_oracle_gate_multidim(qc, w_vec, x_data, y_data, n, m, l)
        apply_multi_iqft(qc, n, k)
        state = Statevector.from_instruction(qc)
        grad_vec = measure_multi_gradient(state, n, m, k)

        signed_grad_vec = []
        for i in range(k):
            sign = detect_gradient_sign_component(w_vec, x_data, y_data, i)
            signed_grad_vec.append(sign * grad_vec[i])

        w_vec = classical_update_vector(w_vec, signed_grad_vec, alpha)
        weights_history.append(w_vec.copy())
        current_loss = loss_function(w_vec, x_data, y_data)
        losses.append(current_loss)

        # Early stopping logic
        if best_loss is None:
            best_loss = current_loss
        elif abs(best_loss - current_loss) < tolerance:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at step {step+1}")
                break
        else:
            best_loss = current_loss
            patience_counter = 0

    return w_vec, weights_history, losses


# Normalize
X_norm, scaler_X = normalize_X(X)
y_norm, y_min, y_max = normalize_y(y)

# =================== MAIN EXECUTION ========================
start_time = time.time()
final_weights, history, losses = quantum_gradient_descent_loop_multi(
    X_norm, y_norm, n=3, m=8, l=0.01, alpha=0.01, steps=1000
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nExecution time: {elapsed_time:.4f} seconds")


# Rescale weights back
original_weights = []
for i in range(len(final_weights)):
    scale = scaler_X.data_range_[i] / 1.0  # range for [-0.5, 0.5] scaling
    w_rescaled = final_weights[i] / scale
    original_weights.append(w_rescaled)

mse_history = [2 * loss for loss in losses]

# Plot MSE loss over iterations
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


# Plot weights history
weights_history = np.array(history)  # shape: [steps, num_weights]

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

