# Qiskit Hackathon 2025: Quantum-Reinforcement-Learning
Optimize quantum circuit by reinforcement learning

This project uses a Proximal Policy Optimization (PPO) agent to train a quantum circuit for finding the ground state energy of a molecule (LiH) using the Variational Quantum Eigensolver (VQE) algorithm.

## 🚀 Getting Started

### Prerequisites

-   Python 3.9+
-   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/qiskit-hackathon-2025.git](https://github.com/YOUR_USERNAME/qiskit-hackathon-2025.git)
    cd qiskit-hackathon-2025
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

To start the training process, run the `main.py` script with the configuration file as a command-line argument:

```bash
python src/main.py src/config_lih.cfg
