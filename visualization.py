import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

def plot_energy_history(energy_history, fci_energy, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(energy_history, 'b-', linewidth=2, label='VQE Energy')
    plt.axhline(y=fci_energy, color='r', linestyle='--', linewidth=2, label='FCI Energy')
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Energy (Ha)', fontsize=14)
    plt.title('Energy Convergence', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved energy plot to {filename}")

def plot_reward_history(reward_history, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, 'g-', linewidth=2)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.title('Training Reward History', fontsize=16)
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved reward plot to {filename}")

def plot_circuit(circuit, filename):
    try:
        # Try to create a nice visual diagram
        fig = circuit.draw(output='mpl', fold=100)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved circuit diagram to {filename}")
    except:
        # Fallback to text representation
        with open(filename + '.txt', 'w') as f:
            f.write(str(circuit))
        print(f"Saved circuit text representation to {filename}.txt")

def visualize_results(config):
    results_dir = "results"
    
    # Load data
    energy_history = np.load(f"{results_dir}/energy_history.npy")
    reward_history = np.load(f"{results_dir}/reward_history.npy")
    
    # Load circuit
    with open(f"{results_dir}/best_circuit.qpy", "rb") as f:
        best_circuit = qpy.load(f)[0]
    
    # Get FCI energy from config
    fci_energy = config.getfloat('MOL', 'fci_energy')
    
    # Create plots
    plot_energy_history(energy_history, fci_energy, f"{results_dir}/energy_history.png")
    plot_reward_history(reward_history, f"{results_dir}/reward_history.png")
    plot_circuit(best_circuit, f"{results_dir}/best_circuit.png")
