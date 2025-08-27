import configparser # Config file.
import numpy as np # Numerical operations.
import ast # Convert string to list.
import sys # Command-line arguments.
import os # Directories.
from qiskit import qpy                                                # To save quantum circuits to Disk.
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_algorithms import VQE
import matplotlib.pyplot as plt  # For plotting results.

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0,cur_path+"/..")
print(f"Current path: {cur_path}")

# Helper functions:
#from src.helper_functions.save_qubit_op import save_qubit_op_to_file
from src.helper_functions.load_qubit_op import load_qubit_op_from_file

# Import the agent and environment classes:
from src.agent import PPOAgent
from src.env import VQEnv

##########################################
if __name__ == '__main__':
    # Parse command-line arguments:
    print(sys.argv)
    config_file = sys.argv[1]

    # Get the path to the config.cfg file:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_dir, config_file)
    print(f"Using configuration file: {config_file_path}")

    # Load the configuration file:
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Molecule hyperparameters:
    mol_name = config['MOL'].get('mol_name', fallback='Unknown')

    # Atoms:
    atoms_str = config['MOL'].get('atoms', fallback=None)
    atoms = ast.literal_eval(atoms_str) if atoms_str else []

    # Coordinates:
    coordinates_str = config['MOL'].get('coordinates', fallback=None)
    coordinates = ast.literal_eval(coordinates_str) if coordinates_str else ()

    # Number of particles:
    num_particles_str = config['MOL'].get('num_particles', fallback = None)
    num_particles = ast.literal_eval(num_particles_str) if num_particles_str else (0, 0)

    # Multiplicity:
    multiplicity = config.getint('MOL', 'multiplicity', fallback=1)
    # Charge:
    charge = config.getint('MOL', 'charge', fallback=0)
    # Electrons:
    num_electrons = config.getint('MOL', 'num_electrons', fallback = None)
    # Spatial orbitals:
    num_spatial_orbitals = config.getint('MOL', 'num_spatial_orbitals', fallback = None) 
    # Number of qubits:
    num_qubits = config.getint('MOL', 'num_qubits', fallback = None)
    gate_types = config['MOL'].get('gate_types', fallback='h,x,y,z,cx,cz,rx,ry,rz,rzz,t,sx')
    gate_types = gate_types.split(',') if gate_types else []
    # FCI energy:
    fci_energy = config.getfloat('MOL', 'fci_energy', fallback = None)

    # Convergence tolerance:
    conv_tol = config.getfloat('TRAIN', 'conv_tol', fallback=1e-5)

    # Training hyperparameters:
    learning_rate = config.getfloat('TRAIN', 'learning_rate', fallback=0.0003)
    gamma = config.getfloat('TRAIN', 'gamma', fallback=0.99) 
    gae_lambda = config.getfloat('TRAIN', 'gae_lambda', fallback=0.95) 
    policy_clip = config.getfloat('TRAIN', 'policy_clip', fallback=0.2) 
    batch_size = config.getint('TRAIN', 'batch_size', fallback=64) 
    num_episodes = config.getint('TRAIN', 'num_episodes', fallback=1000) # This is the number of episodes to train the agent.
    num_steps = config.getint('TRAIN', 'num_steps', fallback=50) # This is the number of steps per episode.
    num_epochs = config.getint('TRAIN', 'num_epochs', fallback=10) # This is the number of passes over the same batch of collected data for policy update.
    max_circuit_depth = config.getint('TRAIN', 'max_circuit_depth', fallback=50) 
    conv_tol = config.getfloat('TRAIN', 'conv_tol', fallback=1e-5)
    optimizer_option = config['TRAIN'].get('optimizer_option', fallback='SGD')

    ##########################################

    '''
    # Create an instance of the VQEnv class:
    env = VQEnv(molecule_name = "LiH", 
                symbols = atoms, 
                geometry = coordinates, 
                multiplicity = multiplicity, 
                charge = charge,
                num_electrons = num_electrons,
                num_spatial_orbitals = num_spatial_orbitals)

    # Save the qubit operator to disk:
    save_qubit_op_to_file(qubit_op = env.qubit_operator, file_name = "qubit_op_LiH.qpy")
    '''
    # At the beginning of main.py:
    os.makedirs('model/ppo', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load the qubit operator from disk:
    qubit_operator = load_qubit_op_from_file(file_path = "./src/operators/qubit_op_LiH.qpy")

    ##########################################

    # Create the environment with the loaded qubit operator:
    env = VQEnv(qubit_operator = qubit_operator, 
                num_spatial_orbitals = num_spatial_orbitals, 
                num_particles = num_particles,
                fci_energy = fci_energy)

    # Agent:
    agent = PPOAgent(
        state_dim = env.observation_space.shape[0],
        action_dim = len(gate_types), #env.action_space, #gate_types,
        n_qubits = env.num_qubits,
        #max_gates = env.max_gates,
        config=config,
        learning_rate = learning_rate,
        gamma = gamma,
        gae_lambda = gae_lambda,
        policy_clip =policy_clip,
        batch_size = batch_size,
        num_epochs = num_epochs,
        optimizer_option = optimizer_option,
        chkpt_dir = 'model/ppo')
    #action_dim=len(env.gate_types),
    #n_qubits=env.num_qubits,

    # Training loop:
    best_energy = float('inf')
    best_circuit = None
    best_params = None
    reward_history = []
    energy_history = []

    for i in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        episode_energy = float('inf')
        done = False
        truncated = False
        step_count = 0

        while not (done or truncated) and step_count < num_steps:
            # Sample action from agent
            action, log_prob, value = agent.sample_action(observation)

            # Take step in environment
            new_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition (pass log_prob directly)
            agent.store_transitions(observation, action, reward, 
                                    log_prob, value, done)

            # Update tracking variables
            episode_reward += reward
            observation = new_observation
            step_count += 1
            
            # Update energy if available
            if 'ep_energy' in info:
                episode_energy = np.max(info['ep_energy'])
        
        # Learn from experience after each episode
        agent.learn()
        
        # Save best circuit
        if episode_energy < best_energy:
            best_energy = episode_energy
            best_circuit = env.ansatz
            best_params = env.params
            print(f"New best energy: {best_energy:.6f} Ha")
        
        # Log progress
        reward_history.append(episode_reward)
        energy_history.append(episode_energy)
        
        print(f"Episode {i+1}/{num_episodes}: "
              f"Energy = {episode_energy:.6f} Ha, "
              f"Reward = {episode_reward:.2f}, "
              f"Steps = {step_count}")
        
        # At end of each episode
        if i % 100 == 0:
            plt.figure(figsize=(12, 4))
            
            # Energy plot
            plt.subplot(131)
            plt.plot(energy_history)
            plt.axhline(y=fci_energy, color='r', linestyle='--')
            
            # Reward plot
            plt.subplot(132)
            plt.plot(reward_history)
            
            # Save circuit diagram
            plt.subplot(133)
            best_circuit.draw(output='mpl')
            plt.savefig(f"results/episode_{i}.png")
            plt.tight_layout()
            plt.show()

        # Save models periodically
        if (i + 1) % 10 == 0:
            agent.save_models()
            print(f"Saved models at episode {i+1}")
        
        # Early stopping if converged
        if abs(episode_energy - fci_energy) < conv_tol:
            print(f"Converged to FCI energy within tolerance at episode {i+1}!")
            break
    
    # Save final results
    agent.save_models()
    print("Training completed. Saving final models.")
    
    optimizer = L_BFGS_B(maxiter=1000)
    vqe = VQE(estimator, best_circuit, optimizer, initial_point=best_params)
    result = vqe.compute_minimum_eigenvalue(env.qubit_operator)
    optimized_energy = result.eigenvalue.real

    # Save best circuit and parameters
    os.makedirs("results", exist_ok=True)
    with open("results/best_circuit.qpy", "wb") as f:
        qpy.dump(best_circuit, f)
    
    np.save("results/best_params.npy", best_params)
    np.save("results/reward_history.npy", reward_history)
    np.save("results/energy_history.npy", energy_history)
    
    print(f"Best energy achieved: {best_energy:.8f} Ha")
    print(f"FCI energy: {fci_energy:.8f} Ha")
    print(f"Difference: {abs(best_energy - fci_energy):.2e} Ha")