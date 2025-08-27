import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class ActorNetwork(nn.Module):
    """
    Actor network for the PPO agent.

    Parameters:
    -----------
    state_dim: int
        Dimension of the state space.
    action_dim: array
        Dimension of the action space.
    """
    def __init__(self, state_dim, action_dim, num_qubits, config=None, max_gates=None):
        # Run the constructor of the parent class (nn.Module):
        super().__init__()

        '''
        Write your code here.
        '''
        num_gate_types = action_dim
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Gate type head
        self.gate_head = nn.Linear(128, num_gate_types)
        
        # Qubit selection heads
        self.qubit1_head = nn.Linear(128, num_qubits)  # Target qubit
        self.qubit2_head = nn.Linear(128, num_qubits)  # Control qubit/angle index

    def forward(self, state):
        """
        Forward pass.
        """

        '''
        Write your code here.
        '''
        
        x = self.shared_layers(state)
        
        # Get logits for each action component
        gate_logits = self.gate_head(x)
        qubit1_logits = self.qubit1_head(x)
        qubit2_logits = self.qubit2_head(x)
        '''
        # Create probability distributions
        gate_probs = F.softmax(gate_logits, dim=-1)
        qubit1_probs = F.softmax(qubit1_logits, dim=-1) 
        qubit2_probs = F.softmax(qubit2_logits, dim=-1)
        '''
        return gate_logits, qubit1_logits, qubit2_logits

class CriticNetwork(nn.Module):
    """
    Critic network for the PPO agent.

    Parameters:
    -----------
    state_dim: int
        Dimension of the state space.
    fc1_dims: int
        Number of neurons in the first hidden layer.
    fc2_dims: int
        Number of neurons in the second hidden layer.
    """
    def __init__(self, state_dim, fc1_dims=256, fc2_dims=256):
        # Run the constructor of the parent class (nn.Module):
        super().__init__()

        # Neural network layers:        
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

    def forward(self, state):
        """
        Forward pass.
        """
        value = self.value_network(state)
        return value.squeeze(-1)  # Remove extra dimension
        