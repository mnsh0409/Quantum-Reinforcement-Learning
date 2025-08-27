# Importing libraries: 
import os # For file system access.
import torch # For tensor operations.
from torch.optim import Adam, SGD # For gradient update.
from typing import Tuple # For typing annotation.
import numpy as np # For numerical operations.
import torch.distributions as dist

# Importing custom modules:
from src.actor_critic_networks import ActorNetwork, CriticNetwork
from src.memory import PPOMemory
import random
from collections import deque


class PPOAgent:
    '''
    Class for the PPO agent.

    Parameters:
    -----------
    state_dim: int
        Dimension of the state space.
    action_dim: array
        Dimension of the action space.
    learning_rate: float
        Learning rate for the optimizer.
    gamma: float
        Discount factor for future rewards.
    gae_lambda: float
        Lambda parameter for Generalized Advantage Estimation (GAE).
    policy_clip: float
        Clipping parameter for the policy loss.
    batch_size: int
        Batch size for training.
    num_epochs: int
        Number of epochs for training.
    optimizer_option: str
        Choice of optimizer ('Adam' or 'SGD').
    chkpt_dir: str
        Directory to save the model checkpoint.
    '''
    def __init__(self,
                 state_dim,
                 action_dim,
                 n_qubits=None,
                 config=None,
                 max_gates=None,
                 learning_rate=0.0003,
                 gamma=0.99,
                 gae_lambda=0.95,
                 policy_clip=0.2,
                 batch_size=64,
                 num_epochs=10,
                 optimizer_option="Adam",
                 chkpt_dir='model/ppo'):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.num_epochs = num_epochs

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Instantiate the Actor and Critic networks:
        self.actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim, num_qubits=n_qubits).to(self.device)
        self.critic = CriticNetwork(state_dim=state_dim).to(self.device)

        # Define a dictionary for optimizers:
        optimizers = {
            "Adam": Adam,
            "SGD": SGD
            }
        OptimizerClass = optimizers.get(optimizer_option, Adam) # Default to Adam if not found.
        
        # Define optimizers:
        self.actor_optimizer = OptimizerClass(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = OptimizerClass(self.critic.parameters(), lr=learning_rate)

        # Buffer for storing transitions:
        self.memory_buffer = PPOMemory(batch_size)

        # Checkpoint paths:
        self.actor_chkpt = os.path.join(chkpt_dir, 'actor_net_torch_ppo.pth')
        self.critic_chkpt = os.path.join(chkpt_dir, 'critic_net_torch_ppo.pth')

        self.use_dpo = config.getboolean('DPO', 'use_dpo', fallback=False)
        self.dpo_beta = config.getfloat('DPO', 'dpo_beta', fallback=0.1)
        self.dpo_loss_weight = config.getfloat('DPO', 'dpo_loss_weight', fallback=0.5)
        self.reference_update_freq = config.getint('DPO', 'reference_update_freq', fallback=5)
        self.preference_buffer_size = config.getint('DPO', 'preference_buffer_size', fallback=1000)
        
        # Reference policy network for DPO
        self.reference_actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim, num_qubits=n_qubits).to(self.device)
        self.reference_actor.load_state_dict(self.actor.state_dict())
        self.reference_actor.eval()
        
        # Preference buffer for DPO
        self.preference_buffer = deque(maxlen=self.preference_buffer_size)
        
        # Add episode counter
        self.episode_count = 0
        

    def validate_path(path):
        """Ensure path exists and is writable"""
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        if not os.access(dir_path, os.W_OK):
            raise PermissionError(f"Cannot write to {dir_path}")

    def save_models(self):
        # Validate paths first
        validate_path(self.actor_chkpt)
        validate_path(self.critic_chkpt)
        
        # Save models
        torch.save(self.actor.state_dict(), self.actor_chkpt)
        torch.save(self.critic.state_dict(), self.critic_chkpt)
        print(f"Models saved successfully to {self.actor_chkpt} and {self.critic_chkpt}")

    def sample_action(self, observation: torch.tensor) -> Tuple[list, list, float]:
        """
        Sample actions from the policy network given the current state (observation).

        Args:
            observation (torch.tensor): the state representation.

        Returns:
            action (list): list of action(s).
            probs (list): list of probability distribution(s) over action(s).
            value (float): the value from the Critic network.
        """
        # Convert observation to tensor and move to device
        #print('observation:', type(observation))
        # Ensure observation is always a numpy array
        if isinstance(observation, tuple):
            observation = observation[0]  # Extract state array from tuple
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        
        # Ensure proper shape (batch_size, state_dim)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension if missing
            
        # Get logits from actor
        with torch.no_grad():
            gate_logits, qubit1_logits, qubit2_logits = self.actor(state)
            
            # Create distributions
            gate_dist = dist.Categorical(logits=gate_logits)
            qubit1_dist = dist.Categorical(logits=qubit1_logits)
            qubit2_dist = dist.Categorical(logits=qubit2_logits)
            
            # Sample actions
            gate_action = gate_dist.sample()
            qubit1_action = qubit1_dist.sample()
            qubit2_action = qubit2_dist.sample()
            
            # Get log probabilities
            log_probs = torch.stack([
                gate_dist.log_prob(gate_action),
                qubit1_dist.log_prob(qubit1_action),
                qubit2_dist.log_prob(qubit2_action)
            ])

            # --- THIS IS THE FIX ---
            # Sum the log probabilities to get a single value for the composite action.
            total_log_prob = log_probs.sum().item()
        
        # Get state value
        value = self.critic(state)

        # Convert to CPU and Python scalars
        action = [
            gate_action.item(), 
            qubit1_action.item(), 
            qubit2_action.item()
        ]    
                
        return action, total_log_prob, value.item()

    def store_transitions(self, state, action, reward, probs, vals, done):
        """
        This method stores transitions in the memory buffer.
        """
        self.memory_buffer.store_memory(state, action, reward, probs, vals, done)

        # If using DPO, store preferences
        if self.use_dpo:
            # Store the current action and value as a preference
            self.preference_buffer.append((action, vals))
            
            # Update reference policy periodically
            if len(self.preference_buffer) >= self.reference_update_freq:
                self.update_reference_policy()

    def update_reference_policy(self):
        """
        Update the reference policy using the current actor's parameters.
        """
        #print(f"Updating reference policy at episode {self.episode_count}")
        #print(f"Preference buffer size before clear: {len(self.preference_buffer)}")
        # Copy current actor's parameters to reference actor
        self.reference_actor.load_state_dict(self.actor.state_dict())
        self.reference_actor.eval()
        #print("Reference policy updated.")
        # Optionally, clear the preference buffer
        self.preference_buffer.clear()
        #print("Preference buffer cleared.")
        #print(f"Preference buffer size: {len(self.preference_buffer)}")
        #print(f"Reference policy updated with {len(self.preference_buffer)} preferences.")
        #print(f"Reference policy updated with {len(self.preference_buffer)} preferences.")

    def dpo_loss(self, states, actions, preferred_actions):
        """
        Compute Direct Preference Optimization loss
        
        Args:
            states: Batch of states
            actions: Batch of sampled actions
            preferred_actions: Batch of preferred actions
            
        Returns:
            loss: DPO loss value
        """
        # Get policy log probabilities
        gate_logits, qubit1_logits, qubit2_logits = self.actor(states)
        
        # Create distributions
        gate_dist = torch.distributions.Categorical(logits=gate_logits)
        qubit1_dist = torch.distributions.Categorical(logits=qubit1_logits)
        qubit2_dist = torch.distributions.Categorical(logits=qubit2_logits)
        
        # Get log probabilities for sampled actions
        log_pi_gate = gate_dist.log_prob(actions[:, 0])
        log_pi_qubit1 = qubit1_dist.log_prob(actions[:, 1])
        log_pi_qubit2 = qubit2_dist.log_prob(actions[:, 2])
        log_pi = log_pi_gate + log_pi_qubit1 + log_pi_qubit2
        
        # Get log probabilities for reference policy
        with torch.no_grad():
            ref_gate_logits, ref_qubit1_logits, ref_qubit2_logits = self.reference_actor(states)
            ref_gate_dist = torch.distributions.Categorical(logits=ref_gate_logits)
            ref_qubit1_dist = torch.distributions.Categorical(logits=ref_qubit1_logits)
            ref_qubit2_dist = torch.distributions.Categorical(logits=ref_qubit2_logits)
            
            log_pref_gate = ref_gate_dist.log_prob(preferred_actions[:, 0])
            log_pref_qubit1 = ref_qubit1_dist.log_prob(preferred_actions[:, 1])
            log_pref_qubit2 = ref_qubit2_dist.log_prob(preferred_actions[:, 2])
            log_pref = log_pref_gate + log_pref_qubit1 + log_pref_qubit2
        
        # Compute DPO loss
        log_ratio = log_pi - log_pref
        dpo_loss = -torch.nn.functional.logsigmoid(self.dpo_beta * log_ratio).mean()
        
        return dpo_loss

    def update_preference_buffer(self, state, action, reward, next_state, next_action):
        """
        Update preference buffer with new experience
        """
        # Simple preference: prefer actions that lead to higher rewards
        preference = 1 if reward > 0 else 0
        
        self.preference_buffer.append({
            'state': state,
            'action': action,
            'preference': preference,
            'next_state': next_state,
            'next_action': next_action
        })
    
    def sample_preference_batch(self, batch_size):
        """
        Sample a batch of preferences from the buffer
        """
        if len(self.preference_buffer) < batch_size:
            return None
            
        batch = random.sample(self.preference_buffer, batch_size)
        states = []
        actions = []
        preferred_actions = []
        
        for item in batch:
            states.append(item['state'])
            actions.append(item['action'])
            # If preference is 1, current action is preferred
            if item['preference'] == 1:
                preferred_actions.append(item['action'])
            else:
                preferred_actions.append(item['next_action'])
        
        return (
            torch.tensor(np.array(states)), 
            torch.tensor(np.array(actions)),
            torch.tensor(np.array(preferred_actions)))
    
    def learn(self):
        """
        This method implements the PPO learning step.
        """
        # Sample all memories
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
            self.memory_buffer.generate_batches()
        
        # Convert to tensors
        states = torch.tensor(state_arr, dtype=torch.float32).to(self.device)
        old_probs = torch.tensor(old_prob_arr, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(reward_arr, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones_arr, dtype=torch.float32).to(self.device)
        actions = torch.tensor(action_arr, dtype=torch.long).to(self.device)
        old_values = torch.tensor(vals_arr, dtype=torch.float32).to(self.device)
        
        # Calculate advantages and returns
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        
        # Compute discounted returns and advantages
        for t in range(len(rewards) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards) - 1):
                a_t += discount * (rewards[k] + self.gamma * vals_arr[k+1] * (1 - dones[k]) - vals_arr[k])
                discount *= self.gamma * self.gae_lambda
            advantages[t] = a_t
            returns[t] = advantages[t] + vals_arr[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy for K epochs
        for _ in range(self.num_epochs):
            # Mini-batch update
            for batch in batches:
                # Get batch indices
                batch_indices = batch
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_probs = old_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Get new action probabilities
                gate_logits, qubit1_logits, qubit2_logits = self.actor(batch_states)
                
                # Create distributions
                gate_dist = torch.distributions.Categorical(logits=gate_logits)
                qubit1_dist = torch.distributions.Categorical(logits=qubit1_logits)
                qubit2_dist = torch.distributions.Categorical(logits=qubit2_logits)
                
                # Get new log probabilities
                gate_new_log_probs = gate_dist.log_prob(batch_actions[:, 0])
                qubit1_new_log_probs = qubit1_dist.log_prob(batch_actions[:, 1])
                qubit2_new_log_probs = qubit2_dist.log_prob(batch_actions[:, 2])
                
                # Combine log probabilities
                new_probs = gate_new_log_probs + qubit1_new_log_probs + qubit2_new_log_probs
                
                # Calculate probability ratio
                ratio = torch.exp(new_probs - batch_old_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * batch_advantages
                
                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_value = self.critic(batch_states).squeeze()
                critic_loss = torch.nn.functional.mse_loss(batch_returns, critic_value)
                
                # Total loss
                total_loss = actor_loss + 0.5 * critic_loss

                # Add DPO learning if enabled
                batch_size = len(batch)
                if self.use_dpo and len(self.preference_buffer) > batch_size:
                    # Sample DPO batch
                    dpo_states, dpo_actions, dpo_preferred_actions = self.sample_preference_batch(batch_size)
                    
                    if dpo_states is not None:
                        dpo_states = dpo_states.to(self.device)
                        dpo_actions = dpo_actions.to(self.device)
                        dpo_preferred_actions = dpo_preferred_actions.to(self.device)
                        
                        # Compute DPO loss
                        dpo_loss = self.dpo_loss(dpo_states, dpo_actions, dpo_preferred_actions)
                        
                        # Combine with PPO loss
                        total_loss = total_loss + self.dpo_loss_weight * dpo_loss
                        
                        # Log DPO loss
                        # (Add to your logging mechanism)
                
                # Update reference policy periodically
                #if self.use_dpo and episode % self.reference_update_freq == 0:
                #    self.reference_actor.load_state_dict(self.actor.state_dict())

                
                # Reset gradients
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                # Backpropagate
                total_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                
                # Update weights
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        # Reference policy update after each learning step (episode)
        if self.use_dpo and self.episode_count % self.reference_update_freq == 0:
            self.update_reference_policy()
            
        # Increment episode counter
        self.episode_count += 1
        
        # Clear memory after learning
        self.memory_buffer.clear_memory()
        #pass

    def save_models(self):
        print("Saving models...")
        torch.save(self.actor.state_dict(), self.actor_chkpt)
        torch.save(self.critic.state_dict(), self.critic_chkpt)

    def load_models(self):
        print("Loading models...")
        self.actor.load_state_dict(torch.load(self.actor_chkpt))
        self.critic.load_state_dict(torch.load(self.critic_chkpt))