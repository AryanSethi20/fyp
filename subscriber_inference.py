import random
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from paho.mqtt import client as mqtt_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("subscriber")

import matplotlib.pyplot as plt
from collections import deque

# class DQNModel(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(DQNModel, self).__init__()
#         self._modules = {
#             '0': nn.Linear(state_size, 128),
#             '1': nn.ReLU(),
#             '2': nn.Linear(128, 128),
#             '3': nn.ReLU(),
#             '4': nn.Linear(128, action_size)
#         }
#
#     def forward(self, x):
#         for layer in self._modules.values():
#             x = layer(x)
#         return x

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        # Match exactly the training model structure
        self.add_module('0', nn.Linear(state_size, 128))
        self.add_module('1', nn.ReLU())
        self.add_module('2', nn.Linear(128, 128))
        self.add_module('3', nn.ReLU())
        self.add_module('4', nn.Linear(128, action_size))

    def forward(self, x):
        for layer in self._modules.values():
            x = layer(x)
        return x

class InferenceSubscriber:
    def __init__(self, model_path="final_model.pth"):
        # MQTT Configuration
        self.client_id = f'subscriber-{random.randint(0, 1000)}'
        self.broker = "192.168.0.105"
        self.port = 1883

        # Topics
        self.status_update_topic = "artc1/status_update"
        self.ack_topic = "artc1/ack"
        self.new_actions_topic = "artc1/new_actions"

        # State tracking
        self.window_size = 5
        self.aoi_window = deque(maxlen=self.window_size)
        self.current_policy = 0
        self.mu = 2.0  # Starting values matching training
        self.lambda_rate = 1.0
        self.last_generation_time = None

        # Setup model with training parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model(model_path)

        # MQTT setup
        self.client = mqtt_client.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        # Metrics tracking
        self.metrics = {
            'aoi_values': [],
            'service_times': [],
            'policies': [],
            'mu_values': [],
            'lambda_values': [],
            'timestamps': []
        }

    def setup_model(self, model_path):
        # Match training parameters exactly
        self.policies = [0, 1]
        self.mu_values = np.linspace(2.0, 6.0, 2)  # Match your training values
        self.ratios = np.linspace(0.2, 0.9, 2)

        # Create action space
        self.action_space = []
        for policy in self.policies:
            for mu in self.mu_values:
                for ratio in self.ratios:
                    lambda_rate = mu * ratio
                    self.action_space.append((policy, mu, lambda_rate))

        # Initialize model
        state_size = 5  # [avg_aoi, current_policy, mu, lambda_rate, ratio]
        action_size = len(self.action_space)
        self.model = DQNModel(state_size, action_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info("Model loaded successfully")

    def get_state(self):
        """Get state vector matching training environment"""
        avg_aoi = np.mean(list(self.aoi_window)) if self.aoi_window else 0
        ratio = self.lambda_rate / self.mu if self.mu != 0 else 0
        return np.array([avg_aoi, self.current_policy, self.mu, self.lambda_rate, ratio])

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT Broker!")
            self.client.subscribe([
                (f"{self.status_update_topic}/CU", 0),
                (f"{self.status_update_topic}/ZW", 0),
            ])
        else:
            logger.error(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            logger.info(f"Received message: {msg.payload.decode()}")
            status_update = json.loads(msg.payload)
            current_time = time.time()

            # Calculate PAoI
            mu = status_update["service_rate"]
            service_time = np.random.exponential(scale=1/mu)
            paoi = current_time - status_update["generation_time"] + service_time

            # Update metrics
            self.metrics['aoi_values'].append(paoi)
            self.metrics['service_times'].append(service_time)
            self.metrics['policies'].append(status_update["policy"])
            self.metrics['mu_values'].append(mu)
            self.metrics['lambda_values'].append(self.lambda_rate)
            self.metrics['timestamps'].append(current_time)

            # Save metrics periodically
            if len(self.metrics['aoi_values']) % 100 == 0:  # Save every 100 updates
                self.save_metrics()

            # Add to window for state calculation
            self.aoi_window.append(paoi)

            # Make new decision if window is full
            if len(self.aoi_window) >= self.window_size:
                state = self.get_state()
                policy, new_mu, new_lambda = self.get_action(state)

                # Send new action to publisher
                action = {
                    "policy": "ZW" if policy == 1 else "CU",
                    "mu": new_mu,
                    "lambda_rate": new_lambda
                }
                if policy != status_update["policy"] or new_mu != status_update["mu"] or new_lambda != status_update["lambda"]:
                    self.client.publish(self.new_actions_topic, json.dumps(action))

                # Update local state
                self.current_policy = policy
                self.mu = new_mu
                self.lambda_rate = new_lambda

                # Log decision
                logger.info(f"New action: policy={action['policy']}, mu={new_mu:.2f}, lambda={new_lambda:.2f}")
                logger.info(f"Current avg AoI: {np.mean(self.aoi_window):.2f}")

            # Send ACK for ZW policy
            if status_update["policy"] == "ZW":
                self.client.publish(self.ack_topic, "ACK")

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

    def get_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            action_idx = torch.argmax(q_values).item()
            return self.action_space[action_idx]

    def save_metrics(self):
        metrics_dict = {
            'aoi_values': self.metrics['aoi_values'],
            'service_times': self.metrics['service_times'],
            'policies': self.metrics['policies'],
            'mu_values': self.metrics['mu_values'],
            'lambda_values': self.metrics['lambda_values'],
            'timestamps': self.metrics['timestamps'],
            'summary': {
                'avg_aoi': np.mean(self.metrics['aoi_values']),
                'max_aoi': np.max(self.metrics['aoi_values']),
                'policy_distribution': {
                    'CU': self.metrics['policies'].count('CU'),
                    'ZW': self.metrics['policies'].count('ZW')
                }
            }
        }
        with open('inference_metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)

    def run(self):
        try:
            self.client.connect(self.broker, self.port)
            self.client.loop_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down subscriber...")
            self.save_metrics()  # Save final metrics
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
        finally:
            self.client.disconnect()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    subscriber = InferenceSubscriber("/Users/aryansethi20/Downloads/fyp/Runs/9/final_model-5thSize.pth")
    subscriber.run()


# class PerformanceMetricsTracker:
#     def __init__(self, window_size=100):
#         # Real-time metrics
#         self.aoi_values = deque(maxlen=window_size)
#         self.service_rates = deque(maxlen=window_size)
#         self.policy_choices = deque(maxlen=window_size)
#         self.throughput_values = deque(maxlen=window_size)
#
#         # Statistics
#         self.total_messages = 0
#         self.total_successful = 0
#         self.start_time = time.time()
#
#         # Window size for moving averages
#         self.window_size = window_size
#
#     def update(self, aoi, service_rate, policy, success=True):
#         self.aoi_values.append(aoi)
#         self.service_rates.append(service_rate)
#         self.policy_choices.append(policy)
#
#         self.total_messages += 1
#         if success:
#             self.total_successful += 1
#
#         # Calculate throughput (messages per second)
#         elapsed_time = time.time() - self.start_time
#         throughput = self.total_successful / elapsed_time
#         self.throughput_values.append(throughput)
#
#     def plot_metrics(self):
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
#
#         # Plot AoI over time
#         ax1.plot(list(self.aoi_values))
#         ax1.set_title('Age of Information Over Time')
#         ax1.set_xlabel('Message Index')
#         ax1.set_ylabel('AoI (seconds)')
#         ax1.grid(True)
#
#         # Plot Service Rate Distribution
#         ax2.hist(self.service_rates, bins=20)
#         ax2.set_title('Service Rate Distribution')
#         ax2.set_xlabel('Service Rate')
#         ax2.set_ylabel('Frequency')
#         ax2.grid(True)
#
#         # Plot Policy Distribution
#         policies = list(self.policy_choices)
#         policy_counts = {'ZW': policies.count('ZW'), 'CU': policies.count('CU')}
#         ax3.bar(policy_counts.keys(), policy_counts.values())
#         ax3.set_title('Policy Distribution')
#         ax3.set_ylabel('Count')
#         ax3.grid(True)
#
#         # Plot Throughput over time
#         ax4.plot(list(self.throughput_values))
#         ax4.set_title('System Throughput')
#         ax4.set_xlabel('Message Index')
#         ax4.set_ylabel('Messages/Second')
#         ax4.grid(True)
#
#         plt.tight_layout()
#         plt.savefig('performance_metrics.png')
#         plt.close()
#
#     def get_summary_stats(self):
#         return {
#             'avg_aoi': np.mean(self.aoi_values),
#             'max_aoi': np.max(self.aoi_values),
#             'avg_service_rate': np.mean(self.service_rates),
#             'throughput': self.total_successful / (time.time() - self.start_time),
#             'success_rate': self.total_successful / self.total_messages if self.total_messages > 0 else 0
#         }
#
#     def save_metrics(self, filename='metrics.json'):
#         metrics = {
#             'aoi_values': list(self.aoi_values),
#             'service_rates': list(self.service_rates),
#             'policy_choices': list(self.policy_choices),
#             'throughput_values': list(self.throughput_values),
#             'summary_stats': self.get_summary_stats()
#         }
#         with open(filename, 'w') as f:
#             json.dump(metrics, f)
#
#
# class InferenceMetrics:
#     def __init__(self, log_interval=100):
#         self.log_interval = log_interval
#         self.reset()
#
#     def reset(self):
#         self.aoi_values = []
#         self.policies = []
#         self.mu_values = []
#         self.lambda_values = []
#         self.ratio_values = []
#         self.start_time = time.time()
#
#     def update(self, aoi, policy, mu, lambda_rate):
#         self.aoi_values.append(aoi)
#         self.policies.append(policy)
#         self.mu_values.append(mu)
#         self.lambda_values.append(lambda_rate)
#         self.ratio_values.append(lambda_rate / mu if mu != 0 else 0)
#
#     def get_stats(self):
#         return {
#             'aoi': {
#                 'mean': np.mean(self.aoi_values),
#                 'std': np.std(self.aoi_values),
#                 'min': np.min(self.aoi_values),
#                 'max': np.max(self.aoi_values)
#             },
#             'policy_distribution': {
#                 'CU': self.policies.count(0) / len(self.policies) * 100,
#                 'ZW': self.policies.count(1) / len(self.policies) * 100
#             },
#             'service_rate': {
#                 'mean': np.mean(self.mu_values),
#                 'std': np.std(self.mu_values)
#             },
#             'arrival_rate': {
#                 'mean': np.mean(self.lambda_values),
#                 'std': np.std(self.lambda_values)
#             },
#             'ratio': {
#                 'mean': np.mean(self.ratio_values),
#                 'std': np.std(self.ratio_values)
#             },
#             'duration': time.time() - self.start_time
#         }
#
#     def save_metrics(self, filename='inference_metrics.json'):
#         stats = self.get_stats()
#         metrics = {
#             'summary_stats': stats,
#             'raw_data': {
#                 'aoi_values': self.aoi_values,
#                 'policies': self.policies,
#                 'mu_values': self.mu_values,
#                 'lambda_values': self.lambda_values,
#                 'ratio_values': self.ratio_values
#             }
#         }
#         with open(filename, 'w') as f:
#             json.dump(metrics, f, indent=4)
#
#     def log_if_needed(self):
#         if len(self.aoi_values) % self.log_interval == 0:
#             stats = self.get_stats()
#             logger.info(f"\nPerformance Metrics (last {self.log_interval} updates):")
#             logger.info(f"AoI: {stats['aoi']['mean']:.2f} ± {stats['aoi']['std']:.2f}")
#             logger.info(f"Policy Split: CU={stats['policy_distribution']['CU']:.1f}%, "
#                         f"ZW={stats['policy_distribution']['ZW']:.1f}%")
#             logger.info(f"Avg μ: {stats['service_rate']['mean']:.2f}, "
#                         f"Avg λ: {stats['arrival_rate']['mean']:.2f}")
#             logger.info(f"Avg λ/μ ratio: {stats['ratio']['mean']:.2f}")
#
# class DQNModel(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(DQNModel, self).__init__()
#         self._modules = {
#             '0': nn.Linear(state_size, 128),
#             '1': nn.ReLU(),
#             '2': nn.Linear(128, 128),
#             '3': nn.ReLU(),
#             '4': nn.Linear(128, action_size)
#         }
#
#     def forward(self, x):
#         for layer in self._modules.values():
#             x = layer(x)
#         return x
#
#
# class InferenceSubscriber:
#     def __init__(self, model_path="final_model.pth"):
#         # MQTT Configuration
#         self.client_id = f'subscriber-{random.randint(0, 1000)}'
#         self.broker = "192.168.1.3"
#         self.port = 1883
#
#         # Topics
#         self.status_update_topic = "artc1/status_update"
#         self.ack_topic = "artc1/ack"
#         self.action_topic = "artc1/action"
#
#         # State tracking
#         self.window_size = 5
#         self.aoi_window = deque(maxlen=self.window_size)
#         self.max_aoi = 10.0  # Maximum expected AoI
#         self.current_policy = 0
#         self.mu = 2.0
#         self.lambda_rate = 1.0
#         self.last_generation_time = None
#
#         # Setup model
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.setup_model(model_path)
#
#         # MQTT setup
#         self.client = mqtt_client.Client(client_id=self.client_id)
#         self.client.on_connect = self.on_connect
#         self.client.on_message = self.on_message
#
#         # Add metrics tracking
#         self.metrics = InferenceMetrics(log_interval=100)
#
#         # Add periodic saving
#         self.save_interval = 1000  # Save every 1000 updates
#
#     def setup_model(self, model_path):
#         # Define action space
#         self.policies = [0, 1]
#         self.mu_values = np.linspace(1.0, 5.0, 8)
#         self.ratios = np.linspace(0.2, 0.9, 5)
#
#         # Create action space
#         self.action_space = []
#         for policy in self.policies:
#             for mu in self.mu_values:
#                 for ratio in self.ratios:
#                     lambda_rate = mu * ratio
#                     self.action_space.append((policy, mu, lambda_rate))
#
#         # Initialize model
#         state_size = 5  # [normalized_aoi, aoi_change, policy, normalized_ratio, prev_aoi]
#         action_size = len(self.action_space)
#         self.model = DQNModel(state_size, action_size).to(self.device)
#
#         # Load trained weights
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()
#         logger.info("Model loaded successfully")
#
#     def get_state(self):
#         avg_aoi = np.mean(list(self.aoi_window)) if self.aoi_window else 0
#         normalized_aoi = min(avg_aoi / self.max_aoi, 1.0)
#
#         # Get AoI change
#         prev_aoi = getattr(self, 'prev_aoi', normalized_aoi)
#         aoi_change = normalized_aoi - prev_aoi
#
#         # Calculate ratio
#         ratio = self.lambda_rate / self.mu if self.mu != 0 else 0
#         normalized_ratio = (ratio - 0.2) / (0.9 - 0.2)  # Normalize to [0,1]
#
#         return np.array([
#             normalized_aoi,
#             aoi_change,
#             self.current_policy,
#             normalized_ratio,
#             prev_aoi
#         ])
#
#     def get_action(self, state):
#         with torch.no_grad():
#             state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#             q_values = self.model(state)
#             action_idx = torch.argmax(q_values).item()
#             return self.action_space[action_idx]
#
#     def on_connect(self, client, userdata, flags, rc):
#         if rc == 0:
#             logger.info("Connected to MQTT Broker!")
#             self.client.subscribe([(f"{self.status_update_topic}/CU", 0),
#                                    (f"{self.status_update_topic}/ZW", 0)])
#         else:
#             logger.error(f"Failed to connect, return code {rc}")
#
#     def on_message(self, client, userdata, msg):
#         try:
#             status_update = json.loads(msg.payload)
#             current_time = time.time()
#
#             # Calculate AoI
#             if self.last_generation_time is not None:
#                 aoi = current_time - self.last_generation_time
#                 self.aoi_window.append(aoi)
#
#                 # Get new action from model
#                 state = self.get_state()
#                 policy, mu, lambda_rate = self.get_action(state)
#
#                 # Update local state
#                 self.current_policy = policy
#                 self.mu = mu
#                 self.lambda_rate = lambda_rate
#
#                 # Update metrics
#                 self.metrics.update(aoi, policy, mu, lambda_rate)
#                 self.metrics.log_if_needed()
#
#                 # Periodic saving of metrics
#                 if len(self.metrics.aoi_values) % self.save_interval == 0:
#                     self.metrics.save_metrics()
#
#                 # Update local state
#                 self.current_policy = policy
#                 self.mu = mu
#                 self.lambda_rate = lambda_rate
#
#                 # Send action update
#                 action = {
#                     "policy": "ZW" if policy == 1 else "CU",
#                     "mu": mu,
#                     "lambda_rate": lambda_rate
#                 }
#                 self.client.publish(self.action_topic, json.dumps(action))
#
#                 # Send ACK for ZW policy
#                 if policy == 1:
#                     self.client.publish(self.ack_topic, "ACK")
#
#             self.last_generation_time = current_time
#
#         except Exception as e:
#             logger.error(f"Error processing message: {e}", exc_info=True)
#
#     def run(self):
#         try:
#             self.client.connect(self.broker, self.port)
#             self.client.loop_forever()
#         except KeyboardInterrupt:
#             logger.info("Shutting down subscriber...")
#             self.metrics.save_metrics()  # Save final metrics
#         except Exception as e:
#             logger.error(f"Error: {e}", exc_info=True)
#         finally:
#             self.client.disconnect()
#
#
# if __name__ == "__main__":
#     path = "./Runs/1/final_model-1.pth"
#     subscriber = InferenceSubscriber(path)
#     subscriber.run()