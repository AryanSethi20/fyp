import time
import random
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from paho.mqtt import client as mqtt_client
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("subscriber")

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

class NonThreadedSubscriber:
    def __init__(self, model_path="final_model.pth", metrics_file = "inference_metrics.json"):
        # MQTT Configuration
        self.client_id = f'subscriber-{random.randint(0, 1000)}'
        self.broker = "192.168.0.105"
        self.port = 1883

        # Topics
        self.status_update_topic = "artc1/status_update"
        self.ack_topic = "artc1/ack"
        self.new_actions_topic = "artc1/new_actions"

        # State tracking
        self.window_size = 10
        self.current_policy = 0
        self.mu = 5.0
        self.lambda_rate = 1.5
        self.batch_id = 0

        # Setup model with training parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model(model_path)

        # MQTT setup
        self.client = mqtt_client.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        # State storage
        self.current_batch = []
        self.current_config = {
            "policy": "CU",
            "mu": self.mu,
            "lambda_rate": self.lambda_rate,
            "batch_id": 0
        }
        
        # Metrics tracking
        self.metrics = {
            'aoi_values': [],
            'theoretical_aoi_values': [],
            'service_times': [],
            'policies': [],
            'mu_values': [],
            'lambda_values': [],
            'timestamps': [],
            'batch_ids': []
        }
        
        # Running flag
        self.running = True

        self.ack_sizes = {
            "tiny": 16,       # 16 bytes
            "small": 64,      # 64 bytes
            "medium": 256,    # 256 bytes
            "large": 1024,    # 1KB
            "xlarge": 4096    # 4KB
        }
        
        # Predefined fixed ACK messages for each size
        self.ack_payloads = {
            # Tiny ACK (16 bytes)
            "tiny": '{"ack":"ACK"}',
            
            # Small ACK (64 bytes)
            "small": '{"ack":"ACK","ts":0,"id":"fixed-small-ack","padding":"XXXXXXXXXXXXX"}',
            
            # Medium ACK (256 bytes)
            "medium": '{"ack":"ACK","timestamp":0,"id":"fixed-medium-ack","size":"medium","padding":"' + 'X' * 185 + '"}',
            
            # Large ACK (1KB)
            "large": '{"ack":"ACK","timestamp":0,"id":"fixed-large-ack","size":"large","padding":"' + 'X' * 953 + '"}',
            
            # Extra large ACK (4KB)
            "xlarge": '{"ack":"ACK","timestamp":0,"id":"fixed-xlarge-ack","size":"xlarge","padding":"' + 'X' * 4025 + '"}'
        }

        self.ack_size = "tiny"
        
        self.metrics_file = metrics_file

    def setup_model(self, model_path):
        # Match training parameters exactly
        self.policies = [0, 1]
        self.mu_values = np.linspace(1.0, 9.0, 2)  # Match your training values
        self.ratios = np.linspace(0.3, 0.8, 2)  # Match your training values

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
        logger.info(f"Model {model_path} loaded successfully")

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
        """MQTT message handler"""
        try:
            current_time = time.time()
            status_update = json.loads(msg.payload)
            
            # Send ACK immediately for ZW policy to avoid timeouts
            if status_update["policy"] == "ZW":
                self.send_fixed_size_ack()
            
            # Calculate actual network delay
            network_delay = current_time - status_update["generation_time"]

            # Only for theoretical modeling (if needed)
            mu = status_update["service_rate"]
            service_time = np.random.exponential(scale=1/mu)
            theoretical_paoi = network_delay + service_time
            
            # Add to metrics
            self.metrics['aoi_values'].append(network_delay)
            self.metrics['theoretical_aoi_values'].append(theoretical_paoi)
            self.metrics['service_times'].append(service_time)
            self.metrics['policies'].append(status_update["policy"])
            self.metrics['mu_values'].append(mu)
            self.metrics['lambda_values'].append(self.lambda_rate)
            self.metrics['timestamps'].append(current_time)
            self.metrics['batch_ids'].append(self.batch_id)
            
            # Add to batch for processing
            self.current_batch.append(theoretical_paoi)
            
            # Periodically save metrics
            if len(self.metrics['aoi_values']) % 100 == 0:
                self.save_metrics()
                
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
            'theoretical_aoi_values': self.metrics['theoretical_aoi_values'],
            'service_times': self.metrics['service_times'],
            'policies': self.metrics['policies'],
            'mu_values': self.metrics['mu_values'],
            'lambda_values': self.metrics['lambda_values'],
            'timestamps': self.metrics['timestamps'],
            'batch_ids': self.metrics['batch_ids'],
            'summary': {  # Initialize this dictionary
                'avg_aoi': np.mean(self.metrics['aoi_values']),
                'avg_theoretical_aoi': np.mean(self.metrics['theoretical_aoi_values']),
                'max_aoi': np.max(self.metrics['aoi_values']),
                'policy_distribution': {
                    'CU': self.metrics['policies'].count('CU'),
                    'ZW': self.metrics['policies'].count('ZW')
                }
            }
        }
        
        # Now add the policy_per_mu data to the existing summary
        metrics_dict['summary']['policy_per_mu'] = {}
        for mu in set(self.metrics['mu_values']):
            mu_indices = [i for i, m in enumerate(self.metrics['mu_values']) if m == mu]
            policies = [self.metrics['policies'][i] for i in mu_indices]
            metrics_dict['summary']['policy_per_mu'][mu] = {
                'CU': policies.count('CU'),
                'ZW': policies.count('ZW'),
                'avg_aoi': np.mean([self.metrics['aoi_values'][i] for i in mu_indices])
            }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

    def send_fixed_size_ack(self):
        """Send ACK with a predefined fixed payload for the configured size"""
        try:
            # Get the predefined payload for the current ACK size
            payload = self.ack_payloads[self.ack_size]
            
            # Add timestamp to the payload if it's not tiny
            if self.ack_size != "tiny":
                # Replace the 0 timestamp with current time
                # This simple string replacement avoids having to re-parse and re-serialize JSON
                payload = payload.replace('"ts":0', f'"ts":{time.time()}')
                payload = payload.replace('"timestamp":0', f'"timestamp":{time.time()}')
            
            # Send the predefined ACK
            result = self.client.publish(self.ack_topic, payload)
            
            # Check if publish was successful
            if not result.rc == mqtt_client.MQTT_ERR_SUCCESS:
                logger.error(f"Failed to publish ACK, result code: {result.rc}")
                
        except Exception as e:
            logger.error(f"Error sending ACK: {e}")
            # Fallback to simple ACK in case of any error
            self.client.publish(self.ack_topic, "ACK")

    def process_batch(self):
        # Convert batch to state
        aoi_window = deque(self.current_batch, maxlen=self.window_size)
        avg_aoi = np.mean(aoi_window)
        ratio = self.lambda_rate / self.mu if self.mu != 0 else 0
        state = np.array([avg_aoi, self.current_policy, self.mu, self.lambda_rate, ratio])
        
        # Get action from model
        policy, new_mu, new_lambda = self.get_action(state)
        
        # Update configuration
        self.batch_id += 1  # Increment batch ID
        self.current_config = {
            "policy": "ZW" if policy == 1 else "CU",
            "mu": new_mu,
            "lambda_rate": new_lambda,
            "batch_id": self.batch_id
        }
        
        # Update local state
        self.current_policy = policy
        self.mu = new_mu
        self.lambda_rate = new_lambda
        
        # Send new action to publisher
        action = {
            "policy": "ZW" if policy == 1 else "CU",
            "mu": new_mu,
            "lambda_rate": new_lambda,
            "batch_id": self.batch_id
        }
        self.client.publish(self.new_actions_topic, json.dumps(action))
        
        # Log decision
        logger.info(f"Batch {self.batch_id}: New action: policy={action['policy']}, mu={new_mu:.2f}, lambda={new_lambda:.2f}")
        logger.info(f"Batch {self.batch_id}: Avg AoI: {avg_aoi:.2f}")
        
        # Clear the batch after processing
        self.current_batch = []

    def run(self):
        try:
            # Connect MQTT client
            self.client.connect(self.broker, self.port)
            self.client.loop_start()
            
            # Main loop
            while self.running:
                try:
                    # Check if we have enough data for a batch
                    if len(self.current_batch) >= self.window_size:
                        self.process_batch()
                    # Sleep a short time to prevent busy waiting
                    time.sleep(0.1)
                    
                except KeyboardInterrupt:
                    self.running = False
                    break
                
        except KeyboardInterrupt:
            logger.info("Shutting down subscriber...")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
        finally:
            self.running = False
            self.save_metrics()
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Subscriber shutdown complete")

def main():
    subscriber = NonThreadedSubscriber("/Users/aryansethi20/Downloads/fyp/ddqn/service-rate/new-config/best_double_dqn_model-rate=9.pth", "inference_metrics-ddqn-rate=9-3.json")
    subscriber.run()

if __name__ == "__main__":
    main()