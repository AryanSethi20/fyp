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
        
        # Add batch processing variables
        self.batch_size = 5
        self.current_batch = []
        self.batch_ready = False

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
            # logger.debug(f"Received message: {msg.payload.decode()}")
            status_update = json.loads(msg.payload)
            
            # Send ACK for ZW policy
            if status_update["policy"] == "ZW":
                self.client.publish(self.ack_topic, "ACK")
                
            # Calculate PAoI
            current_time = time.time()
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
            
            # Add to current batch
            self.current_batch.append(paoi)
            logger.debug(f"Added update to batch. Current size: {len(self.current_batch)}/{self.batch_size}")

            # Add to window for state calculation
            # self.aoi_window.append(paoi)
            
            # Process complete batch
            if len(self.current_batch) >= self.batch_size:
                
                # Update the AoI window with the full batch
                self.aoi_window = deque(self.current_batch, maxlen=self.batch_size)
                
                # Get state and make decision
                state = self.get_state()
                policy, new_mu, new_lambda = self.get_action(state)
                
                # Send new action to publisher
                action = {
                    "policy": "ZW" if policy == 1 else "CU",
                    "mu": new_mu,
                    "lambda_rate": new_lambda
                }
                self.client.publish(self.new_actions_topic, json.dumps(action))
                
                # if policy != status_update["policy"] or new_mu != status_update["mu"] or new_lambda != status_update["lambda"]:
                #     self.client.publish(self.new_actions_topic, json.dumps(action))

                # Update local state
                self.current_policy = policy
                self.mu = new_mu
                self.lambda_rate = new_lambda

                # Log decision
                logger.info(f"New action: policy={action['policy']}, mu={new_mu:.2f}, lambda={new_lambda:.2f}")
                logger.info(f"Current avg AoI: {np.mean(self.aoi_window):.2f}")
                
                # Clear the batch to start collecting a new one
                self.current_batch = []

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

def main():
    logging.basicConfig(level=logging.INFO)
    subscriber = InferenceSubscriber("/Users/aryansethi/Documents/Personal/fyp/Runs/9/final_model-6thSize.pth")
    subscriber.run()

if __name__ == "__main__":
    main()