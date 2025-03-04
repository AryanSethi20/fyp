import time
import random
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from paho.mqtt import client as mqtt_client
from collections import deque
import threading
from queue import Queue

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

class MultithreadedSubscriber:
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
        self.current_policy = 0
        self.mu = 2.0
        self.lambda_rate = 1.0

        # Setup model with training parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model(model_path)

        # MQTT setup
        self.client = mqtt_client.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        # Thread-safe data structures
        self.paoi_queue = Queue()
        self.batch_ready = threading.Event()
        self.config_updated = threading.Event()
        self.current_batch = []
        self.current_config = {
            "policy": "CU",
            "mu": self.mu,
            "lambda_rate": self.lambda_rate,
            "batch_id": 0
        }
        
        # Locking for thread safety
        self.batch_lock = threading.Lock()
        self.config_lock = threading.Lock()
        
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

    def setup_model(self, model_path):
        # Match training parameters exactly
        self.policies = [0, 1]
        self.mu_values = np.linspace(1.0, 5.0, 2)  # Match your training values
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
        """MQTT message handler - runs in MQTT thread"""
        try:
            current_time = time.time()
            status_update = json.loads(msg.payload)
            
            # Send ACK immediately for ZW policy to avoid timeouts
            if status_update["policy"] == "ZW":
                # logger.info(f"ACK sent for update - {status_update["total_updates"]}")
                self.client.publish(self.ack_topic, "ACK")
                
            # # Calculate PAoI
            # current_time = time.time()
            # service_time = np.random.exponential(scale=1/mu)
            # paoi = current_time - status_update["generation_time"] + service_time

            # Calculate actual network delay
            network_delay = current_time - status_update["generation_time"]

            # Only for theoretical modeling (if needed)
            mu = status_update["service_rate"]
            service_time = np.random.exponential(scale=1/mu)
            theoretical_paoi = network_delay + service_time

            # Save the real delay as your primary metric
            
            # Add to metrics
            with self.config_lock:
                batch_id = self.current_config["batch_id"]
            
            self.metrics['aoi_values'].append(network_delay)
            self.metrics['theoretical_aoi_values'].append(theoretical_paoi)  # Optional
            self.metrics['service_times'].append(service_time)
            self.metrics['policies'].append(status_update["policy"])
            self.metrics['mu_values'].append(mu)
            self.metrics['lambda_values'].append(self.lambda_rate)
            self.metrics['timestamps'].append(current_time)
            self.metrics['batch_ids'].append(batch_id)
            
            # Add to processing queue
            self.paoi_queue.put({
                'paoi': network_delay, 
                'policy': status_update["policy"],
                'mu': mu,
                'time': current_time,
                'batch_id': batch_id
            })
            
            # Periodically save metrics
            if len(self.metrics['aoi_values']) % 100 == 0:
                self.save_metrics()
                
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

    def process_updates_thread(self):
        """Thread that processes incoming updates and makes decisions"""
        logger.info("Started update processing thread")
        
        batch_id = 0
        while self.running:
            try:
                # Process queue into batches
                current_batch = []
                current_batch_id = batch_id
                
                # Collect exactly window_size items
                while len(current_batch) < self.window_size and self.running:
                    try:
                        # Short timeout to allow checking running flag
                        update = self.paoi_queue.get(timeout=0.5)
                        current_batch.append(update['paoi'])
                        self.paoi_queue.task_done()
                    except Exception:
                        # Queue empty or timeout, continue
                        continue
                
                if not self.running:
                    break
                
                # Process complete batch
                if len(current_batch) == self.window_size:
                    # Convert batch to state
                    aoi_window = deque(current_batch, maxlen=self.window_size)
                    avg_aoi = np.mean(aoi_window)
                    ratio = self.lambda_rate / self.mu if self.mu != 0 else 0
                    state = np.array([avg_aoi, self.current_policy, self.mu, self.lambda_rate, ratio])
                    
                    # Get action from model
                    policy, new_mu, new_lambda = self.get_action(state)
                    
                    # Update configuration atomically
                    with self.config_lock:
                        batch_id += 1  # Increment batch ID
                        self.current_config = {
                            "policy": "ZW" if policy == 1 else "CU",
                            "mu": new_mu,
                            "lambda_rate": new_lambda,
                            "batch_id": batch_id
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
                        "batch_id": batch_id
                    }
                    self.client.publish(self.new_actions_topic, json.dumps(action))
                    
                    # Log decision
                    logger.info(f"Batch {batch_id}: New action: policy={action['policy']}, mu={new_mu:.2f}, lambda={new_lambda:.2f}")
                    logger.info(f"Batch {batch_id}: Avg AoI: {avg_aoi:.2f}")
                    
                    # Signal that config is updated
                    self.config_updated.set()
                    
            except Exception as e:
                logger.error(f"Error in processing thread: {e}", exc_info=True)
        
        logger.info("Processing thread stopped")

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
        
        with open('inference_metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)

    def run(self):
        try:
            # Connect MQTT client
            self.client.connect(self.broker, self.port)
            self.client.loop_start()
            
            # Start processing thread
            processing_thread = threading.Thread(target=self.process_updates_thread)
            processing_thread.daemon = True
            processing_thread.start()
            
            # Main thread just waits
            while self.running:
                try:
                    time.sleep(1.0)
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
    subscriber = MultithreadedSubscriber("/Users/aryansethi20/Downloads/fyp/Runs/9/final_model-3rdSize.pth")
    subscriber.run()

if __name__ == "__main__":
    main()