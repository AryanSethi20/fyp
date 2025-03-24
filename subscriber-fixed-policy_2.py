import time
import random
import json
import logging
import numpy as np
import argparse
from paho.mqtt import client as mqtt_client
import threading
from queue import Queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("subscriber-fixed-policy")

class FixedPolicySubscriber:
    def __init__(self, policy, mu, lambda_rate):
        # MQTT Configuration
        self.client_id = f'subscriber-fixed-{random.randint(0, 1000)}'
        self.broker = "192.168.0.123"
        self.port = 1883

        # Topics
        self.status_update_topic = "artc1/status_update"
        self.ack_topic = "artc1/ack"
        self.new_actions_topic = "artc1/new_actions"

        # Fixed policy parameters
        self.policy = policy  # "CU" or "ZW"
        self.mu = mu
        self.lambda_rate = lambda_rate

        # MQTT setup
        self.client = mqtt_client.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

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
        
        # Running flag and experiment control
        self.running = True
        self.start_time = None
        self.batch_id = 0

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT Broker!")
            self.client.subscribe([
                (f"{self.status_update_topic}/CU", 0),
                (f"{self.status_update_topic}/ZW", 0),
            ])
            
            # Send initial configuration to publisher
            self.start_time = time.time()
            self.send_configuration()
        else:
            logger.error(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        """MQTT message handler - runs in MQTT thread"""
        try:
            current_time = time.time()
            status_update = json.loads(msg.payload)
            
            # Send ACK immediately for ZW policy to avoid timeouts
            if status_update["policy"] == "ZW":
                self.client.publish(self.ack_topic, "ACK")
                
            # Calculate actual network delay
            network_delay = current_time - status_update["generation_time"]
            # created_time = status_update["generation_time"]
            # logger.info(f"Created Time: {created_time}, Received Time: {current_time}, AOI: {network_delay}")

            # Calculate theoretical peak age of information
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
            
            # Print stats periodically
            if len(self.metrics['aoi_values']) % 100 == 0:
                avg_aoi = np.mean(self.metrics['theoretical_aoi_values'][-100:])
                max_paoi = np.max(self.metrics['theoretical_aoi_values'][-100:])
                logger.info(f"Last 100 messages - Avg AoI: {avg_aoi:.4f}s, PAoI: {max_paoi:.4f}s, Policy: {self.policy}")
                
            # No time limit check - we'll run until manually terminated
                
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

    def send_configuration(self):
        """Send fixed policy configuration to publisher"""
        action = {
            "policy": self.policy,
            "mu": self.mu,
            "lambda_rate": self.lambda_rate,
            "batch_id": self.batch_id
        }
        self.client.publish(self.new_actions_topic, json.dumps(action))
        logger.info(f"Sent fixed configuration: policy={self.policy}, mu={self.mu:.2f}, lambda={self.lambda_rate:.2f}")

    def save_metrics(self):
        """Save metrics to a file with policy/mu/lambda in the filename"""
        try:
            metrics_dict = {
                'aoi_values': self.metrics['aoi_values'],
                'theoretical_aoi_values': self.metrics['theoretical_aoi_values'],
                'service_times': self.metrics['service_times'],
                'policies': self.metrics['policies'],
                'mu_values': self.metrics['mu_values'],
                'lambda_values': self.metrics['lambda_values'],
                'timestamps': self.metrics['timestamps'],
                'batch_ids': self.metrics['batch_ids'],
                'summary': {
                    'avg_aoi': np.mean(self.metrics['aoi_values']) if len(self.metrics['aoi_values']) > 0 else 0,
                    'avg_theoretical_aoi': np.mean(self.metrics['theoretical_aoi_values']) if len(self.metrics['theoretical_aoi_values']) > 0 else 0,
                    'max_aoi': np.max(self.metrics['aoi_values']) if len(self.metrics['aoi_values']) > 0 else 0,
                    'min_aoi': np.min(self.metrics['aoi_values']) if len(self.metrics['aoi_values']) > 0 else 0,
                    'count': len(self.metrics['aoi_values']),
                    'policy': self.policy,
                    'mu': self.mu,
                    'lambda': self.lambda_rate,
                    'ratio': self.lambda_rate / self.mu
                }
            }
            
            # Create filename with parameters
            filename = f'metrics_{self.policy}_mu{self.mu}_lambda{self.lambda_rate}.json'
            
            # Log that we're saving the file
            logger.info(f"Saving metrics to {filename}")
            
            with open(filename, 'w') as f:
                json.dump(metrics_dict, f, indent=4)
                
            logger.info(f"Successfully saved metrics to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}", exc_info=True)

    def run(self):
        try:
            # Connect MQTT client
            self.client.connect(self.broker, self.port)
            self.client.loop_start()
            
            # Main thread just waits for manual termination
            while self.running:
                try:
                    time.sleep(1.0)
                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
                    self.running = False
                    break
                
        except KeyboardInterrupt:
            logger.info("Shutting down subscriber...")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
        finally:
            # Always save metrics when terminating
            self.running = False
            logger.info("Saving metrics before shutdown...")
            self.save_metrics()
            
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting MQTT client: {e}")
                
            logger.info("Subscriber shutdown complete")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run fixed policy experiment')
    parser.add_argument('--policy', type=str, choices=['CU', 'ZW'], required=True,
                        help='Policy to use (CU or ZW)')
    parser.add_argument('--mu', type=float, required=True,
                        help='Service rate (mu)')
    parser.add_argument('--lambda_rate', type=float, required=True,
                        help='Arrival rate (lambda)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting fixed policy experiment: policy={args.policy}, mu={args.mu}, lambda={args.lambda_rate}")
    logger.info("Press Ctrl+C to stop the experiment and save metrics")
    subscriber = FixedPolicySubscriber(args.policy, args.mu, args.lambda_rate)
    subscriber.run()

if __name__ == "__main__":
    main()