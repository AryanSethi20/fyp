import random
import time
import json
from paho.mqtt import client as mqtt_client
from threading import Event
import logging
import datetime
import torch
import torch.nn as nn
import itertools
import numpy as np

dateFormat = "%Y-%m-%d"
timeFormat = "%H-%M-%S.%f"


class PublisherEnvironment:
    def __init__(self):
        self.broker = "192.168.0.105"
        self.port = 1883
        self.status_update_topic = "artc1/status_update"
        self.policy_topic = "artc1/policy"
        self.ack_topic = "artc1/ack"
        self.new_actions_topic = "artc1/new_actions"

        # Initialize with None to indicate waiting for first values
        self.lambda_rate = None
        self.mu = None
        self.total_updates = 0
        self.ZW_policy_flag = Event()
        self.ack_flag = Event()
        # self.window_size = 10
        self.window_size = 5
        self.new_action = Event()
        self.initialized = Event()  # New flag for initialization

def get_status_update(size_level, policy, mu, total_updates):
    """
    Returns a status update packet of different sizes based on size_level (1-10)
    size_level 1: ~83 bytes
    size_level 2: ~185 bytes
    size_level 3: ~283 bytes
    size_level 4: ~399 bytes
    size_level 5: ~590 bytes
    size_level 6: ~820 bytes
    size_level 7: ~1100 bytes
    size_level 8: ~1450 bytes
    size_level 9: ~1800 bytes
    size_level 10: ~2200 bytes
    """
    # Base packet (Size Level 1: ~83 bytes)
    status_update = {
        "generation_time": time.time(),
        "policy": policy,
        "service_rate": mu,
        "total_updates": total_updates
    }

    if size_level >= 2:  # Add metadata (~185 bytes)
        status_update["metadata"] = {
            "sequence_number": total_updates,
            "priority": "high",
            "timestamp_ms": int(time.time() * 1000),
            "status": "active"
        }

    if size_level >= 3:  # Add system metrics (~283 bytes)
        status_update["system_metrics"] = {
            "cpu_usage": random.uniform(0, 100),
            "memory_usage": random.uniform(0, 100),
            "network_latency": random.uniform(1, 50),
            "queue_length": random.randint(0, 1000)
        }

    if size_level >= 4:  # Add environmental data (~399 bytes)
        status_update["environmental_data"] = {
            "temperature": random.uniform(20, 30),
            "humidity": random.uniform(40, 60),
            "pressure": random.uniform(980, 1020),
            "location": {
                "x": random.uniform(0, 100),
                "y": random.uniform(0, 100),
                "z": random.uniform(0, 100)
            }
        }

    if size_level >= 5:  # Add historical metrics (~590 bytes)
        status_update["historical_metrics"] = [
            {
                "timestamp": time.time() - i,
                "value": random.uniform(0, 100)
            } for i in range(4)  # Last 4 measurements
        ]

    if size_level >= 6:  # Add network statistics (~820 bytes)
        status_update["network_statistics"] = {
            "bandwidth": {
                "incoming": random.uniform(0, 1000),
                "outgoing": random.uniform(0, 1000),
                "peak": random.uniform(500, 2000),
                "average": random.uniform(100, 1000)
            },
            "packets": {
                "sent": random.randint(1000, 10000),
                "received": random.randint(1000, 10000),
                "dropped": random.randint(0, 100),
                "errors": random.randint(0, 50)
            },
            "connections": {
                "active": random.randint(1, 100),
                "idle": random.randint(0, 50),
                "failed": random.randint(0, 20)
            }
        }

    if size_level >= 7:  # Add detailed diagnostics (~1100 bytes)
        status_update["diagnostics"] = {
            "system_health": {
                "disk_usage": {
                    "total": random.randint(100000, 1000000),
                    "used": random.randint(10000, 100000),
                    "free": random.randint(50000, 500000),
                    "read_speed": random.uniform(50, 200),
                    "write_speed": random.uniform(30, 150)
                },
                "memory_details": {
                    "total": random.randint(8000, 32000),
                    "available": random.randint(4000, 16000),
                    "cached": random.randint(1000, 8000),
                    "swap_used": random.randint(0, 1000)
                }
            },
            "process_info": [
                {
                    "pid": random.randint(1, 10000),
                    "cpu_percent": random.uniform(0, 100),
                    "memory_percent": random.uniform(0, 100),
                    "status": random.choice(["running", "sleeping", "waiting"])
                } for _ in range(3)
            ]
        }

    if size_level >= 8:  # Add security metrics (~1450 bytes)
        status_update["security"] = {
            "threats_detected": {
                "high": random.randint(0, 5),
                "medium": random.randint(0, 10),
                "low": random.randint(0, 20),
                "details": [
                    {
                        "type": random.choice(["malware", "intrusion", "ddos", "spam"]),
                        "severity": random.choice(["high", "medium", "low"]),
                        "timestamp": time.time() - random.randint(0, 3600),
                        "status": random.choice(["blocked", "quarantined", "investigating"])
                    } for _ in range(3)
                ]
            },
            "authentication": {
                "successful_logins": random.randint(10, 100),
                "failed_attempts": random.randint(0, 20),
                "active_sessions": random.randint(1, 10),
                "last_audit": time.time() - random.randint(0, 86400)
            },
            "encryption_status": {
                "algorithm": "AES-256",
                "key_rotation": time.time() - random.randint(0, 86400),
                "certificates": ["primary", "backup", "recovery"]
            }
        }

    if size_level >= 9:  # Add performance analytics (~1800 bytes)
        status_update["performance_analytics"] = {
            "response_times": [
                {
                    "endpoint": f"/api/v1/endpoint{i}",
                    "min_ms": random.uniform(1, 10),
                    "max_ms": random.uniform(50, 200),
                    "avg_ms": random.uniform(10, 50),
                    "percentiles": {
                        "p50": random.uniform(10, 30),
                        "p90": random.uniform(30, 60),
                        "p99": random.uniform(60, 100)
                    }
                } for i in range(4)
            ],
            "resource_utilization": {
                "threads": {
                    "active": random.randint(10, 100),
                    "idle": random.randint(0, 20),
                    "blocked": random.randint(0, 5),
                    "dead_locked": random.randint(0, 2)
                },
                "garbage_collection": {
                    "collections": random.randint(10, 100),
                    "total_time_ms": random.randint(100, 1000),
                    "freed_memory": random.randint(1000, 10000)
                }
            }
        }

    if size_level >= 10:  # Add system configuration (~2200 bytes)
        status_update["system_configuration"] = {
            "hardware": {
                "cpu": {
                    "model": "Intel Xeon E5-2680",
                    "cores": random.randint(8, 32),
                    "frequency_ghz": random.uniform(2.0, 4.0),
                    "cache_sizes": {
                        "l1": 32768,
                        "l2": 262144,
                        "l3": 20971520
                    }
                },
                "memory_modules": [
                    {
                        "slot": f"DIMM_{i}",
                        "size_gb": 16,
                        "type": "DDR4",
                        "speed": 3200,
                        "manufacturer": "Crucial"
                    } for i in range(4)
                ]
            },
            "software": {
                "os": {
                    "name": "Ubuntu Server",
                    "version": "20.04 LTS",
                    "kernel": "5.4.0-42-generic",
                    "installed_packages": random.randint(1000, 2000)
                },
                "services": [
                    {
                        "name": f"service_{i}",
                        "version": f"1.{random.randint(0, 9)}.{random.randint(0, 9)}",
                        "status": random.choice(["running", "stopped", "restarting"]),
                        "port": random.randint(1024, 65535),
                        "dependencies": [f"dep_{j}" for j in range(random.randint(1, 3))]
                    } for i in range(3)
                ]
            }
        }

    return status_update


class EnhancedMQTTPublisher:
    def __init__(self):
        self.env = PublisherEnvironment()
        self.client_id = f'publish-{random.randint(0, 1000)}'
        self.client = mqtt_client.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.last_update_time = time.time()
        self.logger = logging.getLogger('publisher')

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info("Publisher connected to MQTT Broker!")
            self.client.subscribe(self.env.policy_topic)
            self.client.subscribe(self.env.ack_topic)
            self.client.subscribe(self.env.new_actions_topic)
            # Publish a ready message to notify environment
            # self.client.publish("artc1/publisher_ready", "ready")
            self.logger.info("Published ready message")
        else:
            self.logger.error(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            self.logger.info(datetime.datetime.now().strftime(dateFormat + "|" + timeFormat) +
                             ": Received " + msg.payload.decode() + " from " + msg.topic)
            if msg.topic == self.env.new_actions_topic:
                actions = json.loads(msg.payload)
                self.env.lambda_rate = actions["lambda_rate"]
                self.env.mu = actions["mu"]
                if actions["policy"] == "ZW":
                    self.env.ZW_policy_flag.set()
                else:
                    self.env.ZW_policy_flag.clear()
                self.env.new_action.set()
                # Set initialized flag on first message
                if not self.env.initialized.is_set():
                    self.env.initialized.set()
                    self.logger.info("Received initial parameters from environment")
            elif msg.topic == self.env.ack_topic:
                self.env.ack_flag.set()

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    # Usage in your publish_status_update method:
    def publish_status_update(self, policy, size_level=2):
        if self.env.mu is None or self.env.lambda_rate is None:
            self.logger.warning("Cannot publish: waiting for initial parameters")
            return

        status_update = get_status_update(
            size_level=size_level,
            policy=policy,
            mu=self.env.mu,
            total_updates=self.env.total_updates + 1
        )
    
        topic = f"{self.env.status_update_topic}/{policy}"
        result = self.client.publish(topic, json.dumps(status_update))
    
        if result[0] == 0:
            self.env.total_updates += 1
            self.logger.info(f"Published status update for policy: {policy}, Size Level: {size_level}, Total Updates: {self.env.total_updates}")
        else:
            self.logger.info("Failed to publish status update")

        if self.env.total_updates % self.env.window_size == 0:
            self.env.new_action.wait(60)
            self.env.new_action.clear()
        """
        def publish_status_update(self, policy):
        if self.env.mu is None or self.env.lambda_rate is None:
            self.logger.warning("Cannot publish: waiting for initial parameters")
            return

        status_update = {
            "generation_time": time.time(),
            "policy": policy,
            "service_rate": self.env.mu,
            "total_updates": self.env.total_updates + 1
        }
        topic = f"{self.env.status_update_topic}/{policy}"

        result = self.client.publish(topic, json.dumps(status_update))
        if result[0] == 0:
            self.env.total_updates += 1
            self.logger.info(f"Published status update for policy: {policy}, Total Updates: {self.env.total_updates}")
        else:
            self.logger.info("Failed to publish status update")

        if self.env.total_updates % self.env.window_size == 0:
            self.env.new_action.wait(60)
            self.env.new_action.clear()
        """

    def run(self):
        try:
            self.client.connect(self.env.broker, self.env.port)
            self.client.loop_start()

            # Wait for initial parameters
            self.logger.info("Waiting for initial parameters from environment...")
            self.env.initialized.wait()
            self.logger.info(f"Starting with parameters: mu={self.env.mu}, lambda={self.env.lambda_rate}")

            while True:
                if self.env.ZW_policy_flag.is_set():
                    self.publish_status_update("ZW")
                    self.env.ack_flag.wait(15)
                    self.env.ack_flag.clear()
                else:
                    self.publish_status_update("CU")
                    self.last_update_time = time.time()
                time.sleep(1 / self.env.lambda_rate)

        except KeyboardInterrupt:
            self.logger.warning("\nDisconnecting publisher")
            self.client.loop_stop()
            self.client.disconnect()


    def publish_action_changes(self, policy, mu, lambda_rate):
        self.logger.info(f"Publishing action changes for {policy}, mu={mu}, lambda_rate={lambda_rate}")
        self.env.lambda_rate = lambda_rate
        self.env.mu = mu
        self.env.ZW_policy_flag.set() if policy == "ZW" else self.env.ZW_policy_flag.clear()
        # self.client.publish(self.env.policy_topic, json.dumps(policy_message))
        self.logger.info(f"Published policy change: {policy}")
        self.env.new_action.set()

class ModelInference:
    def __init__(self, model_path, state_size, policies, mu_values, lambda_values):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create action space first
        self.action_space = list(itertools.product(policies, mu_values, lambda_values))
        # Then build and load model
        self.model = self._build_model(state_size).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path))
            print(f"Model loaded successfully from {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file {model_path} not found. Please ensure you have trained the model first.")
        self.model.eval()

    def _build_model(self, state_size):
        return nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.action_space)),
        )

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_values = self.model(state)
            action_idx = torch.argmax(action_values).item()
            return self.action_space[action_idx]

class AutonomousPublisher:
    def __init__(self, model_path):
        self.env = PublisherEnvironment()
        self.client_id = f'publish-{random.randint(0, 1000)}'
        self.client = mqtt_client.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.last_update_time = time.time()
        self.logger = logging.getLogger('publisher')

        # Load the trained model
        self.model = self.load_model(model_path)
        self.current_state = np.array([0, 0, 3.0, 1.2])  # Initial state

        # Initialize metrics tracking
        self.metrics = {
            'aoi_values': [],
            'policies_used': [],
            'mu_values': [],
            'lambda_values': [],
            'timestamps': []
        }

    def load_model(self, model_path):
        try:
            policies = [0, 1]
            mu_values = [1.0, 1.5, 2.0, 2.5, 3.0] # FIXME: Change this to account for all the mu values possible
            lambda_values = [x * 0.4 for x in mu_values] # FIXME: Change this to account for all the lambda values possible
            state_size = 4

            model = ModelInference(model_path, state_size, policies, mu_values, lambda_values)
            self.logger.info("Model loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info("Publisher connected to MQTT Broker!")
            self.client.subscribe(self.env.ack_topic)
        else:
            self.logger.error(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        if msg.topic == self.env.ack_topic:
            self.env.ack_flag.set()

    def update_state(self, avg_aoi, policy, mu, lambda_rate):
        self.current_state = np.array([avg_aoi, policy, mu, lambda_rate])

    def get_next_action(self):
        return self.model.get_action(self.current_state)

    def run_simulation(self, simulation_time=300):  # Run for 5 minutes by default
        try:
            self.client.connect(self.env.broker, self.env.port)
            self.client.loop_start()

            start_time = time.time()
            current_window = []

            while (time.time() - start_time) < simulation_time:
                # Get action from model
                policy, mu, lambda_rate = self.get_next_action()
                policy_name = "ZW" if policy == 1 else "CU"

                # Update environment parameters
                self.env.mu = mu
                self.env.lambda_rate = lambda_rate
                self.env.ZW_policy_flag.set() if policy == 1 else self.env.ZW_policy_flag.clear()

                # Publish update
                status_update = {
                    "generation_time": time.time(),
                    "policy": policy_name,
                    "service_rate": mu,
                    "total_updates": self.env.total_updates + 1
                }

                topic = f"{self.env.status_update_topic}/{policy_name}"
                result = self.client.publish(topic, json.dumps(status_update))

                if result[0] == 0:
                    self.env.total_updates += 1
                    current_time = time.time() - start_time

                    # Wait for ACK if ZW policy
                    if policy_name == "ZW":
                        self.env.ack_flag.wait(15)
                        self.env.ack_flag.clear()

                    # Record metrics
                    self.metrics['timestamps'].append(current_time)
                    self.metrics['policies_used'].append(policy_name)
                    self.metrics['mu_values'].append(mu)
                    self.metrics['lambda_values'].append(lambda_rate)

                    self.logger.info(
                        f"Update {self.env.total_updates}: Policy={policy_name}, μ={mu:.2f}, λ={lambda_rate:.2f}")

                # Sleep according to lambda rate
                time.sleep(lambda_rate)

                # Check if window is complete
                if self.env.total_updates % self.env.window_size == 0:
                    avg_aoi = np.mean(current_window) if current_window else 0
                    self.update_state(avg_aoi, policy, mu, lambda_rate)
                    current_window = []

            # Save final metrics
            self.save_metrics()

        except KeyboardInterrupt:
            self.logger.warning("\nStopping simulation...")
        finally:
            self.client.loop_stop()
            self.client.disconnect()
            self.logger.info("Simulation completed")

    def save_metrics(self, filename="simulation_metrics.json"):
        metrics = {
            'total_updates': self.env.total_updates,
            'run_time': time.time() - self.metrics['timestamps'][0],
            'policy_distribution': {
                'CU': self.metrics['policies_used'].count('CU'),
                'ZW': self.metrics['policies_used'].count('ZW')
            },
            'average_mu': np.mean(self.metrics['mu_values']),
            'average_lambda': np.mean(self.metrics['lambda_values']),
            'timestamps': self.metrics['timestamps'],
            'policies': self.metrics['policies_used'],
            'mu_history': self.metrics['mu_values'],
            'lambda_history': self.metrics['lambda_values']
        }

        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {filename}")


# if __name__ == "__main__":
#    logging.basicConfig(level=logging.DEBUG)
#    publisher = AutonomousPublisher("best_model.pth")
#    publisher.run_simulation(simulation_time=300)  # Run for 5 minutes

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    publisher = EnhancedMQTTPublisher()
    publisher.run()
