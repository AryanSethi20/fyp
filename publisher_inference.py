import random
import time
import json
import logging
from threading import Event
from paho.mqtt import client as mqtt_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("publisher")

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

class InferencePublisher:
    def __init__(self):
        # MQTT Configuration
        self.client_id = f'publisher-{random.randint(0, 1000)}'
        self.broker = "192.168.1.3"
        self.port = 1883

        # Topics
        self.status_topic = "artc1/status_update"
        self.action_topic = "artc1/new_actions"
        self.ack_topic = "artc1/ack"

        # State variables
        self.mu = 1.0  # Default service rate
        self.lambda_rate = 1.0  # Default arrival rate
        self.policy = "CU"  # Default policy
        self.generation_time = None
        self.update_count = 0

        # Event flags
        self.ack_received = Event()
        self.running = True

        # MQTT setup
        self.client = mqtt_client.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT Broker!")
            self.client.subscribe(self.action_topic)
            self.client.subscribe(self.ack_topic)
        else:
            logger.error(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            if msg.topic == self.action_topic:
                action = json.loads(msg.payload)
                logger.info(f"Received action update: {action}")

                # Update local parameters
                self.policy = action["policy"]
                self.mu = action["mu"]
                self.lambda_rate = action["lambda_rate"]

            elif msg.topic == self.ack_topic:
                logger.debug("Received ACK")
                self.ack_received.set()

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def publish_status(self):
        """Publish a status update"""
        self.generation_time = time.time()
        # status_update = {
        #     "generation_time": self.generation_time,
        #     "update_count": self.update_count,
        #     "mu": self.mu,
        #     "lambda": self.lambda_rate,
        #     "policy": self.policy
        # }
        status_update = get_status_update(
            size_level=1,
            policy=self.policy,
            mu=self.mu,
            total_updates=self.update_count
        )

        topic = f"{self.status_topic}/{self.policy}"
        self.client.publish(topic, json.dumps(status_update))
        logger.info(f"Published update {self.update_count} with policy {self.policy}")
        self.update_count += 1

    def run(self):
        try:
            self.client.connect(self.broker, self.port)
            self.client.loop_start()

            while self.running:
                self.publish_status()

                # If using ZW policy, wait for ACK
                if self.policy == "ZW":
                    if not self.ack_received.wait(timeout=5.0):
                        logger.warning("No ACK received within timeout")
                    self.ack_received.clear()

                # Wait according to current lambda rate
                sleep_time = 1 / self.lambda_rate if self.lambda_rate > 0 else 1
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Shutting down publisher...")
        except Exception as e:
            logger.error(f"Error in publisher: {e}", exc_info=True)
        finally:
            self.running = False
            self.client.loop_stop()
            self.client.disconnect()


if __name__ == "__main__":
    publisher = InferencePublisher()
    publisher.run()