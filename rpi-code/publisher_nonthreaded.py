import random
import time
import json
import logging
from paho.mqtt import client as mqtt_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("publisher")

def get_status_update(size_level, policy, mu, total_updates, batch_id):
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
        "total_updates": total_updates,
        "batch_id": batch_id
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
    
    # Note: Code for size levels 4-10 is omitted for brevity but would be
    # identical to the original function. Include it if needed.

    return status_update

class NonThreadedPublisher:
    def __init__(self):
        # MQTT Configuration
        self.client_id = f'publisher-{random.randint(0, 1000)}'
        self.broker = "192.168.0.123"
        self.port = 1883

        # Topics
        self.status_topic = "artc1/status_update"
        self.action_topic = "artc1/new_actions"
        self.ack_topic = "artc1/ack"

        # State variables
        self.config = {
            "policy": "CU",
            "mu": 5.0,
            "lambda_rate": 1.5,
            "batch_id": 0
        }
        self.update_count = 0
        self.batch_update_count = 0  # Count updates within current batch
        self.window_size = 10  # Match subscriber's window size
        self.ack_received = False
        self.last_ack_time = 0
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

                # Update configuration
                self.config = {
                    "policy": action["policy"],
                    "mu": action["mu"],
                    "lambda_rate": action["lambda_rate"],
                    "batch_id": action.get("batch_id", self.config["batch_id"] + 1)
                }

            elif msg.topic == self.ack_topic:
                # Set ack received flag and timestamp
                self.ack_received = True
                self.last_ack_time = time.time()

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def publish_update(self):
        """Publish a single status update"""
        # Get current config
        policy = self.config["policy"]
        mu = self.config["mu"]
        lambda_rate = self.config["lambda_rate"]
        batch_id = self.config["batch_id"]
        
        # Create status update
        status_update = get_status_update(
            size_level=1,  # Use small packets by default
            policy=policy,
            mu=mu,
            total_updates=self.update_count,
            batch_id=batch_id
        )
        
        # Publish status update
        topic = f"{self.status_topic}/{policy}"
        self.client.publish(topic, json.dumps(status_update))
        logger.info(f"Published update {self.update_count} with policy {policy}, batch {batch_id}")
        self.update_count += 1
        self.batch_update_count += 1
        
        # If using ZW policy, check for ACK
        if policy == "ZW":
            # Reset ack flag
            self.ack_received = False
            
            # Wait for ACK with timeout
            wait_start = time.time()
            while not self.ack_received and time.time() - wait_start < 2.0:
                time.sleep(0.01)  # Short sleep to prevent busy waiting
            
            # If no ACK received within timeout, fall back to CU
            if not self.ack_received:
                logger.warning("No ACK received within timeout, falling back to CU")
                if self.config["policy"] == "ZW":  # Only change if still ZW
                    self.config["policy"] = "CU"
        
        # Check if we've completed a batch
        if self.batch_update_count >= self.window_size:
            logger.info(f"Completed batch of {self.window_size} updates. Waiting 3 seconds for subscriber processing...")
            time.sleep(3)  # Wait for subscriber to process batch and send new state
            self.batch_update_count = 0  # Reset batch counter
            logger.info("Resuming updates after wait period")

    def run(self):
        try:
            # Connect MQTT client
            self.client.connect(self.broker, self.port)
            self.client.loop_start()
            
            # Main loop
            while self.running:
                try:
                    # Publish an update
                    self.publish_update()
                    
                    # Sleep according to lambda_rate
                    sleep_time = 1 / self.config["lambda_rate"]
                    time.sleep(sleep_time)
                    
                except KeyboardInterrupt:
                    self.running = False
                    break
                
        except KeyboardInterrupt:
            logger.info("Shutting down publisher...")
        except Exception as e:
            logger.error(f"Error in main thread: {e}", exc_info=True)
        finally:
            self.running = False
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Publisher shutdown complete")

if __name__ == "__main__":
    publisher = NonThreadedPublisher()
    publisher.run()