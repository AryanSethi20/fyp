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


class InferencePublisher:
    def __init__(self):
        # MQTT Configuration
        self.client_id = f'publisher-{random.randint(0, 1000)}'
        self.broker = "192.168.1.3"
        self.port = 1883

        # Topics
        self.status_topic = "artc1/status_update"
        self.action_topic = "artc1/action"
        self.ack_topic = "artc1/ack"

        # State variables
        self.mu = 2.0  # Default service rate
        self.lambda_rate = 1.0  # Default arrival rate
        self.policy = "CU"  # Default policy
        self.generation_time = None
        self.update_count = 0

        # Event flags
        self.ack_received = Event()
        self.running = True

        # MQTT setup
        self.client = mqtt_client.Client(self.client_id)
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
        status_update = {
            "generation_time": self.generation_time,
            "update_count": self.update_count,
            "mu": self.mu,
            "lambda": self.lambda_rate,
            "policy": self.policy
        }

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