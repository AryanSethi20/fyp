import random
import time
import json
from paho.mqtt import client as mqtt_client
from collections import deque
import logging
import numpy as np


class MQTTEnvironment:
    def __init__(self):
        self.broker = "192.168.0.105"
        self.port = 1883
        self.status_update_topic = "artc1/status_update"
        self.policy_topic = "artc1/policy"
        self.ack_topic = "artc1/ack"  # ACK topic
        self.send_status_update_topic = "artc1/send_status_update"
        # self.window_size = 10  # FIXME: For Simulation (Presentation)
        self.window_size = 10
        self.paoi_window = deque(maxlen=self.window_size)

        # ACK size configuration with predefined payloads
        self.ack_sizes = {
            "tiny": 16,  # 16 bytes
            "small": 64,  # 64 bytes
            "medium": 256,  # 256 bytes
            "large": 1024,  # 1KB
            "xlarge": 4096,  # 4KB
        }

        # Predefined fixed ACK messages for each size
        self.ack_payloads = {
            "tiny": '{"ack":"ACK","pad":"X"}',
            "small": '{"ack":"ACK","ts":0,"id":"fixed-small-ack","padding":"'
            + "X" * 32
            + '"}',
            "medium": '{"ack":"ACK","timestamp":0,"id":"fixed-medium-ack","size":"medium","padding":"'
            + "X" * 185
            + '"}',
            "halfkb": '{"ack":"ACK","timestamp":0,"id":"fixed-halfkb-ack","size":"halfkb","padding":"'
            + "X" * 441
            + '"}',
            "large": '{"ack":"ACK","timestamp":0,"id":"fixed-large-ack","size":"large","padding":"'
            + "X" * 953
            + '"}',
            "xlarge": '{"ack":"ACK","timestamp":0,"id":"fixed-xlarge-ack","size":"xlarge","padding":"'
            + "X" * 4025
            + '"}',
        }

        self.ack_size = "halfkb"


class EnhancedMQTTSubscriber:
    def __init__(self):
        self.env = MQTTEnvironment()
        self.client_id = f"subscribe-{random.randint(0, 1000)}"
        self.client = mqtt_client.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.logger = logging.getLogger("subscriber")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info("Subscriber connected to MQTT Broker!")
            self.client.subscribe(
                [
                    (f"{self.env.status_update_topic}/CU", 0),
                    (f"{self.env.status_update_topic}/ZW", 0),
                ]
            )
        else:
            self.logger.error(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        self.logger.info(f"Received message on topic: {msg.topic}")
        try:
            # self.logger.info(f"Received message: {msg.payload.decode()}")
            status_update = json.loads(msg.payload)
            mu = status_update["service_rate"]
            service_time = np.random.exponential(scale=1 / mu)
            paoi = time.time() - status_update["generation_time"] + service_time

            # Send ACK for ZW policy
            if status_update["policy"] == "ZW":
                self.logger.info("Sending ACK for ZW policy")
                # self.client.publish(self.env.ack_topic, "ACK")
                self.send_fixed_size_ack()

            # Append new PAoI to the window
            self.env.paoi_window.append(paoi)
            if status_update["total_updates"] % self.env.window_size == 0:
                self.logger.info("Sending status updates to RL Agent for training")
                self.client.publish(
                    self.env.send_status_update_topic, json.dumps(self.observe_paoi())
                )
                self.reset_paoi_window()
            # print(f"Updated PAoI window: {list(self.env.paoi_window)}")

        except Exception as e:
            print(f"Error processing message: {e}")

    def send_fixed_size_ack(self):
        """Send ACK with a predefined fixed payload for the configured size"""
        try:
            # Get the predefined payload for the current ACK size
            payload = self.env.ack_payloads[self.env.ack_size]

            # Add timestamp to the payload if it's not tiny
            if self.env.ack_size != "tiny":
                # Replace the 0 timestamp with current time
                # This simple string replacement avoids having to re-parse and re-serialize JSON
                payload = payload.replace('"ts":0', f'"ts":{time.time()}')
                payload = payload.replace('"timestamp":0', f'"timestamp":{time.time()}')

            # Send the predefined ACK
            result = self.client.publish(self.env.ack_topic, payload)

            # Check if publish was successful
            if result.rc == mqtt_client.MQTT_ERR_SUCCESS:
                actual_size = len(payload.encode())
                self.logger.info(
                    f"ACK sent successfully: {self.env.ack_size} size, {actual_size} bytes"
                )
            else:
                self.logger.error(f"Failed to publish ACK, result code: {result.rc}")

        except Exception as e:
            self.logger.error(f"Error sending ACK: {e}")
            # Fallback to simple ACK in case of any error
            self.client.publish(self.env.ack_topic, "ACK")

    def observe_paoi(self):
        return list(self.env.paoi_window)

    def reset_paoi_window(self):
        """
        Clears the PAoI window for collecting a new set of updates.
        """
        self.env.paoi_window.clear()
        print("Cleared PAoI window for new updates.")

    def get_average_aoi(self):
        """Get the average AoI from the current window"""
        if len(self.env.paoi_window) == 0:
            return 0.0
        return np.mean(self.env.paoi_window)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    subscriber = EnhancedMQTTSubscriber()
    subscriber.client.connect(subscriber.env.broker, subscriber.env.port)
    subscriber.client.loop_forever()
