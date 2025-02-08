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
        self.window_size = 5
        self.paoi_window = deque(maxlen=self.window_size)

class EnhancedMQTTSubscriber:
    def __init__(self):
        self.env = MQTTEnvironment()
        self.client_id = f'subscribe-{random.randint(0, 1000)}'
        self.client = mqtt_client.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.logger = logging.getLogger("subscriber")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info("Subscriber connected to MQTT Broker!")
            self.client.subscribe([
                (f"{self.env.status_update_topic}/CU", 0),
                (f"{self.env.status_update_topic}/ZW", 0),
            ])
        else:
            self.logger.error(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        self.logger.info(f"Received message on topic: {msg.topic}")
        try:
            self.logger.info(f"Received message: {msg.payload.decode()}")
            status_update = json.loads(msg.payload)
            mu = status_update["service_rate"]
            service_time = np.random.exponential(scale= 1 / mu)
            paoi = time.time() - status_update["generation_time"] + service_time

            # Send ACK for ZW policy
            if status_update["policy"] == "ZW":
                self.logger.info("Sending ACK for ZW policy")
                self.client.publish(self.env.ack_topic, "ACK")

            # Append new PAoI to the window
            self.env.paoi_window.append(paoi)
            if status_update["total_updates"] % self.env.window_size == 0:
                self.logger.info("Sending status updates to RL Agent for training")
                self.client.publish(self.env.send_status_update_topic, json.dumps(self.observe_paoi()))
                self.reset_paoi_window()
            # print(f"Updated PAoI window: {list(self.env.paoi_window)}")

        except Exception as e:
            print(f"Error processing message: {e}")

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