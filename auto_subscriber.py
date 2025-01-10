import random
import time
import json
from paho.mqtt import client as mqtt_client
from collections import deque
import logging
import numpy as np
from datetime import datetime


class MQTTEnvironment:
    def __init__(self):
        self.broker = "broker.emqx.io"
        self.port = 1883
        self.status_update_topic = "artc1/status_update"
        self.ack_topic = "artc1/ack"
        self.window_size = 10  # For simulation
        self.paoi_window = deque(maxlen=self.window_size)


class AutonomousSubscriber:
    def __init__(self):
        self.env = MQTTEnvironment()
        self.client_id = f'subscribe-{random.randint(0, 1000)}'
        self.client = mqtt_client.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.logger = logging.getLogger("subscriber")

        # Metrics tracking
        self.metrics = {
            'cu_aoi_values': [],
            'zw_aoi_values': [],
            'timestamps': [],
            'service_times': [],
            'total_updates': 0,
            'start_time': None
        }

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info("Subscriber connected to MQTT Broker!")
            self.client.subscribe([
                (f"{self.env.status_update_topic}/CU", 0),
                (f"{self.env.status_update_topic}/ZW", 0),
            ])
            self.metrics['start_time'] = time.time()
        else:
            self.logger.error(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            status_update = json.loads(msg.payload)
            self.metrics['total_updates'] += 1

            # Calculate AoI
            mu = status_update["service_rate"]
            service_time = np.random.exponential(scale=1 / mu)
            paoi = time.time() - status_update["generation_time"] + service_time

            # Update metrics
            current_time = time.time() - self.metrics['start_time']
            self.metrics['timestamps'].append(current_time)
            self.metrics['service_times'].append(service_time)

            if status_update["policy"] == "ZW":
                self.metrics['zw_aoi_values'].append(paoi)
                self.logger.info(f"ZW Update - AoI: {paoi:.3f}, Service Time: {service_time:.3f}")
                self.client.publish(self.env.ack_topic, "ACK")
            else:
                self.metrics['cu_aoi_values'].append(paoi)
                self.logger.info(f"CU Update - AoI: {paoi:.3f}, Service Time: {service_time:.3f}")

            # Store in window
            self.env.paoi_window.append(paoi)

            # Regular metrics saving
            if self.metrics['total_updates'] % 50 == 0:  # Save every 50 updates
                self.save_metrics()

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def get_average_aoi(self):
        """Get the average AoI from the current window"""
        if not self.env.paoi_window:
            return float('nan')
        return np.mean(self.env.paoi_window)

    def save_metrics(self, filename=None):
        if filename is None:
            filename = f"subscriber_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        metrics_summary = {
            'run_duration': time.time() - self.metrics['start_time'],
            'total_updates': self.metrics['total_updates'],
            'average_aoi': {
                'overall': np.mean(self.metrics['cu_aoi_values'] + self.metrics['zw_aoi_values']),
                'cu': np.mean(self.metrics['cu_aoi_values']) if self.metrics['cu_aoi_values'] else None,
                'zw': np.mean(self.metrics['zw_aoi_values']) if self.metrics['zw_aoi_values'] else None
            },
            'update_counts': {
                'cu': len(self.metrics['cu_aoi_values']),
                'zw': len(self.metrics['zw_aoi_values'])
            },
            'service_time_stats': {
                'average': np.mean(self.metrics['service_times']),
                'min': np.min(self.metrics['service_times']),
                'max': np.max(self.metrics['service_times'])
            },
            'detailed_metrics': {
                'timestamps': self.metrics['timestamps'],
                'cu_aoi_values': self.metrics['cu_aoi_values'],
                'zw_aoi_values': self.metrics['zw_aoi_values'],
                'service_times': self.metrics['service_times']
            }
        }

        with open(filename, 'w') as f:
            json.dump(metrics_summary, f, indent=4)
        self.logger.info(f"Metrics saved to {filename}")

        # Print current statistics
        self.logger.info("\nCurrent Statistics:")
        self.logger.info(f"Total Updates: {self.metrics['total_updates']}")
        self.logger.info(f"Average AoI - Overall: {metrics_summary['average_aoi']['overall']:.3f}")
        if metrics_summary['average_aoi']['cu']:
            self.logger.info(f"Average AoI - CU: {metrics_summary['average_aoi']['cu']:.3f}")
        if metrics_summary['average_aoi']['zw']:
            self.logger.info(f"Average AoI - ZW: {metrics_summary['average_aoi']['zw']:.3f}")

    def run(self, duration=300):  # Run for 5 minutes by default
        try:
            self.client.connect(self.env.broker, self.env.port)
            self.client.loop_start()

            start_time = time.time()
            self.logger.info(f"Starting subscriber for {duration} seconds...")

            while (time.time() - start_time) < duration:
                time.sleep(1)  # Check every second

            self.save_metrics()

        except KeyboardInterrupt:
            self.logger.info("\nStopping subscriber...")
        finally:
            self.save_metrics()
            self.client.loop_stop()
            self.client.disconnect()
            self.logger.info("Subscriber disconnected")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    subscriber = AutonomousSubscriber()
    subscriber.run(duration=300)  # Run for 5 minutes