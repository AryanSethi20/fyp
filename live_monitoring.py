import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import json
import time
import numpy as np
import torch
import torch.nn as nn
import itertools
from publisher import EnhancedMQTTPublisher
from subscriber import EnhancedMQTTSubscriber
import logging


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


class LivePerformanceMonitor:
    def __init__(self, window_size=100):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.window_size = window_size

        # Data storage
        self.timestamps = deque(maxlen=window_size)
        self.aoi_values = deque(maxlen=window_size)
        self.policy_history = deque(maxlen=window_size)
        self.mu_history = deque(maxlen=window_size)
        self.lambda_history = deque(maxlen=window_size)

        # Initialize plots
        self.aoi_line, = self.ax1.plot([], [], 'b-', label='AoI')
        self.mu_line, = self.ax2.plot([], [], 'r-', label='Service Rate (μ)')
        self.lambda_line, = self.ax2.plot([], [], 'g-', label='Arrival Rate (λ)')

        # Setup axes
        self.ax1.set_title('Real-time AoI Performance')
        self.ax1.set_ylabel('Age of Information')
        self.ax1.grid(True)
        self.ax1.legend()

        self.ax2.set_title('System Parameters')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Rate')
        self.ax2.grid(True)
        self.ax2.legend()

        # Performance metrics
        self.performance_metrics = {
            'cu_aoi_values': [],
            'zw_aoi_values': [],
            'policy_switches': 0,
            'start_time': time.time()
        }

    def update_plot(self, frame):
        if not self.timestamps:
            return self.aoi_line, self.mu_line, self.lambda_line

        # Update x-axis limits
        self.ax1.set_xlim(min(self.timestamps), max(self.timestamps))
        self.ax2.set_xlim(min(self.timestamps), max(self.timestamps))

        # Update y-axis limits with some padding
        if self.aoi_values:
            aoi_min, aoi_max = min(self.aoi_values), max(self.aoi_values)
            self.ax1.set_ylim(aoi_min * 0.9, aoi_max * 1.1)

        if self.mu_history and self.lambda_history:
            rate_values = list(self.mu_history) + list(self.lambda_history)
            rate_min, rate_max = min(rate_values), max(rate_values)
            self.ax2.set_ylim(rate_min * 0.9, rate_max * 1.1)

        # Update line data
        self.aoi_line.set_data(list(self.timestamps), list(self.aoi_values))
        self.mu_line.set_data(list(self.timestamps), list(self.mu_history))
        self.lambda_line.set_data(list(self.timestamps), list(self.lambda_history))

        return self.aoi_line, self.mu_line, self.lambda_line

    def add_data_point(self, timestamp, aoi, policy, mu, lambda_rate):
        self.timestamps.append(timestamp)
        self.aoi_values.append(aoi)
        self.policy_history.append(policy)
        self.mu_history.append(mu)
        self.lambda_history.append(lambda_rate)

        if policy == 'CU':
            self.performance_metrics['cu_aoi_values'].append(aoi)
        else:
            self.performance_metrics['zw_aoi_values'].append(aoi)

        if len(self.policy_history) > 1:
            if self.policy_history[-1] != self.policy_history[-2]:
                self.performance_metrics['policy_switches'] += 1

    def save_metrics(self, filename="inference_metrics.json"):
        runtime = time.time() - self.performance_metrics['start_time']

        metrics = {
            'runtime_seconds': runtime,
            'average_cu_aoi': np.mean(self.performance_metrics['cu_aoi_values']) if self.performance_metrics[
                'cu_aoi_values'] else None,
            'average_zw_aoi': np.mean(self.performance_metrics['zw_aoi_values']) if self.performance_metrics[
                'zw_aoi_values'] else None,
            'policy_switches': self.performance_metrics['policy_switches'],
            'total_samples': len(self.aoi_values),
            'final_values': {
                'aoi': list(self.aoi_values)[-1] if self.aoi_values else None,
                'mu': list(self.mu_history)[-1] if self.mu_history else None,
                'lambda': list(self.lambda_history)[-1] if self.lambda_history else None
            }
        }

        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {filename}")


def run_with_live_monitoring():
    publisher = None
    try:
        # Initialize components
        monitor = LivePerformanceMonitor()

        # Set up animation with explicit save_count
        ani = FuncAnimation(monitor.fig, monitor.update_plot,
                            interval=1000, cache_frame_data=False,
                            save_count=100)  # Limit to 100 frames in cache

        # Initialize model and MQTT clients
        policies = [0, 1]
        mu_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        lambda_values = [x * 0.4 for x in mu_values]
        state_size = 4

        model = ModelInference("best_model.pth", state_size, policies, mu_values, lambda_values)
        publisher = EnhancedMQTTPublisher()
        subscriber = EnhancedMQTTSubscriber()

        # Connect MQTT
        publisher.client.connect(publisher.env.broker, publisher.env.port)
        publisher.client.loop_start()

        start_time = time.time()
        current_state = np.array([0, 0, 3.0, 1.2])

        plt.show(block=False)

        while True:
            policy, mu, lambda_rate = model.get_action(current_state)
            policy_name = "ZW" if policy == 1 else "CU"

            # Update publisher
            publisher.env.mu = mu
            publisher.env.lambda_rate = lambda_rate
            if policy_name == "ZW":
                publisher.env.ZW_policy_flag.set()
            else:
                publisher.env.ZW_policy_flag.clear()

            # Get current AoI and update plots
            current_aoi = subscriber.get_average_aoi()
            current_time = time.time() - start_time

            monitor.add_data_point(
                timestamp=current_time,
                aoi=current_aoi,
                policy=policy_name,
                mu=mu,
                lambda_rate=lambda_rate
            )

            # Update current state
            current_state = np.array([current_aoi, policy, mu, lambda_rate])

            plt.pause(0.1)
            time.sleep(0.9)

    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        if publisher is not None:
            print("Saving metrics and cleaning up...")
            monitor.save_metrics()
            publisher.client.loop_stop()
            publisher.client.disconnect()
        plt.close('all')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_with_live_monitoring()