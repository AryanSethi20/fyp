import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from publisher import EnhancedMQTTPublisher
from subscriber import EnhancedMQTTSubscriber
from paho.mqtt import client as mqtt_client
from threading import Event
import logging
import datetime
import threading
from ast import literal_eval
import itertools

# Date and time format
dateFormat = '%Y-%m-%d'
timeFormat = '%H-%M-%S.%f'

logger = logging.getLogger("__env__")

class MQTT_Environment:
    def __init__(self):
        self.client_id = f'environment-{random.randint(0, 1000)}'
        self.broker = "192.168.1.3"
        self.port = 1883
        self.client = mqtt_client.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.send_status_update_topic = "artc1/send_status_update"
        self.new_actions_topic = "artc1/new_actions"
        self.received_updates = Event()
        self.logger = logging.getLogger("mqtt_environment")
        self.paoi_window = list()

    # This function handles the callback when the broker reponds to the client's MQTT connection request.
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info(datetime.datetime.now().strftime(
                dateFormat + "|" + timeFormat) + ": Connected to MQTT Broker with topic " + self.send_status_update_topic)
            self.client.subscribe(self.send_status_update_topic)
        else:
            self.logger.error(datetime.datetime.now().strftime(
                dateFormat + "|" + timeFormat) + ": Failed to connect, return code {:n}".format(rc))

    def on_message(self, client, userdata, msg):
        self.logger.info(f"Received message on topic: {msg.topic}")
        try:
            self.logger.info(f"Received message: {msg.payload.decode()}")
            payload_str = msg.payload.decode()
            try:
                # Try parsing as comma-separated values
                values = literal_eval(payload_str)
                if not isinstance(values, list):
                    values = [values]
            except (ValueError, SyntaxError):
                # If that fails, try parsing as comma-separated values
                values = [float(x.strip()) for x in payload_str.split(',') if x.strip()]

            self.paoi_window = values
            self.received_updates.set()

        except Exception as e:
            print(f"Error processing message: {e}")

    def retrieve_paoi_window(self):
        return self.paoi_window

class AoIEnvironment:
    def __init__(self, publisher, subscriber, window_size=100, max_updates=500):
        self.env = MQTT_Environment()
        self.publisher = publisher
        self.subscriber = subscriber
        self.window_size = window_size
        self.aoi_window = deque(maxlen=self.window_size)
        self.current_policy = 0
        self.time_step = 0

        # Initialize service and arrival rates
        self.mu = 2.0  # Starting service rate
        self.lambda_rate = 1.0  # Starting arrival rate

        # Define valid ranges
        self.mu_min, self.mu_max = 1.0, 5.0
        self.ratio_min, self.ratio_max = 0.2, 0.9

        self.max_updates = max_updates
        self.total_updates = 0
        self.logger = logging.getLogger("environment")

    def reset(self):
        self.aoi_window.clear()
        self.time_step = 0
        self.current_policy = 0
        self.total_updates = 0
        return self.get_state()

    def step(self, action):
        """
        Take a step in the environment with the given action.
        Action is a tuple: (policy, mu, lambda_rate)
        """
        policy, mu, lambda_rate = action
        ratio = lambda_rate / mu

        # Validate the ratio is within bounds
        if not (self.ratio_min <= ratio <= self.ratio_max):
            print(f"Warning: Invalid ratio {ratio}. Adjusting rates...")
            if ratio < self.ratio_min:
                lambda_rate = mu * self.ratio_min
            elif ratio > self.ratio_max:
                lambda_rate = mu * self.ratio_max

        # Update environment parameters
        self.current_policy = policy
        self.mu = mu
        self.lambda_rate = lambda_rate

        # Log the current configuration
        self.logger.info(
            f"Current config - mu: {self.mu}, lambda: {self.lambda_rate}, ratio: {self.lambda_rate / self.mu}")

        # Send action to MQTT
        policy_name = "ZW" if policy == 1 else "CU"
        action = {
            "policy": policy_name,
            "mu": mu,
            "lambda_rate": lambda_rate,
        }
        self.env.client.publish(self.env.new_actions_topic, json.dumps(action))

        # Reset and collect new measurements
        self.subscriber.reset_paoi_window()
        self.env.received_updates.wait()

        # Process new measurements
        paoi_values = self.env.retrieve_paoi_window()
        avg_aoi = np.mean(paoi_values) if paoi_values else float('nan')
        self.env.received_updates.clear()

        # Calculate reward based on AoI and rate efficiency
        base_reward = -avg_aoi
        efficiency_bonus = 0.1 * (ratio - self.ratio_min) / (self.ratio_max - self.ratio_min)
        reward = base_reward + efficiency_bonus

        # Update counters
        self.time_step += 1
        self.total_updates += len(paoi_values)
        done = self.total_updates >= self.max_updates

        return self.get_state(), reward, done, {
            'avg_aoi': avg_aoi,
            'ratio': ratio,
            'efficiency_bonus': efficiency_bonus
        }


    def get_state(self):
        """
        Get the current state representation.
        State includes: [avg AoI, current_policy, service_rate, lambda_rate, lambda/mu ratio]
        """
        avg_aoi = np.mean(self.aoi_window) if len(self.aoi_window) >= 100 else 0
        ratio = self.lambda_rate / self.mu
        return np.array([avg_aoi, self.current_policy, self.mu, self.lambda_rate, ratio])


class DQNAgent:
    def __init__(self, state_size, policies):
        self.state_size = state_size

        # Create discrete action space
        self.policies = policies
        self.mu_values = np.linspace(1.0, 5.0, 5)  # 5 discrete service rates
        self.ratios = np.linspace(0.2, 0.9, 8)  # 8 discrete ratios

        # Generate all valid combinations
        self.action_space = []
        for policy in policies:
            for mu in self.mu_values:
                for ratio in self.ratios:
                    lambda_rate = mu * ratio
                    self.action_space.append((policy, mu, lambda_rate))

        self.action_size = len(self.action_space)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.learning_rate = 0.001

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

        # Create action to index mapping
        self.action_to_idx = {action: idx for idx, action in enumerate(self.action_space)}

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        action_idx = torch.argmax(action_values).item()
        return self.action_space[action_idx]

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),
        )

    def remember(self, state, action, reward, next_state, done):
        action_idx = self.action_to_idx[action]
        self.memory.append((state, action_idx, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, action_idxs, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(action_idxs)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        current_q_values = self.model(states)
        q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]

        targets = rewards + (1 - dones) * self.gamma * next_q_values
        loss = nn.MSELoss()(q_values, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# def train_rl_agent(env, agent, episodes=500, batch_size=32):
#     for e in range(episodes):
#         state = env.reset()
#         total_reward = 0
#
#         while True:
#             action = agent.act(state)
#             logger.info(f"Testing policy: {'ZW' if action == 1 else 'CU'}")
#
#             next_state, reward, done, _ = env.step(action)
#
#             # Log state, action, and reward
#             logger.info(f"State: {next_state}, Reward: {reward}, Action: {'ZW' if action == 1 else 'CU'}")
#
#             agent.remember(state, action, reward, next_state, done)
#             state = next_state
#             total_reward += reward
#
#             if done:
#                 logger.info(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}")
#                 break
#
#             if len(agent.memory) > batch_size:
#                 agent.replay(batch_size)
#
#         if e % 10 == 0:
#             agent.update_target_model()

class TrainingMetrics:
    def __init__(self):
        self.episode_rewards = []
        self.moving_avg_rewards = []
        self.policy_history = []
        self.mu_history = []
        self.lambda_history = []
        self.ratio_history = []
        self.aoi_history = []

    def update(self, reward, policy, mu, lambda_rate, aoi):
        self.episode_rewards.append(reward)
        self.policy_history.append(policy)
        self.mu_history.append(mu)
        self.lambda_history.append(lambda_rate)
        self.ratio_history.append(lambda_rate / mu)
        self.aoi_history.append(aoi)

        window_size = 100
        self.moving_avg_rewards.append(
            np.mean(self.episode_rewards[-window_size:])
            if len(self.episode_rewards) >= window_size
            else reward
        )

    def save_metrics(self, filename="training_metrics.json"):
        metrics = {
            "rewards": {
                "episode_rewards": self.episode_rewards,
                "moving_avg_rewards": self.moving_avg_rewards
            },
            "parameters": {
                "policy_history": self.policy_history,
                "mu_history": self.mu_history,
                "lambda_history": self.lambda_history,
                "ratio_history": self.ratio_history,
                "aoi_history": self.aoi_history
            }
        }
        with open(filename, 'w') as f:
            json.dump(metrics, f)

    def get_policy_distribution(self):
        policy_counts = {0: 0, 1: 0}  # CU: 0, ZW: 1
        for policy in self.policy_history:
            policy_counts[policy] += 1
        total = len(self.policy_history)
        return {
            'CU': (policy_counts[0] / total) * 100,
            'ZW': (policy_counts[1] / total) * 100
        }

    def get_rate_statistics(self):
        # Calculate statistics for the last window_size episodes
        window_size = min(50, len(self.mu_history))
        recent_mu = self.mu_history[-window_size:]
        recent_lambda = self.lambda_history[-window_size:]
        recent_ratio = self.ratio_history[-window_size:]

        return {
            'mu': {
                'mean': np.mean(recent_mu),
                'std': np.std(recent_mu),
                'min': np.min(recent_mu),
                'max': np.max(recent_mu)
            },
            'lambda': {
                'mean': np.mean(recent_lambda),
                'std': np.std(recent_lambda),
                'min': np.min(recent_lambda),
                'max': np.max(recent_lambda)
            },
            'ratio': {
                'mean': np.mean(recent_ratio),
                'std': np.std(recent_ratio),
                'min': np.min(recent_ratio),
                'max': np.max(recent_ratio)
            }
        }

    def get_performance_metrics(self):
        window_size = min(50, len(self.aoi_history))
        recent_aoi = self.aoi_history[-window_size:]
        recent_rewards = self.episode_rewards[-window_size:]

        return {
            'aoi': {
                'mean': np.mean(recent_aoi),
                'std': np.std(recent_aoi)
            },
            'reward': {
                'mean': np.mean(recent_rewards),
                'std': np.std(recent_rewards)
            }
        }


def print_training_summary(metrics):
    print("\nTraining Summary")
    print("=" * 50)
    print(f"Total Episodes: {len(metrics.episode_rewards)}")
    print(f"Final Average Reward: {metrics.moving_avg_rewards[-1]:.2f}")

    print("\nPolicy Distribution")
    print("-" * 20)
    dist = metrics.get_policy_distribution()
    print(f"CU Policy: {dist['CU']:.1f}%")
    print(f"ZW Policy: {dist['ZW']:.1f}%")

    print("\nRate Statistics (Last 50 Episodes)")
    print("-" * 20)
    stats = metrics.get_rate_statistics()

    print(f"Service Rate (μ):")
    print(f"  Mean: {stats['mu']['mean']:.2f} ± {stats['mu']['std']:.2f}")
    print(f"  Range: [{stats['mu']['min']:.2f}, {stats['mu']['max']:.2f}]")

    print(f"\nArrival Rate (λ):")
    print(f"  Mean: {stats['lambda']['mean']:.2f} ± {stats['lambda']['std']:.2f}")
    print(f"  Range: [{stats['lambda']['min']:.2f}, {stats['lambda']['max']:.2f}]")

    print(f"\nRatio (λ/μ):")
    print(f"  Mean: {stats['ratio']['mean']:.2f} ± {stats['ratio']['std']:.2f}")
    print(f"  Range: [{stats['ratio']['min']:.2f}, {stats['ratio']['max']:.2f}]")

    print("\nPerformance Metrics (Last 50 Episodes)")
    print("-" * 20)
    perf = metrics.get_performance_metrics()
    print(f"Average AoI: {perf['aoi']['mean']:.2f} ± {perf['aoi']['std']:.2f}")
    print(f"Average Reward: {perf['reward']['mean']:.2f} ± {perf['reward']['std']:.2f}")

def train_rl_agent(env, agent, episodes=500, batch_size=32, window_size=100, convergence_threshold=0.01):
    metrics = TrainingMetrics()
    best_reward = float('-inf')
    episodes_without_improvement = 0
    max_episodes_without_improvement = 50

    for e in range(episodes):
        state = env.reset()
        episode_reward = 0
        step_count = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            # Log metrics
            metrics.update(reward, action[0], action[1], action[2], info['avg_aoi'])

            episode_reward += reward
            step_count += 1

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                avg_reward = np.mean(metrics.episode_rewards[-window_size:]) if len(
                    metrics.episode_rewards) >= window_size else episode_reward

                logger.info(f"Episode {e + 1}/{episodes}")
                logger.info(f"Total Reward: {episode_reward:.2f}")
                logger.info(f"Average Reward (last {window_size} episodes): {avg_reward:.2f}")
                logger.info(f"Current mu: {action[1]:.2f}, lambda: {action[2]:.2f}, ratio: {action[2] / action[1]:.2f}")

                if e % 10 == 0:
                    metrics.save_metrics()

                if avg_reward > best_reward:
                    best_reward = avg_reward
                    episodes_without_improvement = 0
                    torch.save(agent.model.state_dict(), "best_model.pth")
                else:
                    episodes_without_improvement += 1

                if episodes_without_improvement >= max_episodes_without_improvement:
                    logger.info(f"Early stopping triggered after {e + 1} episodes")
                    metrics.save_metrics()
                    return True, metrics

                break

        if e % 10 == 0:
            agent.update_target_model()

    metrics.save_metrics()
    return False, metrics

if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.DEBUG)
        publisher = EnhancedMQTTPublisher()
        subscriber = EnhancedMQTTSubscriber()
        desired_updates = 10000  # Set this to your desired number of updates
        env = AoIEnvironment(
            publisher,
            subscriber,
            window_size=100,
            max_updates=desired_updates
        )
        # env = AoIEnvironment(publisher, subscriber)
        env.env.client.connect(env.env.broker, env.env.port)
        mqtt_thread = threading.Thread(target=env.env.client.loop_start)
        mqtt_thread.daemon = True
        mqtt_thread.start()

        policies = [0, 1]  # CU or ZW

        state_size = env.get_state().shape[0]
        agent = DQNAgent(state_size, policies)

        # Train with metrics collection
        converged, metrics = train_rl_agent(
            env,
            agent,
            episodes=500,
            # episodes=10, # FIXME: For Sampling (Presentation)
            batch_size=32,
            window_size=100,
            convergence_threshold=0.01
        )

        if converged:
            print("Model converged successfully!")
        else:
            print("Maximum episodes reached without convergence.")

        print_training_summary(metrics)

    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.env.client.loop_stop()
        env.env.client.disconnect()
        print("MQTT client disconnected")
