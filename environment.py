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
        self.broker = "broker.emqx.io"
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
        self.mu = 3.0  # Default service rate
        self.lambda_rate = self.mu * 0.4  # Default interarrival rate

        self.mu_range = np.arange(1, 5.1, 0.1)  # Discretized service rate range

        self.max_updates = max_updates  # New parameter
        self.total_updates = 0  # Track total updates

        self.logger = logging.getLogger("environment")

    def reset(self):
        self.aoi_window.clear()
        self.time_step = 0
        self.current_policy = 0
        # FIXME: Do we need this?
        self.mu = random.choice(self.mu_range)
        self.lambda_rate = self.mu * 0.4
        return self.get_state()

    def step(self, action):
        """
        Take a step in the environment with the given action.
        `action` is a tuple: (policy, mu, lambda_rate)
        """
        policy, mu, lambda_rate = action
        print(f"Step called with action: {action}")
        print(f"Current PAoI window: {self.aoi_window}")

        # Update policy, service rate, and interarrival rate
        self.current_policy = policy
        self.mu = mu
        self.lambda_rate = lambda_rate

        # Ensure service_rate and lambda_rate are within the valid range
        self.logger.info(self.mu)
        self.logger.info(self.lambda_rate)
        self.logger.info(self.lambda_rate / self.mu)

        # assert self.mu in self.mu_range, "Service rate is out of range!"
        # assert np.isclose(self.lambda_rate / self.mu,
        #                   0.4), "Condition `lambda_rate / service_rate = 0.4` is violated!"

        policy_name = "ZW" if policy == 1 else "CU"
        action = {
            "policy": policy_name,
            "mu": mu,
            "lambda_rate": lambda_rate,
        }
        self.logger.info("Sending Action {}".format(action))
        self.env.client.publish(self.env.new_actions_topic, json.dumps(action))
        # self.publisher.publish_action_changes(policy_name, mu, lambda_rate)

        # Clear the PAoI window in the subscriber
        print(f"Resetting PAoI window and collecting new updates under policy {policy_name}...")
        self.subscriber.reset_paoi_window()

        self.logger.info("Waiting for new paoi window...")
        self.env.received_updates.wait()

        # Collect the new updates and calculate the average AoI
        paoi_values = self.env.retrieve_paoi_window()
        avg_aoi = np.mean(paoi_values) if paoi_values else float('nan')
        print(f"Policy: {policy_name}, New AoI Window: {paoi_values}")
        print(f"Policy: {policy_name}, Avg AoI: {avg_aoi}")
        self.env.received_updates.clear()

        # Calculate reward
        reward = -avg_aoi
        self.time_step += 1
        # done = self.time_step >= 1000  # End the episode after 1000 steps
        # FIXME: Update counters
        self.time_step += 1
        self.total_updates += len(paoi_values)  # Count actual updates received

        # End condition based on total updates
        done = self.total_updates >= self.max_updates
        if done:
            print(f"Simulation completed after {self.total_updates} updates")

        return self.get_state(), reward, done, {}


    def get_state(self):
        """
        Get the current state representation.
        Includes: [avg AoI, current_policy, service_rate, lambda_rate]
        """
        avg_aoi = np.mean(self.aoi_window) if len(self.aoi_window) >= 100 else 0
        return np.array([avg_aoi, self.current_policy, self.mu, self.lambda_rate])

class DQNAgent:
    def __init__(self, state_size, policies, mu_values, lambda_values):
        self.state_size = state_size
        self.action_space = list(itertools.product(policies, mu_values, lambda_values))  # Cartesian product
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
        """
        Choose an action based on epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        action_idx = torch.argmax(action_values).item()
        return self.action_space[action_idx]

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
        )

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Convert action tuple to index before storing
        action_idx = self.action_to_idx[action]
        self.memory.append((state, action_idx, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, action_idxs, rewards, next_states, dones = zip(*minibatch)

        # Convert to numpy arrays first
        states = np.array(states)
        next_states = np.array(next_states)
        action_idxs = np.array(action_idxs)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(action_idxs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current Q values
        current_q_values = self.model(states)
        q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]

        # Compute target Q values
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(q_values, targets.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()


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
        self.cu_aoi_history = []
        self.zw_aoi_history = []
        self.policy_counts = {0: 0, 1: 0}  # 0: CU, 1: ZW
        self.episode_timestamps = []

    def update_rewards(self, episode_reward):
        self.episode_rewards.append(episode_reward)
        window_size = 10
        self.moving_avg_rewards.append(
            np.mean(self.episode_rewards[-window_size:])
            if len(self.episode_rewards) >= window_size
            else episode_reward
        )

    def update_aoi(self, policy, aoi_value):
        if policy == 0:  # CU
            self.cu_aoi_history.append(aoi_value)
        else:  # ZW
            self.zw_aoi_history.append(aoi_value)

    def update_policy_count(self, policy):
        self.policy_counts[policy] += 1

    def save_metrics(self, filename="training_metrics.json"):
        metrics = {
            "rewards": {
                "episode_rewards": self.episode_rewards,
                "moving_avg_rewards": self.moving_avg_rewards
            },
            "aoi_performance": {
                "cu_aoi": self.cu_aoi_history,
                "zw_aoi": self.zw_aoi_history
            },
            "policy_distribution": self.policy_counts
        }
        with open(filename, 'w') as f:
            json.dump(metrics, f)


def train_rl_agent(env, agent, episodes=500, batch_size=32, window_size=10, convergence_threshold=0.01):
    metrics = TrainingMetrics()
    best_reward = float('-inf')
    episodes_without_improvement = 0
    max_episodes_without_improvement = 50

    for e in range(episodes):
        state = env.reset()
        episode_reward = 0
        step_count = 0
        episode_aois = []

        while True:
            action = agent.act(state)
            policy = action[0]
            metrics.update_policy_count(policy)
            logger.info(f"Testing policy: {'ZW' if policy == 1 else 'CU'}")

            next_state, reward, done, _ = env.step(action)
            logger.info(f"State: {next_state}, Reward: {reward}, Action: {action}")

            episode_reward += reward
            step_count += 1

            # Track AoI (negative of reward since reward = -avg_aoi)
            episode_aois.append(-reward)
            metrics.update_aoi(policy, -reward)

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                metrics.update_rewards(episode_reward)
                avg_reward = np.mean(metrics.episode_rewards[-window_size:]) if len(
                    metrics.episode_rewards) >= window_size else episode_reward

                logger.info(f"Episode {e + 1}/{episodes}")
                logger.info(f"Total Reward: {episode_reward:.2f}")
                logger.info(f"Average Reward (last {window_size} episodes): {avg_reward:.2f}")
                logger.info(f"Steps this episode: {step_count}")

                # Save metrics every 10 episodes
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

                if len(metrics.episode_rewards) >= window_size:
                    recent_std = np.std(metrics.episode_rewards[-window_size:])
                    if recent_std < convergence_threshold:
                        logger.info(f"Convergence achieved after {e + 1} episodes")
                        metrics.save_metrics()
                        return True, metrics

                break

        if e % 10 == 0:
            agent.update_target_model()

    metrics.save_metrics()
    return False, metrics

# def train_rl_agent(env, agent, episodes=500, batch_size=32):
#     policy_counts = {0: 0, 1: 0}  # Track how many times each policy is chosen
#
#     for e in range(episodes):
#         state = env.reset()
#         total_reward = 0
#
#         while True:
#             action = agent.act(state)
#             policy = action[0]  # Extract just the policy from the action tuple
#             policy_counts[policy] += 1  # Count the policy choice
#
#             logger.info(f"Testing policy: {'ZW' if policy == 1 else 'CU'}")
#
#             # Log policy distribution every 100 steps
#             total_actions = sum(policy_counts.values())
#             if total_actions % 100 == 0:
#                 cu_percent = (policy_counts[0] / total_actions) * 100
#                 zw_percent = (policy_counts[1] / total_actions) * 100
#                 logger.info(f"Policy distribution - CU: {cu_percent:.1f}%, ZW: {zw_percent:.1f}%")
#
#             next_state, reward, done, _ = env.step(action)
#
#             # Log state, action, and reward
#             logger.info(f"State: {next_state}, Reward: {reward}, Action: {action}")
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

if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.DEBUG)
        publisher = EnhancedMQTTPublisher()
        subscriber = EnhancedMQTTSubscriber()
        desired_updates = 500  # Set this to your desired number of updates
        env = AoIEnvironment(
            publisher,
            subscriber,
            window_size=10,
            max_updates=desired_updates
        )
        # env = AoIEnvironment(publisher, subscriber)
        env.env.client.connect(env.env.broker, env.env.port)
        mqtt_thread = threading.Thread(target=env.env.client.loop_start)
        mqtt_thread.daemon = True
        mqtt_thread.start()

        policies = [0, 1]  # CU or ZW
        mu_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        lambda_values = [x * 0.4 for x in mu_values]

        state_size = env.get_state().shape[0]
        agent = DQNAgent(state_size, policies, mu_values, lambda_values)

        # Train with metrics collection
        converged, metrics = train_rl_agent(
            env,
            agent,
            # episodes=500,
            episodes=10, # FIXME: For Sampling (Presentation)
            batch_size=32,
            window_size=10,
            convergence_threshold=0.01
        )

        if converged:
            print("Model converged successfully!")
        else:
            print("Maximum episodes reached without convergence.")

        print("\nTraining Summary:")
        print(f"Total Episodes: {len(metrics.episode_rewards)}")
        print(f"Final Average Reward: {metrics.moving_avg_rewards[-1]:.2f}")
        print("\nPolicy Distribution:")
        total_actions = sum(metrics.policy_counts.values())
        print(f"CU Policy: {(metrics.policy_counts[0]/total_actions)*100:.1f}%")
        print(f"ZW Policy: {(metrics.policy_counts[1]/total_actions)*100:.1f}%")

    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.env.client.loop_stop()
        env.env.client.disconnect()
        print("MQTT client disconnected")
