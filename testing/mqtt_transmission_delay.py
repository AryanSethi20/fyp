import time
import json
import paho.mqtt.client as mqtt
import numpy as np

broker = "192.168.0.123"
topic = "test/aoi"

aoi_results = {size: [] for size in [16, 32, 64, 128, 256, 512, 1024]}
num_trials = 5

def on_message(client, userdata, msg):
    recv_time = time.time()
    
    try:
        msg_data = json.loads(msg.payload.decode())
        sent_time = msg_data["timestamp"]
        payload_size = msg_data["size"]

        aoi = recv_time - sent_time
        aoi_results[payload_size].append(aoi)

        print(f"Payload: {payload_size} bytes -> AoI: {aoi:.6f} sec")

    except Exception as e:
        print(f"Error decoding message: {e}")

client = mqtt.Client()
client.on_message = on_message
client.connect(broker)
client.subscribe(topic)
client.loop_start()

pub_client = mqtt.Client()
pub_client.connect(broker)

payload_sizes = [16, 32, 64, 128, 256, 512, 1024]

for payload_size in payload_sizes:
    for _ in range(num_trials):
        actual_payload = "A" * payload_size
        send_time = time.time()
        
        message = json.dumps({"timestamp": send_time, "size": payload_size, "data": actual_payload})
        
        pub_client.publish(topic, message)
        time.sleep(0.5)

client.loop_stop()

aoi_means = {size: np.mean(times) for size, times in aoi_results.items()}
max_payload = max(aoi_means, key=aoi_means.get)
max_aoi = aoi_means[max_payload]

print("\n🔍 **Summary of AoI Results (Averaged over Trials)**")
for size, aoi in aoi_means.items():
    print(f"Payload Size: {size} bytes -> Avg AoI: {aoi:.6f} sec")

print(f"\n🔥 **Max AoI observed for Payload Size: {max_payload} bytes -> AoI: {max_aoi:.6f} sec**")
