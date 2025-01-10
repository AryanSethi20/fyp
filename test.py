import paho.mqtt.client as mqtt

client = mqtt.Client()
client.connect("broker.emqx.io", 1883, 60)
test_message = '{"PAoI": [0.201, 0.202, 0.203], "policy": "CU"}'
client.publish("artc1/status_update/CU", test_message)