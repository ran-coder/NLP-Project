import paho.mqtt.client as mqtt
import time

# CRITICAL: You must specify the CallbackAPIVersion for paho-mqtt 2.x+
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

# Define callback behavior
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Successfully connected to Mosquitto!")
    else:
        print(f"Connection failed with code {rc}")

client.on_connect = on_connect

# Replace with 'localhost' if running on the same PC, or use your PC's local IP
BROKER_IP = "192.168.100.19" 
PORT = 1883

try:
    client.connect(BROKER_IP, PORT, 60)
    # Start the network loop in the background to handle data traffic
    client.loop_start()
    
    # Keep the script running to test the connection
    while True:
        time.sleep(1)
        
except Exception as e:
    print(f"Could not connect: {e}")