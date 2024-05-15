import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
import time
import io

from scipy.signal import butter, sosfiltfilt
import pickle

mqttBroker = "broker.emqx.io"
client = mqtt.Client("ESP32")
client.connect(mqttBroker)

def on_message(client, userdata, message):
    if str(message.payload.decode("utf-8")) == "timestamp,ssid,rssi,macaddress":
        print("Resetting payload.csv...")
        with open("payload.csv", "w") as file:
            file.truncate(0)
    print("MSG: "+str(message.payload.decode("utf-8")))
    
    with open("payload.csv", "a") as file:
        file.write(str(message.payload.decode("utf-8")) + "\n")
#client = mqtt.Client("Smartphone")
#client.connect(mqttBroker)

client.loop_start()
client.subscribe("RSSI_measurement")
client.on_message = on_message

print("Subscribed to channel RSSI_measurement. Listening...")

# Keep the client running until a specific condition is met
while True:
    user_input = input("Enter 'q' to quit: ")
    if user_input == 'q':
        break

client.loop_stop()