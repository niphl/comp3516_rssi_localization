#include <WiFi.h>
#include <PubSubClient.h>

const int scanInterval = 5000; // Scan interval in milliseconds
const char* password = "";


const char* connect_network = "Wi-Fi.HK via HKU"; // network we connect to
const char* target_network = ""; // target network to scan for, leave blank string to scan every network

// MQTT Broker
const char *mqtt_broker = "broker.emqx.io";
const char *topic = "RSSI_measurement";
const char *mqtt_username = "emqx";
const char *mqtt_password = "public";
const int mqtt_port = 1883;
WiFiClient espClient;
PubSubClient client(espClient);

// time
int timestamp = 0;

void connectWifi(const char* ssid, const char* password = "") {
  WiFi.mode(WIFI_STA); //Optional
  if (password == "") {
    WiFi.begin(ssid);
  } else {
    WiFi.begin(ssid, password);
  }

  Serial.println("\nConnecting");

  while(WiFi.status() != WL_CONNECTED){
      Serial.print(".");
      delay(100);
  }

  Serial.println("\nConnected to the WiFi network "+String(ssid));
  Serial.print("Local ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

void connectMQTT() {
  client.setServer(mqtt_broker, mqtt_port);
  client.setCallback(callback);
  while (!client.connected()) {
      String client_id = "esp32-client-";
      client_id += String(WiFi.macAddress());
      Serial.printf("The client %s connects to the public MQTT broker\n", client_id.c_str());
      if (client.connect(client_id.c_str(), mqtt_username, mqtt_password)) {
          Serial.println("Public EMQX MQTT broker connected");
      } else {
          Serial.print("failed with state ");
          Serial.print(client.state());
          delay(2000);
      }
  }
}

void callback(char *topic, byte *payload, unsigned int length) {
    //Serial.print("Message arrived in topic: ");
    //Serial.println(topic);
    //Serial.print("Message:");
    for (int i = 0; i < length; i++) {
        //Serial.print((char) payload[i]);
    }
    //Serial.println();
    //Serial.println("-----------------------");
}

void scanNetworks() {
  //Serial.println("Scanning Wifi Networks...");
  int networksFound = WiFi.scanNetworks(false, true);
  if (networksFound == WIFI_SCAN_RUNNING) {
    //Serial.println("Scan in progress. Waiting to complete...");
  } else if (networksFound == 0) {
    //Serial.println("No networks found.");
  } else {
    for (int i = 0; i < networksFound; ++i) {
      if (strcmp(target_network, WiFi.SSID(i).c_str()) == 0 || strlen(target_network) == 0) {
        client.publish(topic, (String(timestamp) + "," + String(WiFi.SSID(i)) + "," + String(WiFi.RSSI(i))+","+String(WiFi.BSSIDstr(i))).c_str());

      }
    }
  }
  timestamp = timestamp + scanInterval;
}

void setup() {
  Serial.begin(115200);
  Serial.println("Initializing...");
  delay(500);
  connectWifi(connect_network);
  connectMQTT();
  
  client.subscribe(topic);
  client.publish(topic, "timestamp,ssid,rssi,macaddress");
  scanNetworks();
}

void loop() {
  if (!client.connected()) {
    String client_id = "esp32-client-";
    client_id += String(WiFi.macAddress());
    client.connect(client_id.c_str(), mqtt_username, mqtt_password);
  }
  client.loop();
  delay(scanInterval);
  scanNetworks();
}