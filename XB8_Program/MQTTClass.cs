using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MQTTnet;
using MQTTnet.Client;
using MQTTnet.Protocol;
using Newtonsoft.Json;

namespace XB8_Program
{
    public class MQTTClass
    {
        private IMqttClient client;
        private MqttClientOptions options;


        public async Task InitializeAsync()
        {
            var factory = new MqttFactory();

            client = factory.CreateMqttClient();

            options = new MqttClientOptionsBuilder()
                .WithTcpServer("10.176.33.76", 1883)
                .WithCredentials("user", "user123")
                .WithCleanSession()
                .Build();

            client.ConnectedAsync += e =>
            {
                Console.WriteLine("MQTT Connected");
                return Task.CompletedTask;
            };

            client.DisconnectedAsync += e =>
            {
                Console.WriteLine("MQTT Disconnected");
                return Task.CompletedTask;
            };

            await client.ConnectAsync(options);
        }
        public async Task SendFail(string reason)
        {
            try
            {
                if (!client.IsConnected)
                    await client.ConnectAsync(options);

                var data = new
                {
                    Factory = "F1",
                    Floor = "2F",
                    Line = "2F17",
                    Project = "XB8",
                    StationName = "LABEL",
                    EquipmentName = "XB8_LABEL",
                    CurrentTime = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                    Level = "ERROR",
                    Category = "VISION",
                    CategoryNumber = "1",
                    ErrorCode = "",
                    Desc = "LABEL_INSPECTION_FAIL",
                    Message = $"FAIL,{reason}"
                };
                
                string payload = JsonConvert.SerializeObject(data);

                var message = new MqttApplicationMessageBuilder()
                    .WithTopic("running_message/station_data")
                    .WithPayload(Encoding.UTF8.GetBytes(payload))
                    .WithQualityOfServiceLevel(MqttQualityOfServiceLevel.AtLeastOnce)
                    .Build();

                await client.PublishAsync(message);
            }
            catch
            {
                // jangan sampai machine stop kalau MQTT gagal
            }
        }
    }
}
