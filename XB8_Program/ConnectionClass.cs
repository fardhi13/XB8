using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Services.Description;
using EasyModbus;
using NModbus;
using NModbus.Device;

namespace XB8_Program
{
    public class ConnectionClass
    {
        public ModbusClient ModbusClient = new ModbusClient();
        private TcpClient tcpClient;
        private IModbusMaster modbusMaster;
        public bool ConnErr;
        public bool isConnected;
        private int ProductReady = int.Parse(ConfigurationManager.AppSettings["ProductReady"]);
        public int ErrorCode = 0;
        public string ErrorString;

        private string ip;
        private string port;

        #region NModbus
        public void Connect(string IP, string PORT)
        {
            ip = IP;
            port = PORT;
            try
            {
                tcpClient = new TcpClient();
                tcpClient.Connect(IP, int.Parse(PORT));
                var factory = new ModbusFactory();
                modbusMaster = factory.CreateMaster(tcpClient);
                isConnected = true;
            }
            catch (Exception ex)
            {
                isConnected = false;
                ErrorString = "[Modbus Connect Error] " + ex.Message;
            }
        }

        public void Disconnect()
        {
            try
            {
                tcpClient?.Close();
                isConnected = false;
            }
            catch (Exception ex)
            {
                ErrorString = "[Modbus Disconnect Error] " + ex.Message;
            }
        }

        public void WriteCoilMemory(int StartAddress, bool Value)
        {
            int IndexAddress = StartAddress;
            if (modbusMaster == null || !isConnected)
            {
                return;
            }

            try
            {
                modbusMaster.WriteSingleCoil(1, (ushort)IndexAddress, Value);
            }
            catch (Exception ex)
            {
                ErrorCode = 1;
                ErrorString = "[Modbus Error occurred]: " + ex.Message;
            }
        }
        private void TryReconnect()
        {
            Disconnect();
            Thread.Sleep(500); 
            Connect(ip, port);
        }
        public bool isProductReady()
        {
            try
            {
                if (isConnected && modbusMaster != null)
                {
                    bool[] status = modbusMaster.ReadCoils(1, (ushort)ProductReady, 1);
                    return status[0];
                }
            }
            catch (Exception ex)
            {
                ErrorCode = 1;
                ErrorString = "[Modbus ReadCoil Error]: " + ex.Message;

                TryReconnect();
            }

            return false;
        }
        public void KeepAlive()
        {
            _ = isProductReady();
        }
        #endregion
    }
}
