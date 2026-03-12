using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Configuration;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using Sdcb.PaddleOCR;
using Sdcb.PaddleOCR.Models.Local;
using Tesseract;
using FuzzySharp;
using OpenCvSharp.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection.Emit;


namespace XB8_Program
{
    public partial class Form1 : Form
    {
        private InferenceSession _session;
        private string _inputName;

        private FileSystemWatcher TheWatcher;
        private PaddleOcrAll PaddleOcrEngine;
        private TesseractEngine Tesseract_Engine;

        GlobalVariable GVL = new GlobalVariable();
        ConnectionClass CC = new ConnectionClass();
        SFISClass SFIS = new SFISClass();

        private string CurrentImagePath = "";
        private string tessPath = @"./tessdata";
        private bool isScanReady = false;
        private bool isOCRReady = false;
        private string InitPLCMsg = "";
        private string QRResult;
        private string BarcodeResult;
        private int MatchScore = 0;
        private double LeftLabelScore = 0;
        private double RightLabelScore = 0;
        private int SelectModel = 0;
        private int ImageRotation = 0;
        private bool isPassQR = false;
        private bool isPassBarcode = false;

        private string PLCIP = ConfigurationManager.AppSettings["PLCIP"];
        private string PLCPORT = ConfigurationManager.AppSettings["PLCPORT"];
        private int NGCoil = int.Parse(ConfigurationManager.AppSettings["NGCoil"]);
        private int PASSCoil = int.Parse(ConfigurationManager.AppSettings["PASSCoil"]);

        private System.Windows.Forms.Timer plcKeepAliveTimer;
        private Thread MThread;
        private bool ThreadRunning = true;

        private int TotalCount = 0;
        private int PassCount = 0;
        private int NgCount = 0;
        public Form1()
        {
            InitializeComponent();
            SFIS.SFIS_LogINOUT(1);
            InitOnnx();
            InitPaddleOcr();
            InitTesseract();
            InitWatcher();
            InitPLC();
        }

        private void InitOnnx()
        {
            string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "XB8.onnx");
            _session = new InferenceSession(modelPath);
            _inputName = _session.InputMetadata.Keys.First();
        }

        private void InitPaddleOcr()
        {
            var model = LocalFullModels.EnglishV3;
            PaddleOcrEngine = new PaddleOcrAll(model)
            {
                AllowRotateDetection = false,
                Enable180Classification = false,
            };
        }

        private void InitTesseract()
        {
            try
            {
                Tesseract_Engine = new TesseractEngine(tessPath, "eng", EngineMode.Default);
                Tesseract_Engine.SetVariable("tessedit_char_whitelist", "-:ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz");
                Tesseract_Engine.SetVariable("user_defined_dpi", "300");
            }
            catch (Exception ex)
            {
                MessageBox.Show("Tesseract initialization failed: " + ex.Message);
            }
        }

        private void InitWatcher()
        {
            TheWatcher = new FileSystemWatcher();
            TheWatcher.Path = $@"{GVL.WatcherDir}";
            TheWatcher.Filter = "*.*";
            TheWatcher.NotifyFilter = NotifyFilters.FileName | NotifyFilters.LastWrite;
            TheWatcher.Created += OnNewImage;
            TheWatcher.EnableRaisingEvents = true;
        }

        private void InitPLC()
        {
            try
            {
                CC.Connect(PLCIP, PLCPORT);
                if (CC.isConnected == false)
                {
                    InitPLCMsg = "PLC Connection Fail";
                    MessageBox.Show(InitPLCMsg);
                    return;
                }
                else
                {
                    InitPLCMsg = "PLC Connected";
                    MessageBox.Show(InitPLCMsg);
                }

                plcKeepAliveTimer = new System.Windows.Forms.Timer();
                plcKeepAliveTimer.Interval = 600000; 
                plcKeepAliveTimer.Tick += (s, e) => CC.KeepAlive();
                plcKeepAliveTimer.Start();
            }
            catch (Exception ex)
            {
                InitPLCMsg = "Error: " + ex + Environment.NewLine +
                    "Please check the LAN cable between PLC to the computer and make sure PLC IP Address already configured";
                MessageBox.Show(InitPLCMsg);
            }
        }

        private bool AllBoxLabelsDetected()
        {
            return GVL.WidthMAC > 0 && GVL.WidthSSID > 0 && GVL.WidthBARCODE > 0 &&
                   GVL.WidthPN > 0 && GVL.WidthFCCID > 0 && GVL.WidthQR > 0 &&
                   GVL.WidthPASSWORD > 0 && GVL.WidthRIGHTSIDE > 0 && GVL.WidthLEFTSIDE > 0;
        }
        

        private void ResetBoxCoordinates()
        {
            GVL.StartXPN = GVL.StartYPN = GVL.WidthPN = GVL.HeightPN = 0;
            GVL.StartXFCCID = GVL.StartYFCCID = GVL.WidthFCCID = GVL.HeightFCCID = 0;
            GVL.StartXMAC = GVL.StartYMAC = GVL.WidthMAC = GVL.HeightMAC = 0;
            GVL.StartXSSID = GVL.StartYSSID = GVL.WidthSSID = GVL.HeightSSID = 0;
            GVL.StartXPASSWORD = GVL.StartYPASSWORD = GVL.WidthPASSWORD = GVL.HeightPASSWORD = 0;
            GVL.StartXBARCODE = GVL.StartYBARCODE = GVL.WidthBARCODE = GVL.HeightBARCODE = 0;
            GVL.StartXQR = GVL.StartYQR = GVL.WidthQR = GVL.HeightQR = 0;
            GVL.StartXLEFTSIDE = GVL.StartYLEFTSIDE = GVL.WidthLEFTSIDE = GVL.HeightLEFTSIDE = 0;
            GVL.StartXRIGHTSIDE = GVL.StartYRIGHTSIDE = GVL.WidthRIGHTSIDE = GVL.HeightRIGHTSIDE = 0;
            GVL.StartXLOGOAREA = GVL.StartYLOGOAREA = GVL.WidthLOGOAREA = GVL.HeightLOGOAREA = 0;
            GVL.IsLogoDetected = false;
        }

        private HashSet<string> processingFiles = new HashSet<string>();

        private void OnNewImage(object sender, FileSystemEventArgs e)
        {
            if (!new[] { ".jpg", ".png", ".bmp" }
                .Contains(Path.GetExtension(e.FullPath).ToLower()))
                return;

            lock (processingFiles)
            {
                if (processingFiles.Contains(e.FullPath))
                    return;

                processingFiles.Add(e.FullPath);
            }

            Task.Run(() =>
            {
                try
                {
                    // Tunggu sampai file selesai ditulis
                    int retries = 0;
                    while (retries < 10)
                    {
                        try
                        {
                            using (var stream = new FileStream(
                                e.FullPath,
                                FileMode.Open,
                                FileAccess.Read,
                                FileShare.ReadWrite))
                            {
                                if (stream.Length > 0)
                                    break;
                            }
                        }
                        catch
                        {
                            Thread.Sleep(20);
                            retries++;
                        }
                    }

                    this.Invoke((MethodInvoker)(() =>
                    {
                        CurrentImagePath = e.FullPath;
                        CleanupWatcherDirectory(e.FullPath);
                        ProcessImage(e.FullPath);
                    }));
                }
                finally
                {
                    lock (processingFiles)
                    {
                        processingFiles.Remove(e.FullPath);
                    }
                }
            });
        }

        private void CleanupWatcherDirectory(string newFilePath)
        {
            try
            {
                string dir = GVL.WatcherDir;

                if (!Directory.Exists(dir))
                    return;

                var files = new DirectoryInfo(dir)
                    .GetFiles()
                    .OrderByDescending(f => f.CreationTime)
                    .ToList();

                if (files.Count <= 500)
                    return;

                foreach (var file in files.Skip(1)) // skip file terbaru
                {
                    try
                    {
                        if (file.FullName != newFilePath)
                        {
                            file.Delete();
                        }
                    }
                    catch
                    {
                        // ignore
                    }
                }
            }
            catch
            {
                // ignore error
            }
        }

        private void CleanupResultDirectory(string folderPath, string latestFilePath, int maxFiles = 1000)
        {
            try
            {
                if (!Directory.Exists(folderPath))
                    return;

                var files = Directory.GetFiles(folderPath);

                if (files.Length <= maxFiles)
                    return;

                foreach (var file in files)
                {
                    try
                    {
                        if (!file.Equals(latestFilePath, StringComparison.OrdinalIgnoreCase))
                        {
                            File.Delete(file);
                        }
                    }
                    catch
                    {
                        // ignore 
                    }
                }
            }
            catch
            {
                // ignore error
            }
        }
        private void Form1_Load(object sender, EventArgs e)
        {
            ComboBox_Rotation.SelectedIndex = 2;
            ComboBox_Model.SelectedIndex = 0;
        }

        private void UpdateCounter(bool isPass)
        {
            TotalCount++;

            if (isPass)
                PassCount++;
            else
                NgCount++;

            double yieldRate = 0;

            if (TotalCount > 0)
                yieldRate = (double)PassCount / TotalCount * 100.0;

            label_total.Text = TotalCount.ToString();
            label_pass.Text = PassCount.ToString();
            label_ng.Text = NgCount.ToString();
            label_yieldrate.Text = yieldRate.ToString("F2") + " %";
        }


        private void CheckingProcess()
        {
            bool isPassModel = false, isPassFCC = false, isPassCM = false, isPassWAN = false, isPassMTA = false, isPassSN = false, isPassNetworkName = false, isPassPassword = false, 
                isPassLeftLabel = false, isPassRightLabel = false, isPassLogo = false;

            isScanReady = false;
            isOCRReady = false;

            string time = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}";
            AppendNewEntry($"[{time}] ===NEW ENTRY===", Color.Black);


            //SN
            var SNChecking = SFIS.SFIS_CheckSN(GVL.QR_SN); 
            if (SNChecking.Contains("NOT FOUND") || SNChecking.Contains("not Defined") || SNChecking.StartsWith("0") || string.IsNullOrEmpty(SNChecking))
            {
                isPassSN = false;

                string SFISResponse = SFIS.StringSNReader;
                string[] DATA = SFISResponse.Split(new char[] { '\x7f' });
                GVL.StartResult = DATA[0];
                GVL.TypeResult = DATA[1];
                GVL.Model = "NOT FOUND";
                GVL.SSID = "NOT FOUND";
                GVL.Password = "NOT FOUND";
                AppendLogText($"[{GVL.QR_SN}] - SN is not valid, check again and retry!", Color.Red);
                this.Invoke(new Action(() =>
                {
                    SN_TextBox.ForeColor = Color.Red;
                }));
                SNChecking = "";
            }
            else if (SNChecking.Contains("GET OK") || SNChecking.StartsWith("1"))
            {
                string SFISResponse = SFIS.StringSNReader;
                string[] DATA = SFISResponse.Split(new char[] { '\x7f' });

                GVL.StartResult = DATA[0];
                GVL.TypeResult = DATA[1];
                GVL.Model = DATA[2];
                GVL.SSID = DATA[3];
                GVL.Password = DATA[4];
                Console.WriteLine(DATA[2] + "\n" + DATA[3] + "\n" + DATA[4]);

                isPassSN = true;
                AppendLogText($"[{GVL.QR_SN}] - SN is valid", Color.Green);
                this.Invoke(new Action(() =>
                {
                    SN_TextBox.ForeColor = Color.Black;
                }));
                GVL.QR_SN = "";
            }

            //Technicolor model/PN/Comcast model

            if (GVL.OCR_PN.Equals(GVL.QR_TechnicolorModel))
            {
                isPassModel = true;
                AppendLogText($"[{GVL.QR_TechnicolorModel}] - Model is valid", Color.Green);
                this.Invoke(new Action(() =>
                {
                    PN_TextBox.ForeColor = Color.Black;
                }));
                GVL.QR_TechnicolorModel = "";

                //if (GVL.QR_ComcastModel.Contains(GVL.Model))
                //{
                //    isPassModel = true;
                //    AppendLogText($"[{GVL.QR_TechnicolorModel}] - Model is valid", Color.Green);
                //    this.Invoke(new Action(() =>
                //    {
                //        PN_TextBox.ForeColor = Color.Black;
                //    }));
                //    GVL.QR_TechnicolorModel = "";
                //}
                //else
                //{
                //    isPassModel = false;
                //    AppendLogText($"[{GVL.QR_TechnicolorModel}] - Model not matched, OCR READ = [{GVL.OCR_PN}], check again and retry!", Color.Red);
                //    this.Invoke(new Action(() =>
                //    {
                //        PN_TextBox.ForeColor = Color.Red;
                //    }));
                //    GVL.QR_TechnicolorModel = "";
                //}
            }
            else
            {
                string fallbackModel = ProcessOCR_Tesseract(new Bitmap(ModelPicture.Image), GVL.QR_TechnicolorModel, @"[A-Z0-9]{10}");
                if (fallbackModel == GVL.QR_TechnicolorModel)
                {
                    isPassModel = true;
                    AppendLogText($"[{GVL.QR_TechnicolorModel}] - Model is valid (Tesseract fallback)", Color.Orange);
                    this.Invoke(new Action(() =>
                    {
                        PN_TextBox.ForeColor = Color.DarkOrange;
                        PN_TextBox.Text = fallbackModel;
                    }));
                    GVL.QR_TechnicolorModel = "";
                }
                else if (FuzzyMatch(GVL.OCR_PN, GVL.QR_TechnicolorModel))
                {
                    isPassModel = true;
                    AppendLogText($"[{GVL.QR_TechnicolorModel}] - Model is valid (Fuzzymatch fallback)", Color.Orange);
                    this.Invoke(new Action(() =>
                    {
                        PN_TextBox.ForeColor = Color.DarkOrange;
                        PN_TextBox.Text = GVL.QR_TechnicolorModel;
                    }));
                    GVL.QR_TechnicolorModel = "";
                }
                else
                {
                    isPassModel = false;
                    AppendLogText($"[{GVL.QR_TechnicolorModel}] -  Model not matched, OCR READ = [{GVL.OCR_PN}], check again and retry!", Color.Red);
                    this.Invoke(new Action(() =>
                    {
                        PN_TextBox.ForeColor = Color.Red;
                    }));
                }
            }

            //FCC ID
            if (GVL.OCR_FCCID.Equals("G954981X"))
            {
                isPassFCC = true;
                AppendLogText($"[{GVL.OCR_FCCID}] - FCC ID is valid", Color.Green);
                this.Invoke(new Action(() =>
                {
                    FCCID_TextBox.ForeColor = Color.Black;
                }));
            }
            else
            {
                string fallbackFCC = ProcessOCR_Tesseract(new Bitmap(FCCIDPicture.Image), "G954981X", @"[A-Z0-9]{8}");
                if (fallbackFCC == "G954981X")
                {
                    isPassFCC = true;
                    AppendLogText($"[G954981X] - FCC ID is valid (Tesseract fallback)", Color.Orange);
                    this.Invoke(new Action(() =>
                    {
                        FCCID_TextBox.ForeColor = Color.DarkOrange;
                        FCCID_TextBox.Text = fallbackFCC;
                    }));
                }
                else if (FuzzyMatch(GVL.OCR_FCCID, "G954981X"))
                {
                    isPassFCC = true;
                    AppendLogText($"[G954981X] - FCC ID is valid (Fuzzy Matching fallback)", Color.Orange);
                    this.Invoke(new Action(() =>
                    {
                        FCCID_TextBox.ForeColor = Color.DarkOrange;
                        FCCID_TextBox.Text = "G954981X";
                    }));
                }
                else
                {
                    isPassFCC = false;
                    AppendLogText($"[{GVL.OCR_FCCID}] - FCC ID not matched, check again and retry!", Color.Red);
                    this.Invoke(new Action(() =>
                    {
                        FCCID_TextBox.ForeColor = Color.Red;
                    }));
                }
            }
            

            // CMMac
            if (GVL.OCR_CM_MAC.Equals(GVL.QR_CM_MAC))
            {
                isPassCM = true;
                AppendLogText($"[{GVL.QR_CM_MAC}] - CM MAC matched", Color.Green);
                this.Invoke(new Action(() =>
                {
                    OCRCMMAC_TextBox.ForeColor = Color.Black;
                }));
                GVL.QR_CM_MAC = "";
            }
            else
            {
                string fallbackCM = ProcessOCR_Tesseract(new Bitmap(MACPicture.Image), GVL.QR_CM_MAC, @"[A-Z0-9]{12}");
                if (fallbackCM == GVL.QR_CM_MAC)
                {
                    isPassCM = true;
                    AppendLogText($"[{GVL.QR_CM_MAC}] - CM MAC matched (Tesseract fallback)", Color.Orange);
                    OCRCMMAC_TextBox.Text = fallbackCM;
                    OCRCMMAC_TextBox.ForeColor = Color.DarkOrange;
                    GVL.QR_CM_MAC = "";
                }
                else if (FuzzyMatch(GVL.OCR_CM_MAC, GVL.QR_CM_MAC))
                {
                    isPassCM = true;
                    AppendLogText($"[{GVL.QR_CM_MAC}] - CM MAC matched (Fuzzy Matching fallback)", Color.Orange);
                    OCRCMMAC_TextBox.Text = GVL.QR_CM_MAC;
                    OCRCMMAC_TextBox.ForeColor = Color.DarkOrange;
                    GVL.QR_CM_MAC = "";
                }
                else
                {
                    isPassCM = false;
                    AppendLogText($"[{GVL.QR_CM_MAC}] - CM MAC mismatch. Paddle: {GVL.OCR_CM_MAC}, Tesseract: {fallbackCM}", Color.Red);
                    OCRCMMAC_TextBox.ForeColor = Color.Red;
                }
            }

            // WANMac
            if (GVL.OCR_WAN_MAC.Equals(GVL.QR_WAN_MAC))
            {
                isPassWAN = true;
                AppendLogText($"[{GVL.QR_WAN_MAC}] - WAN MAC matched", Color.Green);
                this.Invoke(new Action(() =>
                {
                    OCRWANMAC_TextBox.ForeColor = Color.Black;
                }));
                GVL.QR_WAN_MAC = "";
            }
            else
            {
                string fallbackWAN = ProcessOCR_Tesseract(new Bitmap(MACPicture.Image), GVL.QR_WAN_MAC, @"[A-Z0-9]{12}");
                if (fallbackWAN == GVL.QR_WAN_MAC)
                {
                    isPassWAN = true;
                    AppendLogText($"[{GVL.QR_WAN_MAC}] - WAN MAC matched (Tesseract fallback)", Color.Orange);
                    OCRWANMAC_TextBox.Text = fallbackWAN;
                    OCRWANMAC_TextBox.ForeColor = Color.DarkOrange;
                    GVL.QR_WAN_MAC = "";
                }
                else if (FuzzyMatch(GVL.OCR_WAN_MAC, GVL.QR_WAN_MAC))
                {
                    isPassWAN = true;
                    AppendLogText($"[{GVL.QR_WAN_MAC}] - WAN MAC matched (Fuzzy Matching fallback)", Color.Orange);
                    OCRWANMAC_TextBox.Text = GVL.QR_WAN_MAC;
                    OCRWANMAC_TextBox.ForeColor = Color.DarkOrange;
                    GVL.QR_WAN_MAC = "";
                }
                else
                {
                    isPassWAN = false;
                    AppendLogText($"[{GVL.QR_WAN_MAC}] - WAN MAC mismatch. Paddle: {GVL.OCR_WAN_MAC}, Tesseract: {fallbackWAN}", Color.Red);
                    OCRWANMAC_TextBox.ForeColor = Color.Red;
                }
            }

            // MTAMac
            if (GVL.OCR_MTA_MAC.Equals(GVL.QR_MTA_MAC))
            {
                isPassMTA = true;
                AppendLogText($"[{GVL.QR_MTA_MAC}] - MTA MAC matched", Color.Green);
                this.Invoke(new Action(() =>
                {
                    OCRMTAMAC_TextBox.ForeColor = Color.Black;
                }));
                GVL.QR_MTA_MAC = "";
            }
            else
            {
                string fallbackMTA = ProcessOCR_Tesseract(new Bitmap(MACPicture.Image), GVL.QR_MTA_MAC, @"[A-Z0-9]{12}");
                if (fallbackMTA == GVL.QR_MTA_MAC)
                {
                    isPassMTA = true;
                    AppendLogText($"[{GVL.QR_MTA_MAC}] - MTA MAC matched (Tesseract fallback)", Color.Orange);
                    OCRMTAMAC_TextBox.Text = fallbackMTA;
                    OCRMTAMAC_TextBox.ForeColor = Color.DarkOrange;
                    GVL.QR_MTA_MAC = "";
                }
                else if (FuzzyMatch(GVL.OCR_MTA_MAC, GVL.QR_MTA_MAC))
                {
                    isPassMTA = true;
                    AppendLogText($"[{GVL.QR_MTA_MAC}] - MTA MAC matched (Fuzzy Matching fallback)", Color.Orange);
                    OCRMTAMAC_TextBox.Text = GVL.QR_MTA_MAC;
                    OCRMTAMAC_TextBox.ForeColor = Color.DarkOrange;
                    GVL.QR_MTA_MAC = "";
                }
                else
                {
                    isPassMTA = false;
                    AppendLogText($"[{GVL.QR_MTA_MAC}] - MTA MAC mismatch. Paddle: {GVL.OCR_MTA_MAC}, Tesseract: {fallbackMTA}", Color.Red);
                    OCRMTAMAC_TextBox.ForeColor = Color.Red;
                }
            }

            // NetworkName
            if (GVL.SSID.Contains(GVL.OCR_LastFourNetworkName))
            {
                Console.WriteLine(GVL.OCR_NetworkName);
                if (GVL.OCR_NetworkName.Equals(GVL.SSID) || FuzzyMatch(GVL.OCR_NetworkName, GVL.SSID))
                {
                    isPassNetworkName = true;
                    AppendLogText($"[{GVL.OCR_NetworkName}] - SSID is valid", Color.Green);
                    this.Invoke(new Action(() =>
                    {
                        NETWORKNAME_TextBox.ForeColor = Color.Black;
                    }));
                }
                else
                {
                    string fallbackSSID = ProcessOCR_Tesseract(new Bitmap(SSIDPicture.Image), GVL.SSID, @"^(XF)?SETUP\-[A-Z0-9]{4}$");
                    if (fallbackSSID == GVL.SSID)
                    {
                        isPassNetworkName = true;
                        AppendLogText($"[{fallbackSSID}] - SSID matched (Tesseract fallback)", Color.Orange);
                        NETWORKNAME_TextBox.Text = fallbackSSID;
                        NETWORKNAME_TextBox.ForeColor = Color.DarkOrange;
                    }
                    else if(FuzzyMatch(GVL.OCR_NetworkName, GVL.SSID))
                    {
                        isPassNetworkName = true;
                        AppendLogText($"[{GVL.SSID}] - SSID matched (Fuzzy Matching fallback)", Color.Orange);
                        NETWORKNAME_TextBox.Text = GVL.SSID;
                        NETWORKNAME_TextBox.ForeColor = Color.DarkOrange;
                    }
                    else
                    {
                        isPassNetworkName = false;
                        AppendLogText($"[{GVL.OCR_NetworkName}] - SSID Label not matched, SFIS SSID = [{GVL.SSID}], fallback = [{fallbackSSID}]", Color.Red);
                        this.Invoke(new Action(() =>
                        {
                            NETWORKNAME_TextBox.ForeColor = Color.Red;
                        }));
                    }
                }
            }
            else
            {
                string fallbackSSID = ProcessOCR_Tesseract(new Bitmap(SSIDPicture.Image), GVL.SSID, @"^(XF)?SETUP\-[A-Z0-9]{4}$");
                if (fallbackSSID == GVL.SSID)
                {
                    isPassNetworkName = true;
                    AppendLogText($"[{fallbackSSID}] - SSID matched (Tesseract fallback)", Color.Orange);
                    NETWORKNAME_TextBox.Text = fallbackSSID;
                    NETWORKNAME_TextBox.ForeColor = Color.DarkOrange;
                }
                else if (FuzzyMatch(GVL.OCR_NetworkName, GVL.SSID))
                {
                    isPassNetworkName = true;
                    AppendLogText($"[{GVL.SSID}] - SSID matched (Fuzzy Matching fallback)", Color.Orange);
                    NETWORKNAME_TextBox.Text = GVL.SSID;
                    NETWORKNAME_TextBox.ForeColor = Color.DarkOrange;
                }
                else
                {
                    isPassNetworkName = false;
                    AppendLogText($"[{GVL.OCR_NetworkName}] - SSID Label not matched, SFIS SSID = [{GVL.SSID}], fallback = [{fallbackSSID}]", Color.Red);
                    this.Invoke(new Action(() =>
                    {
                        NETWORKNAME_TextBox.ForeColor = Color.Red;
                    }));
                }
            }

            //Password
            if (GVL.OCR_Password.Equals(GVL.Password)) 
            {
                isPassPassword = true;
                AppendLogText($"[{GVL.OCR_Password}] - Password is valid", Color.Green);
                this.Invoke(new Action(() =>
                {
                    PASSWORD_TextBox.ForeColor = Color.Black;
                }));
            }
            else
            {
                string fallbackPassword = ProcessOCR_Tesseract(new Bitmap(PasswordPicture.Image), GVL.Password, @"^[a-z0-9]+$");
                if (fallbackPassword == GVL.Password)
                {
                    isPassPassword = true;
                    AppendLogText($"[{fallbackPassword}] - Password matched (Tesseract fallback)", Color.Orange);
                    PASSWORD_TextBox.Text = fallbackPassword;
                    PASSWORD_TextBox.ForeColor = Color.DarkOrange;
                }
                else if (FuzzyMatchPassword(GVL.OCR_Password, GVL.Password))
                {
                    isPassPassword = true;
                    AppendLogText($"[{GVL.Password}] - Password matched (Fuzzy Match fallback)", Color.Orange);
                    PASSWORD_TextBox.Text = GVL.Password;
                    PASSWORD_TextBox.ForeColor = Color.DarkOrange;
                }
                else
                {
                    isPassPassword = false;
                    AppendLogText($"[{GVL.OCR_Password}] - Password mismatch. Tesseract fallback = [{fallbackPassword}], SFIS = [{GVL.Password}]", Color.Red);
                    PASSWORD_TextBox.ForeColor = Color.Red;
                }
            }

            //Left Label
            if (LeftLabelScore >= 99)
            {
                isPassLeftLabel = true;
                AppendLogText($"Left Label {LeftLabelScore:0.0}% matched", Color.Green);
            }
            else
            {
                isPassLeftLabel = false;
                AppendLogText($"Left Label {LeftLabelScore:0.0}% matched, must be more than 98%, check again and retry!", Color.Red);
            }

            //Right Label
            if (RightLabelScore >= 99)
            {
                isPassRightLabel = true;
                AppendLogText($"Right Label {RightLabelScore:0.0}% matched", Color.Green);
            }
            else
            {
                isPassRightLabel = false;
                AppendLogText($"Right Label {RightLabelScore:0.0}% matched, must be more than 98%, check again and retry!", Color.Red);
            }

            // LOGOAREA CHECK
            if (GVL.IsLogoDetected && GVL.WidthLOGOAREA > 0 && GVL.HeightLOGOAREA > 0)
            {
                isPassLogo = true;
                AppendLogText("Logo detected", Color.Green);
            }
            else
            {
                isPassLogo = false;
                AppendLogText("Logo NOT detected!", Color.Red);
            }
            this.Invoke(new Action(() =>
            {
                if (isPassModel && isPassFCC && isPassCM && isPassWAN && isPassMTA && isPassSN && isPassPassword && isPassNetworkName && isPassLeftLabel && isPassRightLabel && isPassBarcode && isPassQR && isPassLogo)
                {
                    isPassBarcode = false;
                    isPassQR = false;
                    CC.WriteCoilMemory(PASSCoil, true);
                    LabelResult.Text = "PASS";
                    LabelResult.ForeColor = Color.Green;
                    AppendLogText("PASS", Color.Green);
                    Thread.Sleep(200);
                    CC.WriteCoilMemory(PASSCoil, false);
                    if (CC.ErrorCode == 1)
                    {
                        AppendLogText(CC.ErrorString, Color.Red);
                        CC.ErrorCode = 0;
                    }
                    SaveImageByResult(SourcePicture.Image, true);
                    UpdateCounter(true);
                }
                else
                { 
                    CC.WriteCoilMemory(NGCoil, true);
                    LabelResult.Text = "NG";
                    LabelResult.ForeColor = Color.Red;
                    AppendLogText("NG", Color.Red);
                    Thread.Sleep(200);
                    CC.WriteCoilMemory(NGCoil, false);
                    if (CC.ErrorCode == 1)
                    {
                        AppendLogText(CC.ErrorString, Color.Red);
                        CC.ErrorCode = 0;
                    }
                    SaveImageByResult(SourcePicture.Image, false);
                    UpdateCounter(false);
                }
            }));
        }

        private bool FuzzyMatch(string s1, string s2)
        {
            if (s1.Length != s2.Length) return false;

            Dictionary<char, char[]> similar = new Dictionary<char, char[]>
            {
                { '0', new[] { 'O', 'D' , 'C' } },
                { 'O', new[] { '0', 'D', 'C' } },
                { 'D', new[] { '0', 'O', 'C' } },
                { '1', new[] { 'I' } },
                { 'I', new[] { '1', 'J' } },
                { 'J', new[] { '1', 'I' } },
                { '5', new[] { 'S' } },
                { 'S', new[] { '5' } },
                { '8', new[] { 'B', 'F' } },
                { 'B', new[] { '8', 'E' } },
                { 'E', new[] { 'B', 'F', '8' } },
                { 'F', new[] {'E'} },
                { 'Z', new[] { '2' } },
                { '2', new[] { 'Z' } },
                { 'G', new[] { '6', 'C' } },
                { '6', new[] { 'G' } },
                { 'C', new[] { '0', 'O', 'G', 'D' } },
            };

            s1 = s1.ToUpperInvariant();
            s2 = s2.ToUpperInvariant();

            for (int i = 0; i < s1.Length; i++)
            {
                if (s1[i] == s2[i]) continue;

                if (!similar.TryGetValue(s1[i], out var alternates) || !alternates.Contains(s2[i]))
                {
                    return false;
                }
            }
            return true;
        }


        private bool FuzzyMatchPassword(string s1, string s2)
        {
            if (s1.Length != s2.Length) return false;

            Dictionary<char, char[]> similar = new Dictionary<char, char[]>
            {
                { '0', new[] { 'o', 'c' } },
                { 'O', new[] { '0' } },
                { '1', new[] { 'l', 'i' } },
                { 'l', new[] { '1', 'i' } },
                { 'i', new[] { '1', 'l', 'j' } },
                { 'j', new[] { 'i' } },
                { 'f', new[] { 't' } },
                { 't', new[] { 'f' } },
                { '5', new[] { 's' } },
                { 's', new[] { '5' } },
                { '2', new[] { 'z', 'Z' } },
                { 'z', new[] { '2' } },
                { '9', new[] { '8', 'g' } },
                { '8', new[] { 'b', 'B' } },
                { 'b', new[] { '8', 'o', 'd', 'q', 'a', '6' } },
                { 'a', new[] { 'd', 'o', 'q', 'b' } },
                { 'd', new[] { 'a', 'o', 'c', 'q' } },
                { 'c', new[] { 'e', 'o' } },
                { 'e', new[] { 'c', 'o' } },
                { 'o', new[] { 'd', 'c', 'b', 'q', '0' } },
                { 'g', new[] { '8', '9' } },
            };

            s1 = s1.ToLowerInvariant();
            s2 = s2.ToLowerInvariant();
            s1 = s1.Trim();
            s2 = s2.Trim();

            for (int i = 0; i < s1.Length; i++)
            {
                if (s1[i] == s2[i]) continue;

                if (!similar.TryGetValue(s1[i], out var alternates) || !alternates.Contains(s2[i]))
                {
                    return false;
                }
            }

            return true;
        }

        private void SaveImageByResult(Image image, bool isPass)
        {
            try
            {
                string resultDir = isPass ? "PASS" : "NG";
                string desktop = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
                string targetFolder = Path.Combine(desktop, resultDir);

                if (!Directory.Exists(targetFolder))
                    Directory.CreateDirectory(targetFolder);

                if (!string.IsNullOrEmpty(CurrentImagePath) && File.Exists(CurrentImagePath))
                {
                    string fileName = Path.GetFileName(CurrentImagePath);
                    string destPath = Path.Combine(targetFolder, fileName);
                    //image.RotateFlip(RotateFlipType.Rotate180FlipNone);
                    image.Save(destPath);

                    CleanupResultDirectory(targetFolder, destPath, 1000);
                }
            }
            catch (Exception ex)
            {
                AppendLogText($"[SAVE ERROR]: {ex.Message}", Color.Red);
            }
        }

        public void AppendLogText(string message, Color color)
        {
            if (richTextBox_Log.InvokeRequired)
            {
                richTextBox_Log.Invoke(new Action(() =>
                {
                    AppendLog(message, color);
                }));
            }
            else
            {
                AppendLog(message, color);
            }
        }

        private void AppendNewEntry(string message, Color color)
        {
            richTextBox_Log.SelectionStart = richTextBox_Log.TextLength;
            richTextBox_Log.SelectionLength = 0;
            richTextBox_Log.SelectionColor = color;
            richTextBox_Log.AppendText(message + Environment.NewLine);
            richTextBox_Log.SelectionColor = richTextBox_Log.ForeColor;
            richTextBox_Log.ScrollToCaret();

            string logDirectory = Path.Combine(Application.StartupPath, "Log");
            if (!Directory.Exists(logDirectory))
            {
                Directory.CreateDirectory(logDirectory);
            }

            string logFileName = $"XB8_Log_{DateTime.Now:ddMMyyyy}.txt";
            string logFilePath = Path.Combine(logDirectory, logFileName);
            string logEntry = $"{message}{Environment.NewLine}";
            File.AppendAllText(logFilePath, logEntry);
        }

        private void AppendLog(string message, Color color)
        {
            richTextBox_Log.SelectionStart = richTextBox_Log.TextLength;
            richTextBox_Log.SelectionLength = 0;
            richTextBox_Log.SelectionColor = color;
            richTextBox_Log.AppendText(message + Environment.NewLine);
            richTextBox_Log.SelectionColor = richTextBox_Log.ForeColor;
            richTextBox_Log.ScrollToCaret();

            string logDirectory = Path.Combine(Application.StartupPath, "Log");
            if (!Directory.Exists(logDirectory))
            {
                Directory.CreateDirectory(logDirectory);
            }

            string logFileName = $"XB8_Log_{DateTime.Now:ddMMyyyy}.txt";
            string logFilePath = Path.Combine(logDirectory, logFileName);
            string logEntry = $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {message}{Environment.NewLine}";
            File.AppendAllText(logFilePath, logEntry);
        }
        private double CalculateSimilarity(string source, string target)
        {
            source = source.ToUpperInvariant().Replace(" ", "");
            target = target.ToUpperInvariant().Replace(" ", "");

            int[,] dp = new int[source.Length + 1, target.Length + 1];
            for (int i = 0; i <= source.Length; i++) dp[i, 0] = i;
            for (int j = 0; j <= target.Length; j++) dp[0, j] = j;

            for (int i = 1; i <= source.Length; i++)
            {
                for (int j = 1; j <= target.Length; j++)
                {
                    int cost = source[i - 1] == target[j - 1] ? 0 : 1;
                    dp[i, j] = Math.Min(Math.Min(
                        dp[i - 1, j] + 1,
                        dp[i, j - 1] + 1),
                        dp[i - 1, j - 1] + cost);
                }
            }

            int maxLen = Math.Max(source.Length, target.Length);
            int distance = dp[source.Length, target.Length];
            double score = 100.0 * (1.0 - (double)distance / maxLen);
            return score;
        }

        private Bitmap CropImage(Bitmap source, Rectangle rect)
        {
            return source.Clone(rect, source.PixelFormat);
        }

        private Bitmap ConvertToGrayscale8bpp(Bitmap src)
        {
            int width = src.Width;
            int height = src.Height;
            Bitmap grayBmp = new Bitmap(width, height, PixelFormat.Format8bppIndexed);

            ColorPalette palette = grayBmp.Palette;
            for (int i = 0; i < 256; i++)
                palette.Entries[i] = Color.FromArgb(i, i, i);
            grayBmp.Palette = palette;

            BitmapData srcData = src.LockBits(new Rectangle(0, 0, width, height),
                ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            BitmapData grayData = grayBmp.LockBits(new Rectangle(0, 0, width, height),
                ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);

            unsafe
            {
                byte* srcPtr = (byte*)srcData.Scan0;
                byte* grayPtr = (byte*)grayData.Scan0;
                int srcStride = srcData.Stride;
                int grayStride = grayData.Stride;

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        byte b = srcPtr[y * srcStride + x * 3];
                        byte g = srcPtr[y * srcStride + x * 3 + 1];
                        byte r = srcPtr[y * srcStride + x * 3 + 2];
                        byte gray = (byte)((r + g + b) / 3);
                        grayPtr[y * grayStride + x] = gray;
                    }
                }
            }
            src.UnlockBits(srcData);
            grayBmp.UnlockBits(grayData);
            return grayBmp;
        }

        private string ProcessOCR_Paddle(Bitmap imBmp)
        {
            try
            {
                Bitmap convertedBmp = new Bitmap(imBmp.Width, imBmp.Height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
                using (Graphics g = Graphics.FromImage(convertedBmp))
                {
                    g.DrawImage(imBmp, 0, 0);
                }

                using (var mat = BitmapConverter.ToMat(convertedBmp))
                {
                    var result = PaddleOcrEngine.Run(mat);

                    if (result.Regions != null && result.Regions.Count() > 0)
                    {
                        string allText = string.Join("\n", result.Regions.Select(r => r.Text));
                        foreach (var region in result.Regions)
                        {
                            Console.WriteLine($"Text: {region.Text}, Score: {region.Score}");
                        }

                        return allText.Trim();
                    }
                    else
                    {
                        return "No Text Found";
                    }
                }
            }
            catch (Exception ex)
            {
                return "OCR Error: " + ex.Message;
            }
        }

        private string ProcessOCR_Tesseract(Bitmap bmp, string referenceString, string pattern)
        {
            using (var pix = PixConverter.ToPix(bmp))
            using (var page = Tesseract_Engine.Process(pix, PageSegMode.SingleBlock))
            {
                string[] lines = page.GetText()
                                 .Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

                Console.WriteLine("===== TESSERACT OCR RAW OUTPUT =====");
                for (int i = 0; i < lines.Length; i++)
                {
                    Console.WriteLine($"Line {i}: {lines[i]}");
                }

                bool isPasswordCase = pattern == @"^[a-z0-9]+$";

                string fallback = lines.Select(l => DeciderComparison(l, !isPasswordCase)).FirstOrDefault(cleaned => cleaned.Contains(DeciderComparison(referenceString, !isPasswordCase)));

                if (!string.IsNullOrEmpty(fallback))
                {
                    Console.WriteLine($"[FALLBACK] Reference match found: {fallback}");

                    var match = Regex.Match(fallback, pattern);
                    if (match.Success)
                    {
                        Console.WriteLine($"[FALLBACK RESULT] Using matched value: {match.Value}");
                        return match.Value;
                    }
                }

                Console.WriteLine("[FINAL RESULT] No valid match found.");
                return "";
            }
        }

        private string DeciderComparison(string input, bool toUpper)
        {
            string cleaned = input.Trim().Replace(" ", "");
            return toUpper ? cleaned.ToUpperInvariant() : cleaned.ToLowerInvariant();
        }

        private string ExtractValueFromOcrResult(string result, string key)
        {
            string[] lines = result.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            key = key.Replace(" ", "").ToUpperInvariant();
            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i].Trim();
                string lineNoSpace = line.Replace(" ", "").ToUpperInvariant();

                // Case 1: key:VALUE dalam satu baris
                if (lineNoSpace.StartsWith(key + ":"))
                {
                    int sepIndex = line.IndexOf(':');
                    if (sepIndex >= 0 && sepIndex < line.Length - 1)
                        return line.Substring(sepIndex + 1).Trim();
                }

                // Case 2: key di satu baris, value di baris berikutnya
                if (lineNoSpace.Contains(key))
                {
                    // Baris berikutnya
                    if (i < lines.Length - 1)
                    {
                        string nextLine = lines[i + 1].Trim();
                        if (!string.IsNullOrEmpty(nextLine) && nextLine.Length > 3)
                            return nextLine;
                    }

                    // Case 3: value ada di baris ketiga
                    if (i < lines.Length - 2)
                    {
                        string thirdLine = lines[i + 2].Trim();
                        if (!string.IsNullOrEmpty(thirdLine) && thirdLine.Length > 3)
                            return thirdLine;
                    }
                }
            }

            return "";
        }

        private void ProcessBarcodeZbar(Bitmap ImageBitmap)
        {
            Mat src = BitmapConverter.ToMat(ImageBitmap);

            Mat gray = new Mat();
            Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

            Mat binary = new Mat();
            Cv2.Threshold(gray, binary, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

            Mat morph = new Mat();
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(3, 3));
            Cv2.MorphologyEx(binary, morph, MorphTypes.Close, kernel);
            Cv2.FindContours(morph, out OpenCvSharp.Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            using (Bitmap rawBitmap = BitmapConverter.ToBitmap(gray))
            {
                Bitmap grayscaleBitmap = ConvertToGrayscale8bpp(rawBitmap);
                ZBARDECODECMD(grayscaleBitmap);
            }
        }

        private void ZBARDECODECMD(Bitmap bitmap)
        {
            try
            {
                string TempPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ZBar", "ZBarTemp");
                if (!Directory.Exists(TempPath))
                    Directory.CreateDirectory(TempPath);

                string tempFilePath = Path.Combine(TempPath, "zbar_temp.jpg");
                bitmap.Save(tempFilePath, System.Drawing.Imaging.ImageFormat.Jpeg);

                string exePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ZBar", "zbarimg.exe");

                string output = RunZbar(exePath, tempFilePath, out string error);

                if (string.IsNullOrWhiteSpace(output))
                {
                    Console.WriteLine("[INFO] ZBar gagal decode, mencoba semua kombinasi brightness & contrast...");

                    using (var mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(bitmap))
                    {
                        bool found = false;

                        // Loop kontras 1.3 to 2.0
                        for (double alpha = 1.3; alpha <= 2.0 && !found; alpha += 0.1)
                        {
                            // Loop brightness -120 to -180
                            for (double beta = -120; beta >= -180; beta -= 10)
                            {
                                Mat enhanced = new Mat();
                                mat.ConvertTo(enhanced, MatType.CV_8UC3, alpha, beta);

                                string enhancedPath = Path.Combine(TempPath, $"zbar_temp_{alpha:F1}_{beta}.jpg");
                                Cv2.ImWrite(enhancedPath, enhanced);
                                enhanced.Dispose();

                                Console.WriteLine($"[DEBUG] Percobaan: Kontras={alpha:F1}, Brightness={beta}");

                                output = RunZbar(exePath, enhancedPath, out error);

                                if (!string.IsNullOrWhiteSpace(output))
                                {
                                    Console.WriteLine($"[SUCCESS] QR berhasil didecode pada Kontras={alpha:F1}, Brightness={beta}");
                                    found = true;
                                }

                                if (File.Exists(enhancedPath))
                                    File.Delete(enhancedPath);

                                if (found) break;
                            }
                        }
                    }
                }

                // result
                if (!string.IsNullOrWhiteSpace(output))
                {
                    QRResult = output.Trim() + "\n";
                    BarcodeResult = output.Trim() + "\n";
                }
                else
                {
                    QRResult = "ZBar: No QR detected.\n";
                    BarcodeResult = "ZBar: No Barcode detected.\n";
                }

                string errorLower = error?.ToLower() ?? "";
                if (!string.IsNullOrWhiteSpace(error) &&
                    (errorLower.Contains("error") || errorLower.Contains("fail") || errorLower.Contains("missing")))
                {
                    QRResult = "ZBar error: " + error + "\n";
                    BarcodeResult = "ZBar error: " + error + "\n";
                }
            }
            catch (Exception ex)
            {
                richTextBox_Barcode.Invoke((MethodInvoker)(() =>
                {
                    richTextBox_Barcode.AppendText("Error running zbarimg: " + ex.Message + "\n");
                }));
            }
        }

        private string RunZbar(string exePath, string imagePath, out string error)
        {
            error = "";
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = exePath,
                    Arguments = "--raw \"" + imagePath + "\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using (System.Diagnostics.Process process = System.Diagnostics.Process.Start(psi))
                {
                    string output = process.StandardOutput.ReadToEnd();
                    error = process.StandardError.ReadToEnd();
                    process.WaitForExit();
                    Console.WriteLine(output);
                    return output;
                }
            }
            catch (Exception ex)
            {
                error = ex.Message;
                return "";
            }
        }

        void CheckReady()
        {
            if (isOCRReady && isScanReady)
            {
                CheckingProcess();
            }
        }

        private DenseTensor<float> BitmapToTensor(Bitmap bmp, int inputW, int inputH)
        {
            Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(bmp);

            // Resize input YOLO
            Cv2.Resize(mat, mat, new OpenCvSharp.Size(inputW, inputH));

            // BGR RGB
            Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2RGB);

            // Normalize
            mat.ConvertTo(mat, MatType.CV_32FC3, 1.0 / 255);

            float[] data = new float[1 * 3 * inputW * inputH];
            int idx = 0;

            for (int c = 0; c < 3; c++)
                for (int y = 0; y < inputH; y++)
                    for (int x = 0; x < inputW; x++)
                        data[idx++] = mat.At<Vec3f>(y, x)[c];

            return new DenseTensor<float>(data, new[] { 1, 3, inputH, inputW });
        }

        private float CalculateIoU(OpenCvSharp.Rect b1, OpenCvSharp.Rect b2)
        {
            int x1 = Math.Max(b1.X, b2.X);
            int y1 = Math.Max(b1.Y, b2.Y);
            int x2 = Math.Min(b1.X + b1.Width, b2.X + b2.Width);
            int y2 = Math.Min(b1.Y + b1.Height, b2.Y + b2.Height);

            int interWidth = Math.Max(0, x2 - x1);
            int interHeight = Math.Max(0, y2 - y1);
            float intersection = interWidth * interHeight;

            float area1 = b1.Width * b1.Height;
            float area2 = b2.Width * b2.Height;
            float union = area1 + area2 - intersection;

            return (union > 0) ? (intersection / union) : 0;
        }

        private string GetLabelName(int classId)
        {
            string[] names =
            {
                "PN",
                "FCCID",
                "LEFTSIDE",
                "BARCODE",
                "MAC",
                "SSID",
                "PASSWORD",
                "RIGHTSIDE",
                "QR",
                "LOGOAREA"
            };

            if (classId >= 0 && classId < names.Length)
                return names[classId];

            return "UNKNOWN";
        }



        private void ProcessImage(string fullPath)
        {
            ResetBoxCoordinates();

            if (SelectModel != 1)
            {
                MessageBox.Show("Please select the correct model first!");
                return;
            }

            try
            {
                bool isPNReady = false;
                bool isFCCIDReady = false;
                bool isCMMACReady = false;
                bool isWANMACReady = false;
                bool isMTAMACReady = false;
                bool isNETWORKReady = false;
                bool isPasswordReady = false;
                bool isLeftLabelReady = false;
                bool isRightLabelReady = false;

                PN_TextBox.Text = "";
                FCCID_TextBox.Text = "";
                OCRCMMAC_TextBox.Text = "";
                OCRWANMAC_TextBox.Text = "";
                OCRMTAMAC_TextBox.Text = "";
                NETWORKNAME_TextBox.Text = "";
                PASSWORD_TextBox.Text = "";

                GVL.OCR_PN = "";
                GVL.OCR_FCCID = "";
                GVL.OCR_CM_MAC = "";
                GVL.OCR_WAN_MAC = "";
                GVL.OCR_MTA_MAC = "";
                GVL.OCR_NetworkName = "";
                GVL.OCR_Password = "";
                GVL.OCR_LastFourNetworkName = "";

                Bitmap imageBitmap = new Bitmap(fullPath);

                // ROTATION
                if (ImageRotation == 90)
                {
                    imageBitmap.RotateFlip(RotateFlipType.Rotate90FlipNone);
                }
                else if (ImageRotation == 180)
                {
                    imageBitmap.RotateFlip(RotateFlipType.Rotate180FlipNone);
                }    
                else if (ImageRotation == 270)
                {
                    imageBitmap.RotateFlip(RotateFlipType.Rotate270FlipNone);
                }

                SourcePicture.Image?.Dispose();
                SourcePicture.Image = new Bitmap(imageBitmap);
                SourcePicture.Refresh();

                const int INPUT_SIZE = 640;
                const float CONF_TH = 0.5f;
                const float NMS_TH = 0.45f;

                var inputTensor = BitmapToTensor(imageBitmap, INPUT_SIZE, INPUT_SIZE);

                var results = _session.Run(new[]
                {
    NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
});

                var output = results.First().AsTensor<float>();
                var dims = output.Dimensions.ToArray(); // expected [1,84,8400]

                int rows = dims[1];
                int cols = dims[2];

                float scaleX = (float)imageBitmap.Width / INPUT_SIZE;
                float scaleY = (float)imageBitmap.Height / INPUT_SIZE;

                var candidates = new List<(OpenCvSharp.Rect Box, float Score, int ClassId)>();

                for (int i = 0; i < cols; i++)
                {
                    float maxScore = 0;
                    int classId = 0;

                    for (int c = 4; c < rows; c++)
                    {
                        if (output[0, c, i] > maxScore)
                        {
                            maxScore = output[0, c, i];
                            classId = c - 4;
                        }
                    }

                    if (maxScore > CONF_TH)
                    {
                        float cx = output[0, 0, i] * scaleX;
                        float cy = output[0, 1, i] * scaleY;
                        float w = output[0, 2, i] * scaleX;
                        float h = output[0, 3, i] * scaleY;

                        int x = (int)(cx - w / 2);
                        int y = (int)(cy - h / 2);

                        candidates.Add((new OpenCvSharp.Rect(x, y, (int)w, (int)h), maxScore, classId));
                    }
                }

                // NMS
                var sorted = candidates.OrderByDescending(c => c.Score).ToList();
                var finalResults = new List<(OpenCvSharp.Rect Box, float Score, int ClassId)>();

                while (sorted.Count > 0)
                {
                    var current = sorted[0];
                    finalResults.Add(current);
                    sorted.RemoveAt(0);

                    for (int i = sorted.Count - 1; i >= 0; i--)
                    {
                        if (CalculateIoU(current.Box, sorted[i].Box) > NMS_TH)
                            sorted.RemoveAt(i);
                    }
                }


                foreach (var res in finalResults)
                {
                    string label = GetLabelName(res.ClassId);

                    switch (label)
                    {
                        case "MAC":
                            GVL.StartXMAC = res.Box.X;
                            GVL.StartYMAC = res.Box.Y;
                            GVL.WidthMAC = res.Box.Width;
                            GVL.HeightMAC = res.Box.Height;
                            break;

                        case "SSID":
                            GVL.StartXSSID = res.Box.X;
                            GVL.StartYSSID = res.Box.Y;
                            GVL.WidthSSID = res.Box.Width;
                            GVL.HeightSSID = res.Box.Height;
                            break;

                        case "PASSWORD":
                            GVL.StartXPASSWORD = res.Box.X;
                            GVL.StartYPASSWORD = res.Box.Y;
                            GVL.WidthPASSWORD = res.Box.Width;
                            GVL.HeightPASSWORD = res.Box.Height;
                            break;

                        case "QR":
                            GVL.StartXQR = res.Box.X;
                            GVL.StartYQR = res.Box.Y;
                            GVL.WidthQR = res.Box.Width;
                            GVL.HeightQR = res.Box.Height;
                            break;

                        case "BARCODE":
                            GVL.StartXBARCODE = res.Box.X;
                            GVL.StartYBARCODE = res.Box.Y;
                            GVL.WidthBARCODE = res.Box.Width;
                            GVL.HeightBARCODE = res.Box.Height;
                            break;

                        case "PN":
                            GVL.StartXPN = res.Box.X;
                            GVL.StartYPN = res.Box.Y;
                            GVL.WidthPN = res.Box.Width;
                            GVL.HeightPN = res.Box.Height;
                            break;

                        case "FCCID":
                            GVL.StartXFCCID = res.Box.X;
                            GVL.StartYFCCID = res.Box.Y;
                            GVL.WidthFCCID = res.Box.Width;
                            GVL.HeightFCCID = res.Box.Height;
                            break;

                        case "LEFTSIDE":
                            GVL.StartXLEFTSIDE = res.Box.X;
                            GVL.StartYLEFTSIDE = res.Box.Y;
                            GVL.WidthLEFTSIDE = res.Box.Width;
                            GVL.HeightLEFTSIDE = res.Box.Height;
                            break;

                        case "RIGHTSIDE":
                            GVL.StartXRIGHTSIDE = res.Box.X;
                            GVL.StartYRIGHTSIDE = res.Box.Y;
                            GVL.WidthRIGHTSIDE = res.Box.Width;
                            GVL.HeightRIGHTSIDE = res.Box.Height;
                            break;

                        case "LOGOAREA":
                            GVL.StartXLOGOAREA = res.Box.X;
                            GVL.StartYLOGOAREA = res.Box.Y;
                            GVL.WidthLOGOAREA = res.Box.Width;
                            GVL.HeightLOGOAREA = res.Box.Height;
                            GVL.IsLogoDetected = true;
                            break;
                    }
                }


                Dictionary<string, Rectangle> regions = new Dictionary<string, Rectangle>
                {
                    { "PN", new Rectangle(GVL.StartXPN, GVL.StartYPN, GVL.WidthPN, GVL.HeightPN) },
                    { "FCCID", new Rectangle(GVL.StartXFCCID, GVL.StartYFCCID, GVL.WidthFCCID, GVL.HeightFCCID) },
                    { "MAC", new Rectangle(GVL.StartXMAC, GVL.StartYMAC, GVL.WidthMAC, GVL.HeightMAC) },
                    { "QR", new Rectangle(GVL.StartXQR, GVL.StartYQR, GVL.WidthQR, GVL.HeightQR) },
                    { "BARCODE", new Rectangle(GVL.StartXBARCODE, GVL.StartYBARCODE, GVL.WidthBARCODE, GVL.HeightBARCODE) },
                    { "SSID", new Rectangle(GVL.StartXSSID, GVL.StartYSSID, GVL.WidthSSID, GVL.HeightSSID) },
                    { "PASSWORD", new Rectangle(GVL.StartXPASSWORD, GVL.StartYPASSWORD, GVL.WidthPASSWORD, GVL.HeightPASSWORD) },
                    { "LEFTSIDE", new Rectangle(GVL.StartXLEFTSIDE, GVL.StartYLEFTSIDE, GVL.WidthLEFTSIDE, GVL.HeightLEFTSIDE) },
                    { "RIGHTSIDE", new Rectangle(GVL.StartXRIGHTSIDE, GVL.StartYRIGHTSIDE, GVL.WidthRIGHTSIDE, GVL.HeightRIGHTSIDE) }
                };

                richTextBox_Barcode.Clear();

                foreach (var region in regions)
                {
                    if (region.Value.Width <= 0 || region.Value.Height <= 0)
                        continue;

                    Bitmap cropped = CropImage(new Bitmap(imageBitmap), region.Value);

                    if (region.Key == "QR")
                    {
                        QRPicture.Image = cropped;
                        ProcessBarcodeZbar(cropped);
                        richTextBox_Barcode.AppendText($"{QRResult}\n");
                    }
                    else if (region.Key == "BARCODE")
                    {
                        BarcodePicture.Image = cropped;
                        ProcessBarcodeZbar(cropped);
                        richTextBox_Barcode.AppendText($"{BarcodeResult}\n");
                    }
                    else if (region.Key == "SSID")
                    {
                        SSIDPicture.Image = cropped;
                        string result = ProcessOCR_Paddle(cropped);
                        string value = ExtractValueFromOcrResult(result, "SSID");

                        if (!string.IsNullOrEmpty(value))
                        {
                            NETWORKNAME_TextBox.Text = value;
                            GVL.OCR_NetworkName = value;

                            string[] parts = value.Split('-');
                            if (parts.Length > 1)
                            {
                                string lastPart = parts.Last().Trim();
                                if (lastPart.Length >= 4)
                                    GVL.OCR_LastFourNetworkName = lastPart.Substring(lastPart.Length - 4);
                                else
                                    GVL.OCR_LastFourNetworkName = lastPart;
                            }
                        }
                        isNETWORKReady = true;
                    }
                    else if (region.Key == "PASSWORD")
                    {
                        PasswordPicture.Image = cropped;
                        string result = ProcessOCR_Paddle(cropped);
                        string value = ExtractValueFromOcrResult(result, "WORD");

                        if (!string.IsNullOrEmpty(value))
                        {
                            PASSWORD_TextBox.Text = value;
                            GVL.OCR_Password = value;
                        }
                        isPasswordReady = true;
                    }
                    else if (region.Key == "PN")
                    {
                        Mat matCropped = BitmapConverter.ToMat(cropped);
                        Mat enhanced = new Mat();
                        matCropped.ConvertTo(enhanced, MatType.CV_8UC3, alpha: 1.5, beta: 7); // alpha=kontras, beta=brightness

                        Bitmap enhancedBitmap = BitmapConverter.ToBitmap(enhanced);
                        ModelPicture.Image = enhancedBitmap;
                        string result = ProcessOCR_Paddle(enhancedBitmap);
                        string[] lines = result.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

                        foreach (var line in lines)
                        {
                            string cleanLine = line.Trim()
                                .Replace(" ", "")
                                .Replace(".", "")
                                .Replace("49B1", "4981")
                                .Replace("498I", "4981")
                                .Replace("49BI", "4981")
                                .Replace("4881", "4981")
                                .Replace("4B81", "4981")
                                .Replace("4BB1", "4981")
                                .Replace("48BI", "4981")
                                .Replace("4BBI", "4981")
                                .Replace("C0M", "COM")
                                .ToUpperInvariant();

                            var match = Regex.Match(cleanLine, @"PN[:-]?([A-Z0-9]+)");
                            if (match.Success && match.Groups.Count > 1)
                            {
                                string value = match.Groups[1].Value;
                                GVL.OCR_PN = value;
                                PN_TextBox.Text = value;
                                break;
                            }
                        }
                        isPNReady = true;
                    }
                    else if (region.Key == "FCCID")
                    {
                        Mat matCropped = BitmapConverter.ToMat(cropped);
                        Mat enhanced = new Mat();
                        matCropped.ConvertTo(enhanced, MatType.CV_8UC3, alpha: 1.5, beta: 7); // alpha=kontras, beta=brightness

                        Bitmap enhancedBitmap = BitmapConverter.ToBitmap(enhanced);
                        FCCIDPicture.Image = enhancedBitmap;
                        string result = ProcessOCR_Paddle(enhancedBitmap);
                        string[] lines = result.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

                        foreach (var line in lines)
                        {
                            string cleanLine = line.Trim()
                                .Replace(".", "")
                                .Replace("49B1", "4981")
                                .Replace("498I", "4981")
                                .Replace("49BI", "4981")
                                .Replace("4881", "4981")
                                .Replace("4B81", "4981")
                                .Replace("4BB1", "4981")
                                .Replace("48BI", "4981")
                                .Replace("4BBI", "4981")
                                .Replace(" ", "")
                                .ToUpperInvariant();
                            var match = Regex.Match(cleanLine, @"FCC[I1Jl][DO0][:]?([A-Z0-9]+)");
                            if (match.Success && match.Groups.Count > 1)
                            {
                                string value = match.Groups[1].Value;
                                GVL.OCR_FCCID = value;
                                FCCID_TextBox.Text = value;
                                break;
                            }
                        }
                        isFCCIDReady = true;
                    }
                    else if (region.Key == "MAC")
                    {
                        MACPicture.Image = cropped;
                        string result = ProcessOCR_Paddle(cropped);

                        var validLines = result
                            .Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                            .Select(l => l.Trim().ToUpperInvariant())
                            .Where(l => !string.IsNullOrWhiteSpace(l) && l != ",")
                            .ToList();

                        var matches = validLines
                            .SelectMany(l => Regex.Matches(l, @"[A-Z0-9]{12}").Cast<Match>())
                            .Select(m => m.Value)
                            .ToList();

                        if (matches.Count >= 3)
                        {
                            GVL.OCR_CM_MAC = matches[0];
                            GVL.OCR_WAN_MAC = matches[1];
                            GVL.OCR_MTA_MAC = matches[2];

                            OCRCMMAC_TextBox.Text = matches[0];
                            OCRWANMAC_TextBox.Text = matches[1];
                            OCRMTAMAC_TextBox.Text = matches[2];
                        }
                        else
                        {
                            if (matches.Count > 0) { GVL.OCR_CM_MAC = matches[0]; OCRCMMAC_TextBox.Text = matches[0]; } else { OCRCMMAC_TextBox.Text = "Not Found"; }
                            if (matches.Count > 1) { GVL.OCR_WAN_MAC = matches[1]; OCRWANMAC_TextBox.Text = matches[1]; } else { OCRWANMAC_TextBox.Text = "Not Found"; }
                            if (matches.Count > 2) { GVL.OCR_MTA_MAC = matches[2]; OCRMTAMAC_TextBox.Text = matches[2]; } else { OCRMTAMAC_TextBox.Text = "Not Found"; }
                        }

                        isCMMACReady = true;
                        isWANMACReady = true;
                        isMTAMACReady = true;
                    }
                    else if (region.Key == "LEFTSIDE")
                    {
                        Mat matCropped = BitmapConverter.ToMat(cropped);
                        Mat enhanced = new Mat();
                        matCropped.ConvertTo(enhanced, MatType.CV_8UC3, alpha: 1.5, beta: 10); // alpha=kontras, beta=brightness

                        Bitmap enhancedBitmap = BitmapConverter.ToBitmap(enhanced);

                        LeftPicture.Image = enhancedBitmap;
                        string result = ProcessOCR_Paddle(enhancedBitmap);


                        if (!string.IsNullOrWhiteSpace(result))
                        {
                            string referenceText = "INPUT120V46AFCCREGULATIONSRESTRICTTOINDOORUSEONLY";

                            if (result.StartsWith("nput") || result.StartsWith("1nput") || result.StartsWith("lnput"))
                            {
                                result = result
                                    .Replace("nput", "Input")
                                    .Replace("1nput", "Input")
                                    .Replace("lnput", "Input");
                            }

                            string cleanedResult = result
                                .Replace("\n", " ")
                                .Replace("\r", " ")
                                .Replace(":", " ")
                                .Replace(".", " ")
                                .Replace("-", " ")
                                .Replace("OV", "0V")
                                .Replace("rogulatlons", "regulations")
                                .Replace("roguiatlons", "regulations")
                                .Replace("roguiations", "regulations")
                                .Replace("regulatlons", "regulations")
                                .Replace("reguiatlons", "regulations")
                                .Replace("reguiations", "regulations")
                                .Replace("restrlct", "restrict")
                                .Replace("resfrlct", "restrict")
                                .Replace("resfrict", "restrict")
                                .Replace("inaoor", "indoor")
                                .Replace("Inaoor", "indoor")
                                .Replace("lnaoor", "indoor")
                                .Replace("lndoor", "indoor")
                                .Replace("1ndoor", "indoor")
                                .Replace("1naoor", "indoor")
                                .Replace("onty", "only")
                                .Replace("onfy", "only")
                                .Replace("oniy", "only")
                                .Replace("onjy", "only")
                                .Replace("onIy", "only")
                                .Replace("  ", " ")
                                .Trim()
                                .ToUpperInvariant()
                                .Replace(" ", "");

                            string cleanedReference = referenceText
                                .Replace(" ", "")
                                .ToUpperInvariant();

                            double matchScore = CalculateSimilarity(cleanedResult, cleanedReference);
                            LeftLabelScore = matchScore;

                            Console.WriteLine($"[Left Label OCR]: {result}");
                            Console.WriteLine($"[Cleaned]: {cleanedResult}");
                            Console.WriteLine($"[LeftLabel Match Score] {matchScore:0.0}%");

                        }
                        isLeftLabelReady = true;
                    }
                    else if (region.Key == "RIGHTSIDE")
                    {
                        RightPicture.Image = cropped;
                        string result = ProcessOCR_Paddle(cropped);

                        if (!string.IsNullOrWhiteSpace(result))
                        {
                            string referenceText = "PROPERTYOFCOMCASTNOTFORRESALEREMOVINGTHISSTICKERORTAMPERINGWITHTHISEQUIPMENTISAVIOLATIONOFFEDERALANDSTATELAW";

                            string cleanedResult = result
                                .Replace("\n", " ")
                                .Replace("\r", " ")
                                .Replace("  ", " ")
                                .Replace(".", " ")
                                .Replace("OE", "OF")
                                .Replace("0", "O")
                                .Replace("4", "A")
                                .Replace("l", "I")
                                
                                .Trim()
                                .ToUpperInvariant()
                                .Replace(" ", "");

                            string cleanedReference = referenceText
                                .Replace(" ", "")
                                .ToUpperInvariant();

                            double matchScore = CalculateSimilarity(cleanedResult, cleanedReference);
                            RightLabelScore = matchScore;

                            Console.WriteLine($"[Right Label OCR]: {result}");
                            Console.WriteLine($"[Cleaned]: {cleanedResult}");
                            Console.WriteLine($"[RightLabel Match Score] {matchScore:0.0}%");

                        }
                        isRightLabelReady = true;
                    }
                }

                if (isPNReady && isFCCIDReady && isCMMACReady && isWANMACReady &&
                    isMTAMACReady && isNETWORKReady && isPasswordReady &&
                    isLeftLabelReady && isRightLabelReady)
                {
                    isOCRReady = true;
                }
                else
                {
                    CC.WriteCoilMemory(NGCoil, true);
                    string time = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}";
                    AppendNewEntry($"[{time}] ===NEW ENTRY===", Color.Black);
                    AppendLogText("Some target is not detected, please adjust image quality and try again!", Color.Red);
                    this.Invoke(new Action(() =>
                    {
                        LabelResult.Text = "NG";
                        LabelResult.ForeColor = Color.Red;
                    }));
                    Thread.Sleep(200);
                    CC.WriteCoilMemory(NGCoil, false);
                    if (CC.ErrorCode == 1)
                    {
                        AppendLogText(CC.ErrorString, Color.Red);
                        CC.ErrorCode = 0;
                    }
                    SaveImageByResult(SourcePicture.Image, false);
                    UpdateCounter(false);
                }

                CheckReady();
            }
            catch (Exception ex)
            {
                CC.WriteCoilMemory(NGCoil, true);
                string time = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}";
                AppendNewEntry($"[{time}] ===NEW ENTRY===", Color.Black);
                AppendLogText("Error ocured" + ex.Message, Color.Red);
                this.Invoke(new Action(() =>
                {
                    LabelResult.Text = "NG";
                    LabelResult.ForeColor = Color.Red;
                }));
                Thread.Sleep(200);
                CC.WriteCoilMemory(NGCoil, false);
                if (CC.ErrorCode == 1)
                {
                    AppendLogText(CC.ErrorString, Color.Red);
                    CC.ErrorCode = 0;
                }
                SaveImageByResult(SourcePicture.Image, false);
                UpdateCounter(false);
            }
        }

        private void richTextBox_Barcode_TextChanged(object sender, EventArgs e)
        {
            if (richTextBox_Barcode.Text.Contains("ZBar: No QR detected.") || richTextBox_Barcode.Text.Contains("ZBar: No Barcode detected.") || richTextBox_Barcode.Text.Contains("ZBar error:"))
            {
                SN_TextBox.Text = "";
                QRCMMAC_TextBox.Text = "";
                QRWANMAC_TextBox.Text = "";
                QRMTAMAC_TextBox.Text = "";
                QRCOMCAST_TextBox.Text = "";
                QRTECHNICOLOR_TextBox.Text = "";
                isScanReady = true;
                if (richTextBox_Barcode.Text.Contains("No QR detected"))
                {
                    isPassQR = false;
                }
                else if (richTextBox_Barcode.Text.Contains("No Barcode detected"))
                {
                    isPassBarcode = false;
                }
            }
            else
            {
                string[] lines = richTextBox_Barcode.Lines;
                List<string> result = new List<string>();

                foreach (string line in lines)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        result.Add(line.Trim());
                    }
                }

                if (result.Count > 1)
                {
                    string firstLine = result[0];
                    string[] parts = firstLine.Split('.');

                    string ComcastModel = parts[0];
                    string TechnicolorModel = parts[1];
                    string SN = parts[2];
                    string CM_MAC = parts[3];
                    string MTA_MAC = parts[4].Substring(2);
                    string WAN_MAC = parts[5].Substring(2);
                    Thread.Sleep(30);
                    SN_TextBox.Text = SN;
                    QRCOMCAST_TextBox.Text = ComcastModel;
                    QRTECHNICOLOR_TextBox.Text = TechnicolorModel;
                    QRCMMAC_TextBox.Text = CM_MAC;
                    QRWANMAC_TextBox.Text = WAN_MAC;
                    QRMTAMAC_TextBox.Text = MTA_MAC;
                    Thread.Sleep(10);
                    GVL.QR_SN = SN;
                    GVL.QR_ComcastModel = ComcastModel;
                    GVL.QR_TechnicolorModel = TechnicolorModel;
                    GVL.QR_CM_MAC = CM_MAC;
                    GVL.QR_WAN_MAC = WAN_MAC;
                    GVL.QR_MTA_MAC = MTA_MAC;
                    isScanReady = true;
                    isPassQR = true;
                    isPassBarcode = true;
                }
            }
        }

        private void File_btn_Click(object sender, EventArgs e)
        {
            if (SelectModel != 1)
            {
                MessageBox.Show("Please select the correct model first!");
                return;
            }

            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Image Files|*.png;*.jpg;*.jpeg;*.bmp;*.gif";

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                ResetBoxCoordinates();

                try
                {
                    bool isPNReady = false;
                    bool isFCCIDReady = false;
                    bool isCMMACReady = false;
                    bool isWANMACReady = false;
                    bool isMTAMACReady = false;
                    bool isNETWORKReady = false;
                    bool isPasswordReady = false;
                    bool isLeftLabelReady = false;
                    bool isRightLabelReady = false;

                    PN_TextBox.Text = "";
                    FCCID_TextBox.Text = "";
                    OCRCMMAC_TextBox.Text = "";
                    OCRWANMAC_TextBox.Text = "";
                    OCRMTAMAC_TextBox.Text = "";
                    NETWORKNAME_TextBox.Text = "";
                    PASSWORD_TextBox.Text = "";

                    GVL.OCR_PN = "";
                    GVL.OCR_FCCID = "";
                    GVL.OCR_CM_MAC = "";
                    GVL.OCR_WAN_MAC = "";
                    GVL.OCR_MTA_MAC = "";
                    GVL.OCR_NetworkName = "";
                    GVL.OCR_Password = "";
                    GVL.OCR_LastFourNetworkName = "";

                    string ImagePath = openFileDialog.FileName;
                    Bitmap imageBitmap = new Bitmap(ImagePath);

                    // ROTATION
                    if (ImageRotation == 90)
                    {
                        imageBitmap.RotateFlip(RotateFlipType.Rotate90FlipNone);
                    }
                    else if (ImageRotation == 180)
                    {
                        imageBitmap.RotateFlip(RotateFlipType.Rotate180FlipNone);
                    }
                    else if (ImageRotation == 270)
                    {
                        imageBitmap.RotateFlip(RotateFlipType.Rotate270FlipNone);
                    }

                    SourcePicture.Image?.Dispose();
                    SourcePicture.Image = new Bitmap(imageBitmap);
                    SourcePicture.Refresh();
                    
                    const int INPUT_SIZE = 640;
                    const float CONF_TH = 0.5f;
                    const float NMS_TH = 0.45f;

                    var inputTensor = BitmapToTensor(imageBitmap, INPUT_SIZE, INPUT_SIZE);

                    var results = _session.Run(new[]
                    {
    NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
});

                    var output = results.First().AsTensor<float>();
                    var dims = output.Dimensions.ToArray(); // expected [1,84,8400]

                    int rows = dims[1];
                    int cols = dims[2];

                    float scaleX = (float)imageBitmap.Width / INPUT_SIZE;
                    float scaleY = (float)imageBitmap.Height / INPUT_SIZE;

                    var candidates = new List<(OpenCvSharp.Rect Box, float Score, int ClassId)>();

                    for (int i = 0; i < cols; i++)
                    {
                        float maxScore = 0;
                        int classId = 0;

                        for (int c = 4; c < rows; c++)
                        {
                            if (output[0, c, i] > maxScore)
                            {
                                maxScore = output[0, c, i];
                                classId = c - 4;
                            }
                        }

                        if (maxScore > CONF_TH)
                        {
                            float cx = output[0, 0, i] * scaleX;
                            float cy = output[0, 1, i] * scaleY;
                            float w = output[0, 2, i] * scaleX;
                            float h = output[0, 3, i] * scaleY;

                            int x = (int)(cx - w / 2);
                            int y = (int)(cy - h / 2);

                            candidates.Add((new OpenCvSharp.Rect(x, y, (int)w, (int)h), maxScore, classId));
                        }
                    }

                    // NMS
                    var sorted = candidates.OrderByDescending(c => c.Score).ToList();
                    var finalResults = new List<(OpenCvSharp.Rect Box, float Score, int ClassId)>();

                    while (sorted.Count > 0)
                    {
                        var current = sorted[0];
                        finalResults.Add(current);
                        sorted.RemoveAt(0);

                        for (int i = sorted.Count - 1; i >= 0; i--)
                        {
                            if (CalculateIoU(current.Box, sorted[i].Box) > NMS_TH)
                                sorted.RemoveAt(i);
                        }
                    }

                    foreach (var res in finalResults)
                    {
                        string label = GetLabelName(res.ClassId);

                        switch (label)
                        {
                            case "MAC":
                                GVL.StartXMAC = res.Box.X;
                                GVL.StartYMAC = res.Box.Y;
                                GVL.WidthMAC = res.Box.Width;
                                GVL.HeightMAC = res.Box.Height;
                                break;

                            case "SSID":
                                GVL.StartXSSID = res.Box.X;
                                GVL.StartYSSID = res.Box.Y;
                                GVL.WidthSSID = res.Box.Width;
                                GVL.HeightSSID = res.Box.Height;
                                break;

                            case "PASSWORD":
                                GVL.StartXPASSWORD = res.Box.X;
                                GVL.StartYPASSWORD = res.Box.Y;
                                GVL.WidthPASSWORD = res.Box.Width;
                                GVL.HeightPASSWORD = res.Box.Height;
                                break;

                            case "QR":
                                GVL.StartXQR = res.Box.X;
                                GVL.StartYQR = res.Box.Y;
                                GVL.WidthQR = res.Box.Width;
                                GVL.HeightQR = res.Box.Height;
                                break;

                            case "BARCODE":
                                GVL.StartXBARCODE = res.Box.X;
                                GVL.StartYBARCODE = res.Box.Y;
                                GVL.WidthBARCODE = res.Box.Width;
                                GVL.HeightBARCODE = res.Box.Height;
                                break;

                            case "PN":
                                GVL.StartXPN = res.Box.X;
                                GVL.StartYPN = res.Box.Y;
                                GVL.WidthPN = res.Box.Width;
                                GVL.HeightPN = res.Box.Height;
                                break;

                            case "FCCID":
                                GVL.StartXFCCID = res.Box.X;
                                GVL.StartYFCCID = res.Box.Y;
                                GVL.WidthFCCID = res.Box.Width;
                                GVL.HeightFCCID = res.Box.Height;
                                break;

                            case "LEFTSIDE":
                                GVL.StartXLEFTSIDE = res.Box.X;
                                GVL.StartYLEFTSIDE = res.Box.Y;
                                GVL.WidthLEFTSIDE = res.Box.Width;
                                GVL.HeightLEFTSIDE = res.Box.Height;
                                break;

                            case "RIGHTSIDE":
                                GVL.StartXRIGHTSIDE = res.Box.X;
                                GVL.StartYRIGHTSIDE = res.Box.Y;
                                GVL.WidthRIGHTSIDE = res.Box.Width;
                                GVL.HeightRIGHTSIDE = res.Box.Height;
                                break;

                            case "LOGOAREA":
                                GVL.StartXLOGOAREA = res.Box.X;
                                GVL.StartYLOGOAREA = res.Box.Y;
                                GVL.WidthLOGOAREA = res.Box.Width;
                                GVL.HeightLOGOAREA = res.Box.Height;
                                GVL.IsLogoDetected = true;
                                break;
                        }
                    }

                    // CROPPING + OCR
                    Dictionary<string, Rectangle> regions = new Dictionary<string, Rectangle>
                    {
                        { "PN", new Rectangle(GVL.StartXPN, GVL.StartYPN, GVL.WidthPN, GVL.HeightPN) },
                        { "FCCID", new Rectangle(GVL.StartXFCCID, GVL.StartYFCCID, GVL.WidthFCCID, GVL.HeightFCCID) },
                        { "MAC", new Rectangle(GVL.StartXMAC, GVL.StartYMAC, GVL.WidthMAC, GVL.HeightMAC) },
                        { "QR", new Rectangle(GVL.StartXQR, GVL.StartYQR, GVL.WidthQR, GVL.HeightQR) },
                        { "BARCODE", new Rectangle(GVL.StartXBARCODE, GVL.StartYBARCODE, GVL.WidthBARCODE, GVL.HeightBARCODE) },
                        { "SSID", new Rectangle(GVL.StartXSSID, GVL.StartYSSID, GVL.WidthSSID, GVL.HeightSSID) },
                        { "PASSWORD", new Rectangle(GVL.StartXPASSWORD, GVL.StartYPASSWORD, GVL.WidthPASSWORD, GVL.HeightPASSWORD) },
                        { "LEFTSIDE", new Rectangle(GVL.StartXLEFTSIDE, GVL.StartYLEFTSIDE, GVL.WidthLEFTSIDE, GVL.HeightLEFTSIDE) },
                        { "RIGHTSIDE", new Rectangle(GVL.StartXRIGHTSIDE, GVL.StartYRIGHTSIDE, GVL.WidthRIGHTSIDE, GVL.HeightRIGHTSIDE) }
                    };

                    richTextBox_Barcode.Clear();

                    foreach (var region in regions)
                    {
                        if (region.Value.Width <= 0 || region.Value.Height <= 0)
                            continue;

                        Bitmap cropped = CropImage(new Bitmap(imageBitmap), region.Value);

                        if (region.Key == "QR")
                        {
                            QRPicture.Image = cropped;
                            ProcessBarcodeZbar(cropped);
                            richTextBox_Barcode.AppendText($"{QRResult}\n");
                        }
                        else if (region.Key == "BARCODE")
                        {
                            BarcodePicture.Image = cropped;
                            ProcessBarcodeZbar(cropped);
                            richTextBox_Barcode.AppendText($"{BarcodeResult}\n");
                        }
                        else if (region.Key == "SSID")
                        {
                            SSIDPicture.Image = cropped;
                            string result = ProcessOCR_Paddle(cropped);
                            string value = ExtractValueFromOcrResult(result, "SSID");

                            if (!string.IsNullOrEmpty(value))
                            {
                                NETWORKNAME_TextBox.Text = value;
                                GVL.OCR_NetworkName = value;

                                string[] parts = value.Split('-');
                                if (parts.Length > 1)
                                {
                                    string lastPart = parts.Last().Trim();
                                    if (lastPart.Length >= 4)
                                        GVL.OCR_LastFourNetworkName = lastPart.Substring(lastPart.Length - 4);
                                    else
                                        GVL.OCR_LastFourNetworkName = lastPart;
                                }
                            }
                            isNETWORKReady = true;
                        }
                        else if (region.Key == "PASSWORD")
                        {
                            PasswordPicture.Image = cropped;
                            string result = ProcessOCR_Paddle(cropped);
                            string value = ExtractValueFromOcrResult(result, "WORD");

                            if (!string.IsNullOrEmpty(value))
                            {
                                PASSWORD_TextBox.Text = value;
                                GVL.OCR_Password = value;
                            }
                            isPasswordReady = true;
                        }
                        else if (region.Key == "PN")
                        {
                            Mat matCropped = BitmapConverter.ToMat(cropped);
                            Mat enhanced = new Mat();
                            matCropped.ConvertTo(enhanced, MatType.CV_8UC3, alpha: 1.5, beta: 7); // alpha=kontras, beta=brightness

                            Bitmap enhancedBitmap = BitmapConverter.ToBitmap(enhanced);
                            ModelPicture.Image = enhancedBitmap;
                            string result = ProcessOCR_Paddle(enhancedBitmap);
                            string[] lines = result.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

                            foreach (var line in lines)
                            {
                                string cleanLine = line.Trim()
                                    .Replace(" ", "")
                                    .Replace(".", "")
                                    .Replace("49B1", "4981")
                                    .Replace("498I", "4981")
                                    .Replace("49BI", "4981")
                                    .Replace("4881", "4981")
                                    .Replace("4B81", "4981")
                                    .Replace("4BB1", "4981")
                                    .Replace("48BI", "4981")
                                    .Replace("4BBI", "4981")
                                    .Replace("C0M", "COM")
                                    .ToUpperInvariant();

                                var match = Regex.Match(cleanLine, @"PN[:-]?([A-Z0-9]{6,})");
                                Console.WriteLine(cleanLine);
                                if (match.Success && match.Groups.Count > 1)
                                {
                                    string value = match.Groups[1].Value;
                                    GVL.OCR_PN = value;
                                    PN_TextBox.Text = value;
                                    break;
                                }
                            }
                            isPNReady = true;
                        }
                        else if (region.Key == "FCCID")
                        {
                            Mat matCropped = BitmapConverter.ToMat(cropped);
                            Mat enhanced = new Mat();
                            matCropped.ConvertTo(enhanced, MatType.CV_8UC3, alpha: 1.5, beta: 7); // alpha=kontras, beta=brightness

                            Bitmap enhancedBitmap = BitmapConverter.ToBitmap(enhanced);
                            FCCIDPicture.Image = enhancedBitmap;
                            string result = ProcessOCR_Paddle(enhancedBitmap);
                            string[] lines = result.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

                            foreach (var line in lines)
                            {
                                string cleanLine = line.Trim()
                                    .Replace(".", "")
                                    .Replace("49B1", "4981")
                                    .Replace("498I", "4981")
                                    .Replace("49BI", "4981")
                                    .Replace("4881", "4981")
                                    .Replace("4B81", "4981")
                                    .Replace("4BB1", "4981")
                                    .Replace("48BI", "4981")
                                    .Replace("4BBI", "4981")
                                    .Replace(" ", "")
                                    .ToUpperInvariant();
                                var match = Regex.Match(cleanLine, @"FCC[I1Jl][DO0][:]?([A-Z0-9]+)");
                                if (match.Success && match.Groups.Count > 1)
                                {
                                    string value = match.Groups[1].Value;
                                    GVL.OCR_FCCID = value;
                                    FCCID_TextBox.Text = value;
                                    break;
                                }
                            }
                            isFCCIDReady = true;
                        }
                        else if (region.Key == "MAC")
                        {
                            MACPicture.Image = cropped;
                            string result = ProcessOCR_Paddle(cropped);

                            var validLines = result
                                .Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                                .Select(l => l.Trim().ToUpperInvariant())
                                .Where(l => !string.IsNullOrWhiteSpace(l) && l != ",")
                                .ToList();

                            var matches = validLines
                                .SelectMany(l => Regex.Matches(l, @"[A-Z0-9]{12}").Cast<Match>())
                                .Select(m => m.Value)
                                .ToList();

                            if (matches.Count >= 3)
                            {
                                GVL.OCR_CM_MAC = matches[0];
                                GVL.OCR_WAN_MAC = matches[1];
                                GVL.OCR_MTA_MAC = matches[2];

                                OCRCMMAC_TextBox.Text = matches[0];
                                OCRWANMAC_TextBox.Text = matches[1];
                                OCRMTAMAC_TextBox.Text = matches[2];
                            }
                            else
                            {
                                if (matches.Count > 0) { GVL.OCR_CM_MAC = matches[0]; OCRCMMAC_TextBox.Text = matches[0]; } else { OCRCMMAC_TextBox.Text = "Not Found"; }
                                if (matches.Count > 1) { GVL.OCR_WAN_MAC = matches[1]; OCRWANMAC_TextBox.Text = matches[1]; } else { OCRWANMAC_TextBox.Text = "Not Found"; }
                                if (matches.Count > 2) { GVL.OCR_MTA_MAC = matches[2]; OCRMTAMAC_TextBox.Text = matches[2]; } else { OCRMTAMAC_TextBox.Text = "Not Found"; }
                            }

                            isCMMACReady = true;
                            isWANMACReady = true;
                            isMTAMACReady = true;
                        }
                        else if (region.Key == "LEFTSIDE")
                        {
                            Mat matCropped = BitmapConverter.ToMat(cropped);
                            Mat enhanced = new Mat();
                            matCropped.ConvertTo(enhanced, MatType.CV_8UC3, alpha: 1.5, beta: 10); // alpha=kontras, beta=brightness

                            Bitmap enhancedBitmap = BitmapConverter.ToBitmap(enhanced);

                            LeftPicture.Image = enhancedBitmap;
                            string result = ProcessOCR_Paddle(enhancedBitmap);


                            if (!string.IsNullOrWhiteSpace(result))
                            {
                                string referenceText = "INPUT120V46AFCCREGULATIONSRESTRICTTOINDOORUSEONLY";

                                if (result.StartsWith("nput") || result.StartsWith("1nput") || result.StartsWith("lnput"))
                                {
                                    result = result
                                        .Replace("nput", "Input")
                                        .Replace("1nput", "Input")
                                        .Replace("lnput", "Input");
                                }

                                string cleanedResult = result
                                    .Replace("\n", " ")
                                    .Replace("\r", " ")
                                    .Replace(":", " ")
                                    .Replace(".", " ")
                                    .Replace("-", " ")
                                    .Replace("OV", "0V")
                                    .Replace("rogulatlons", "regulations")
                                    .Replace("roguiatlons", "regulations")
                                    .Replace("roguiations", "regulations")
                                    .Replace("regulatlons", "regulations")
                                    .Replace("reguiatlons", "regulations")
                                    .Replace("reguiations", "regulations")
                                    .Replace("restrlct", "restrict")
                                    .Replace("resfrlct", "restrict")
                                    .Replace("resfrict", "restrict")
                                    .Replace("inaoor", "indoor")
                                    .Replace("Inaoor", "indoor")
                                    .Replace("lnaoor", "indoor")
                                    .Replace("lndoor", "indoor")
                                    .Replace("1ndoor", "indoor")
                                    .Replace("1naoor", "indoor")
                                    .Replace("onty", "only")
                                    .Replace("onfy", "only")
                                    .Replace("oniy", "only")
                                    .Replace("onjy", "only")
                                    .Replace("onIy", "only")
                                    .Replace("  ", " ")
                                    .Trim()
                                    .ToUpperInvariant()
                                    .Replace(" ", "");

                                string cleanedReference = referenceText
                                    .Replace(" ", "")
                                    .ToUpperInvariant();

                                double matchScore = CalculateSimilarity(cleanedResult, cleanedReference);
                                LeftLabelScore = matchScore;

                                Console.WriteLine($"[Left Label OCR]: {result}");
                                Console.WriteLine($"[Cleaned]: {cleanedResult}");
                                Console.WriteLine($"[LeftLabel Match Score] {matchScore:0.0}%");

                            }
                            isLeftLabelReady = true;
                        }
                        else if (region.Key == "RIGHTSIDE")
                        {
                            RightPicture.Image = cropped;
                            string result = ProcessOCR_Paddle(cropped);

                            if (!string.IsNullOrWhiteSpace(result))
                            {
                                string referenceText = "PROPERTYOFCOMCASTNOTFORRESALEREMOVINGTHISSTICKERORTAMPERINGWITHTHISEQUIPMENTISAVIOLATIONOFFEDERALANDSTATELAW";

                                string cleanedResult = result
                                    .Replace("\n", " ")
                                .Replace("\r", " ")
                                .Replace("  ", " ")
                                .Replace(".", " ")
                                .Replace("OE", "OF")
                                .Replace("0", "O")
                                .Replace("4", "A")
                                .Replace("l", "I")
                                    .Trim()
                                    .ToUpperInvariant()
                                    .Replace(" ", "");

                                string cleanedReference = referenceText
                                    .Replace(" ", "")
                                    .ToUpperInvariant();

                                double matchScore = CalculateSimilarity(cleanedResult, cleanedReference);
                                RightLabelScore = matchScore;

                                Console.WriteLine($"[Right Label OCR]: {result}");
                                Console.WriteLine($"[Cleaned]: {cleanedResult}");
                                Console.WriteLine($"[RightLabel Match Score] {matchScore:0.0}%");

                            }
                            isRightLabelReady = true;
                        }
                    }

                    if (isPNReady && isFCCIDReady && isCMMACReady && isWANMACReady &&
                        isMTAMACReady && isNETWORKReady && isPasswordReady &&
                        isLeftLabelReady && isRightLabelReady)
                    {
                        isOCRReady = true;
                    }
                    else
                    {
                        CC.WriteCoilMemory(NGCoil, true);
                        string time = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}";
                        AppendNewEntry($"[{time}] ===NEW ENTRY===", Color.Black);
                        AppendLogText("Some target is not detected, please adjust image quality and try again!", Color.Red);
                        this.Invoke(new Action(() =>
                        {
                            LabelResult.Text = "NG";
                            LabelResult.ForeColor = Color.Red;
                        }));
                        Thread.Sleep(200);
                        CC.WriteCoilMemory(NGCoil, false);
                        if (CC.ErrorCode == 1)
                        {
                            AppendLogText(CC.ErrorString, Color.Red);
                            CC.ErrorCode = 0;
                        }
                        SaveImageByResult(SourcePicture.Image, false);
                        UpdateCounter(false);
                    }

                    CheckReady();
                }
                catch (Exception ex)
                {
                    CC.WriteCoilMemory(NGCoil, true);
                    string time = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}";
                    AppendNewEntry($"[{time}] ===NEW ENTRY===", Color.Black);
                    AppendLogText("Error ocured" + ex.Message, Color.Red);
                    this.Invoke(new Action(() =>
                    {
                        LabelResult.Text = "NG";
                        LabelResult.ForeColor = Color.Red;
                    }));
                    Thread.Sleep(200);
                    CC.WriteCoilMemory(NGCoil, false);
                    if (CC.ErrorCode == 1)
                    {
                        AppendLogText(CC.ErrorString, Color.Red);
                        CC.ErrorCode = 0;
                    }
                    SaveImageByResult(SourcePicture.Image, false);
                    UpdateCounter(false);
                }
            }
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            ThreadRunning = false;
            SFIS.SFIS_LogINOUT(2);
            CC.Disconnect();
            Thread.Sleep(100);
            Application.Exit();
        }

        private void ComboBox_Model_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (ComboBox_Model.SelectedIndex == 0)
            {
                SelectModel = 1;
            }
            else if (ComboBox_Model.SelectedIndex == 1)
            {
                SelectModel = 2;
            }
        }

        private void ComboBox_Rotation_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (ComboBox_Rotation.SelectedIndex == 0)
            {
                ImageRotation = 0;
            }
            else if (ComboBox_Rotation.SelectedIndex == 1)
            {
                ImageRotation = 90;
            }
            else if (ComboBox_Rotation.SelectedIndex == 2)
            {
                ImageRotation = 180;
            }
            else if (ComboBox_Rotation.SelectedIndex == 3)
            {
                ImageRotation = 270;
            }
        }
    }
}
