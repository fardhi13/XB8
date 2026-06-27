using System;
using System.Collections.Generic;
using System.Configuration;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace XB8_Program
{
    public class GlobalVariable
    {
        public bool OnlyDateNow = bool.Parse(ConfigurationManager.AppSettings["OnlyDateNow"]);
        public bool SaveCropped = bool.Parse(ConfigurationManager.AppSettings["SaveCropped"]);
        public string RunningModel = ConfigurationManager.AppSettings["RunningModel"];
        public string WatcherDir = ConfigurationManager.AppSettings["WatcherDir"];
        public int MinimumScoreCOM = int.Parse(ConfigurationManager.AppSettings["MinimumScoreCOM"]);
        public int MinimumScoreCOX = int.Parse(ConfigurationManager.AppSettings["MinimumScoreCOX"]);
        public string DateTemplatePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Templates", "DATE_Template.jpg");
        public string OCR_Date;
        public bool IsDateValid = true;  

        public int StartXPN, StartYPN, WidthPN, HeightPN;
        public int StartXMAC, StartYMAC, WidthMAC, HeightMAC;
        public int StartXSSID, StartYSSID, WidthSSID, HeightSSID;
        public int StartXPASSWORD, StartYPASSWORD, WidthPASSWORD, HeightPASSWORD;
        public int StartXBARCODE, StartYBARCODE, WidthBARCODE, HeightBARCODE;
        public int StartXQR, StartYQR, WidthQR, HeightQR;
        public int StartXLEFTSIDE, StartYLEFTSIDE, WidthLEFTSIDE, HeightLEFTSIDE;
        public int StartXRIGHTSIDE, StartYRIGHTSIDE, WidthRIGHTSIDE, HeightRIGHTSIDE;
        public int StartXFCCID, StartYFCCID, WidthFCCID, HeightFCCID;
        public int StartXLOGOAREA, StartYLOGOAREA, WidthLOGOAREA, HeightLOGOAREA;
        public int StartXINPUT, StartYINPUT, WidthINPUT, HeightINPUT;
        public int StartXSN, StartYSN, WidthSN, HeightSN;
        public int StartXMADE, StartYMADE, WidthMADE, HeightMADE;
        public int StartXREGULATION, StartYREGULATION, WidthREGULATION, HeightREGULATION;

        public int StartXDATE, StartYDATE, WidthDATE, HeightDATE;

        public bool IsLogoDetected = false;

        public string QR_ComcastModel;
        public string QR_TechnicolorModel;
        public string QR_SN;
        public string QR_CM_MAC;
        public string QR_WAN_MAC;
        public string QR_MTA_MAC;
        public string Barcode_CM_MAC;
        public string OCR_PN;
        public string OCR_FCCID;
        public string OCR_CM_MAC;
        public string OCR_WAN_MAC;
        public string OCR_MTA_MAC;
        public string OCR_NetworkName;
        public string OCR_LastFourNetworkName;
        public string OCR_Password;

        public string OCR_SN;
        public string OCR_INPUT;
        public string OCR_Made;
        public string OCR_Regulation;
        public string Barcode_SN;

        #region SFIS RESPONSE
        public string StartResult;
        public string TypeResult;
        public string Model;
        public string SSID;
        public string Password;
        #endregion
    }
}
