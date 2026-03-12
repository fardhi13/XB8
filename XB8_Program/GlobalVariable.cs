using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace XB8_Program
{
    public class GlobalVariable
    {
        public string WatcherDir = ConfigurationManager.AppSettings["WatcherDir"];

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

        #region SFIS RESPONSE
        public string StartResult;
        public string TypeResult;
        public string Model;
        public string SSID;
        public string Password;
        #endregion
    }
}
