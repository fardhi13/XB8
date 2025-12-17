using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using XB8_Program.WebReference;

namespace XB8_Program
{
    public class SFISClass
    {
        public string StringLogin;
        public string StringSNReader;
        public string StringSSIDReader;
        public string StringPasswordReader;
        string DeviceID = ConfigurationManager.AppSettings["DeviceId"];
        string ProgramID = ConfigurationManager.AppSettings["ProgramID"];   
        string ProgramPass = ConfigurationManager.AppSettings["ProgramPass"];
        string UserID = ConfigurationManager.AppSettings["UserID"];
        string UserPass = ConfigurationManager.AppSettings["UserPass"];
        string ErrorCode = ConfigurationManager.AppSettings["ErrorCode"];
        SFISTSPWebService SFIS = new SFISTSPWebService();

        public void SFIS_LogINOUT(int status)
        {
            var logIN = SFIS.WTSP_LOGINOUT(ProgramID, ProgramPass, UserID, UserPass, DeviceID, "", status);
            Console.WriteLine(logIN);
        }

        public string SFIS_CheckSN(string ISN)
        {
            var SN = SFIS.WTSP_GETVERSION(ProgramID, ProgramPass, ISN, DeviceID, "GET_MODELUNUNIQUE", "", "");
            Console.WriteLine(SN);
            StringSNReader = SN.ToString();
            return StringSNReader;
        }
    }
}
