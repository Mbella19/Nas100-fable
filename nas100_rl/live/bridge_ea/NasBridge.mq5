//+------------------------------------------------------------------+
//| NasBridge.mq5 — file bridge between MT5 (incl. Wine) and the     |
//| Python live runner (FileBridgeGateway).                          |
//|                                                                  |
//| Exchange directory: the terminal's MQL5/Files folder (sandboxed).|
//| Point config.LIVE["bridge_dir"] at it, e.g. under Wine:          |
//|   ~/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files          |
//| or <terminal data dir>/MQL5/Files (check File > Open Data Folder)|
//|                                                                  |
//| Files:                                                           |
//|   bars.csv     out: closed M1 bars  srv,o,h,l,c,vol,spread       |
//|   account.csv  out: balance,equity                               |
//|   spec.csv     out: lot_step,min_lot,usd_per_point_per_lot       |
//|   positions.csv out: ticket,side,lots,price,sl,tp,comment        |
//|   commands.csv in : id,action,side,lots,sl,tp,ticket,comment     |
//|   acks.csv     out: id,ok,ticket,price,msg                       |
//+------------------------------------------------------------------+
#property strict
#include <Trade/Trade.mqh>

input string  InpSymbol      = "NAS100";
input int     InpTimerMs     = 1000;
input int     InpBackfill    = 5000;     // bars exported on first run

CTrade  trade;
datetime g_last_bar = 0;
long     g_cmd_offset = 0;

int OnInit()
{
   SymbolSelect(InpSymbol, true);
   trade.SetDeviationInPoints(50);
   WriteSpec();
   EventSetMillisecondTimer(InpTimerMs);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { EventKillTimer(); }

void WriteSpec()
{
   double step = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_STEP);
   double minl = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_MIN);
   double tv   = SymbolInfoDouble(InpSymbol, SYMBOL_TRADE_TICK_VALUE);
   double ts   = SymbolInfoDouble(InpSymbol, SYMBOL_TRADE_TICK_SIZE);
   double upp  = (ts > 0) ? tv / ts : 1.0;   // usd per index point per lot
   int h = FileOpen("spec.csv", FILE_WRITE|FILE_TXT|FILE_ANSI);
   if(h != INVALID_HANDLE)
   {
      FileWriteString(h, StringFormat("%.4f,%.4f,%.6f\n", step, minl, upp));
      FileClose(h);
   }
}

void ExportBars()
{
   MqlRates rates[];
   int n = CopyRates(InpSymbol, PERIOD_M1, 0, (g_last_bar == 0 ? InpBackfill : 200), rates);
   if(n < 2) return;
   // rates[n-1] is the forming bar — export only closed ones after g_last_bar
   int h = FileOpen("bars.csv", FILE_READ|FILE_WRITE|FILE_TXT|FILE_ANSI);
   if(h == INVALID_HANDLE) return;
   FileSeek(h, 0, SEEK_END);
   for(int i = 0; i < n - 1; i++)
   {
      if(rates[i].time <= g_last_bar) continue;
      FileWriteString(h, StringFormat("%s,%.2f,%.2f,%.2f,%.2f,%d,%d\n",
         TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES),
         rates[i].open, rates[i].high, rates[i].low, rates[i].close,
         (int)rates[i].tick_volume, rates[i].spread));
      g_last_bar = rates[i].time;
   }
   FileClose(h);
}

void ExportAccount()
{
   int h = FileOpen("account.csv", FILE_WRITE|FILE_TXT|FILE_ANSI);
   if(h != INVALID_HANDLE)
   {
      FileWriteString(h, StringFormat("%.2f,%.2f\n",
         AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY)));
      FileClose(h);
   }
   int hp = FileOpen("positions.csv", FILE_WRITE|FILE_TXT|FILE_ANSI);
   if(hp != INVALID_HANDLE)
   {
      for(int i = 0; i < PositionsTotal(); i++)
      {
         ulong tk = PositionGetTicket(i);
         if(PositionSelectByTicket(tk) && PositionGetString(POSITION_SYMBOL) == InpSymbol)
            FileWriteString(hp, StringFormat("%I64u,%d,%.2f,%.2f,%.2f,%.2f,%s\n",
               tk, (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : -1),
               PositionGetDouble(POSITION_VOLUME), PositionGetDouble(POSITION_PRICE_OPEN),
               PositionGetDouble(POSITION_SL), PositionGetDouble(POSITION_TP),
               PositionGetString(POSITION_COMMENT)));
      }
      FileClose(hp);
   }
}

void Ack(string id, bool ok, string ticket, double price, string msg)
{
   int h = FileOpen("acks.csv", FILE_READ|FILE_WRITE|FILE_TXT|FILE_ANSI);
   if(h == INVALID_HANDLE) return;
   FileSeek(h, 0, SEEK_END);
   FileWriteString(h, StringFormat("%s,%s,%s,%.2f,%s\n", id, ok ? "1" : "0", ticket, price, msg));
   FileClose(h);
}

void ProcessCommands()
{
   if(!FileIsExist("commands.csv")) return;
   int h = FileOpen("commands.csv", FILE_READ|FILE_TXT|FILE_ANSI);
   if(h == INVALID_HANDLE) return;
   FileSeek(h, g_cmd_offset, SEEK_SET);
   while(!FileIsEnding(h))
   {
      string line = FileReadString(h);
      g_cmd_offset = FileTell(h);
      if(StringLen(line) < 3) continue;
      string p[];
      int n = StringSplit(line, ',', p);
      if(n < 8) continue;
      string id = p[0], action = p[1], ticket = p[6], comment = p[7];
      int side = (int)StringToInteger(p[2]);
      double lots = StringToDouble(p[3]);
      double sl = (p[4] == "") ? 0.0 : StringToDouble(p[4]);
      double tp = (p[5] == "") ? 0.0 : StringToDouble(p[5]);

      if(action == "open")
      {
         bool ok = (side > 0)
            ? trade.Buy(lots, InpSymbol, 0.0, sl, tp, comment)
            : trade.Sell(lots, InpSymbol, 0.0, sl, tp, comment);
         Ack(id, ok, StringFormat("%I64u", trade.ResultOrder()),
             trade.ResultPrice(), IntegerToString((int)trade.ResultRetcode()));
      }
      else if(action == "modify")
      {
         ulong tk = (ulong)StringToInteger(ticket);
         bool ok = false;
         if(PositionSelectByTicket(tk))
            ok = trade.PositionModify(tk,
                  sl > 0 ? sl : PositionGetDouble(POSITION_SL),
                  tp > 0 ? tp : PositionGetDouble(POSITION_TP));
         Ack(id, ok, ticket, 0.0, "");
      }
      else if(action == "close")
      {
         ulong tk = (ulong)StringToInteger(ticket);
         bool ok = trade.PositionClose(tk);
         Ack(id, ok, ticket, trade.ResultPrice(),
             IntegerToString((int)trade.ResultRetcode()));
      }
   }
   FileClose(h);
}

void OnTimer()
{
   ExportBars();
   ExportAccount();
   ProcessCommands();
}
//+------------------------------------------------------------------+
