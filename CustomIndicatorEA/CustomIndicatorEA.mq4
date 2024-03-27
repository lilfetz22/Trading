//+------------------------------------------------------------------+
//|                                          MQLTutorialLesson12.mq4 |
//|                                                         Alistair |
//|                                        www.toolkitfortraders.com |
//+------------------------------------------------------------------+
// 1. OrderSend function to open orders on the indicators signal
// 2. Retain the order ticket number
// 3. Close the order on the reverse of the signal


#property copyright "Alistair"
#property link      "www.toolkitfortraders.com"
#property version   "1.00"
#property strict

//input double max_spread = 7;
input string indicatorName = "T4T\\T4T LowPass v1.0";
input int MA_Period = 50;
input ENUM_MA_METHOD MA_Type = MODE_EMA;
input ENUM_APPLIED_PRICE MA_Price = PRICE_CLOSE;
input double Fixed_Lot_Size = 0.01;
input double Risk_Percent = 0;
input double Stop_Loss_Points = 200;
input bool Use_ATR = false;
input int ATR_Period = 14;
input double ATR_Multiplier = 1.0;

// GLOBAL VARIABLES
int gBuyTicket = 0;
int gSellTicket = 0;
datetime gBarTime;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
      // new local variable to hold the current open time of the furthest right bar
      datetime rightBarTime = iTime(_Symbol, _Period, 0);
      
      // check if furthest right bar has the same open time as our global variable
      if(rightBarTime != gBarTime)
      {
         // set the global varaible to be the time of the new bar
         gBarTime = rightBarTime;
         // call our on bar function
         OnBar();
      }
   
  }
  
// when called, the code here will execute
void OnBar()
   {
      double MA_current = iCustom(_Symbol, _Period, indicatorName, MA_Period, 0, MA_Price, 0, 1);
      double MA_was = iCustom(_Symbol, _Period, indicatorName, MA_Period, 0, MA_Price, 0, 2);
      double MA_was_was = iCustom(_Symbol, _Period, indicatorName, MA_Period, 0, MA_Price, 0, 3);   
           
      // if current momentum is above threshold, open a buy order, close any sells
      if(MA_was_was > MA_was && MA_was < MA_current)
      {
         Comment("GO LONG!!!");
         
         double stopLoss = GetStopLoss();
         double lots = GetLots(stopLoss); // use function to get number of lots to trade
         
         gBuyTicket = OrderSend(_Symbol, OP_BUY, 0.01, Ask, 100, Ask - (_Point*stopLoss), 0, "MQL4 Tutorial Lesson 10", 30012021);
         if(gSellTicket > 0)
         {
            CloseOrder(gSellTicket);
         }
      // if current momentum is below threshold, open a sell order, close any buys
      } else if(MA_was_was < MA_was && MA_was > MA_current)
      {
         Comment("GO SHORT!!!");
         
         double stopLoss = GetStopLoss();
         double lots = GetLots(stopLoss); // use function to get number of lots to trade
         
         gSellTicket = OrderSend(_Symbol, OP_SELL, 0.01, Bid, 100, Bid + (_Point*stopLoss), 0, "MQL4 Tutorial Lesson 10", 30012021);
         if(gBuyTicket > 0)
         {
            CloseOrder(gBuyTicket);
         }
      }
   }
//+------------------------------------------------------------------+

double CalculateSpread(double pBid, double pAsk)
{
   // create a spread variable, and give the bid ask spread as a value.
   double current_spread = pAsk - pBid;
   // divide by _Point variable, to give the spread in points
   double point_spread;
   point_spread = current_spread/_Point;
   return point_spread;
}

void CloseOrder(int pTicket)
{
   while(IsTradeContextBusy())
   {
      Sleep(50);
   }
   
   if(OrderSelect(pTicket, SELECT_BY_TICKET))
   {
      double lots = OrderLots();
      double price = 0;
      if(OrderType() == OP_BUY) price = Bid;
      else if(OrderType() == OP_SELL) price = Ask;
      
      bool closed = OrderClose(pTicket, lots, price, 100);
      
      if(!closed) Alert("Trade no closed: ", pTicket);
   }
}

double GetLots(double pStopLoss)
{
   // set default to be fixed lot size
   double lots = Fixed_Lot_Size;
   
   // use risk percent if user sets in input
   if(Risk_Percent > 0)
   {
      double risk_amount = AccountBalance() * Risk_Percent;
      double tick_value = MarketInfo(_Symbol, MODE_TICKVALUE);
      
      lots = (risk_amount/pStopLoss) / tick_value;
   }
   // Verify lot size
   if(lots < MarketInfo(Symbol(), MODE_MINLOT)) 
   {
      Alert("Lots traded is too small for your broker.");
      lots = 0;
   }
   else if(lots > MarketInfo(_Symbol, MODE_MAXLOT)) 
   {
      Alert("Lots traded is too large for your broker.");
      lots = 0;
   } 
   return lots;
}

double GetStopLoss()
{  
   double stopLoss = Stop_Loss_Points;
   
   if(Use_ATR == true)
   {
      double atr = iATR(_Symbol, _Period, ATR_Period, 1);
      stopLoss = ATR_Multiplier*atr/_Point;
   }
   return stopLoss;
}