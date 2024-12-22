//+------------------------------------------------------------------+
//|                                            Double SMA Renko.mq4  |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property strict
//--- input parameters
input string   indicators = "----- Indicators settings -----";
input string   SMA_Inputs ="--- SMA settings ---";
input bool     UseSMA=True;
input int      SMA_length=4;
input string   Smoothing_Inputs ="--- Smoothing SMA settings ---";
input bool     UseSmoothing=True;
input int      smoothing_sma_length=4;

input string   Order_Inputs ="----- Order settings -----";
input double   input_lot_size=3;
input int      trades_in_runway=10;
// input double   takeprofit=0;
input int      Start_Hour=1;
input int      End_Hour=22;
input string   Account_Inputs ="--- Account Protection settings ---";
input double   max_Daily_Drawdown_Perc=0.03;
input double   max_Total_Drawdown_Amt=184000;
input bool     Max_Payout_Bool=False;
input int      Max_Payout_Amt=12000;
input int      Initial_Acct_Size=200000;
input string   News_Inputs ="--- News settings ---";
input bool     News_Trading_Allowed=True;
input bool     Renko_bars=True;
input bool     trailing_stoploss=False;

// GLOBAL VARIABLES
int      gBuyTicket = 0;
int      gSellTicket = 0;

double   acct_equity = AccountInfoDouble(ACCOUNT_EQUITY);
double   balanceAtLastClose = 0;
bool     more_trades = True;
bool     acct_protection = False;

int      BarCount;
int      Current;
bool     TickCheck = False;
bool     EachTickMode = False;
double   lot_size = 0;
double   stoploss = 0.0003;
double   sma_array[];
double   smoothing_sma_array[];
double   compare_sma_array[];
bool     neg_swap_short = FALSE;
bool     neg_swap_long = FALSE;
bool     swap_protection_long = FALSE;
bool     swap_protection_short = FALSE;





//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   BarCount = Bars;

   if (EachTickMode) Current = 0; else Current = 1;
   if (balanceAtLastClose == 0) balanceAtLastClose = AccountInfoDouble(ACCOUNT_BALANCE);
   ArraySetAsSeries(sma_array, True);
   ArrayResize(sma_array, smoothing_sma_length);
   ArraySetAsSeries(smoothing_sma_array, True);
   ArrayResize(smoothing_sma_array, 3);
   ArraySetAsSeries(compare_sma_array, True);
   ArrayResize(compare_sma_array, 3);
   // fill in the smoothing sma array with the smoothing value of the last 3 bars
   for (int i = 0; i < 3; i++)
   {
     
    // fill the sma_array with the value of the last sma_length*smoothing_sma_length bars
      for (int j = 0; j < smoothing_sma_length; j++)
      {
        int start = j + i + 1;

        sma_array[j] = ohlc4(start, smoothing_sma_length) / SMA_length;
      }
      smoothing_sma_array[i] = iMAOnArray(sma_array, 0, smoothing_sma_length, 0, MODE_SMA, 0);
   }

   // determine which direction "buy" or "sell" the negative swap is in
    double swapRateLong = SymbolInfoDouble(_Symbol, SYMBOL_SWAP_LONG); // For long positions
    double swapRateShort = SymbolInfoDouble(_Symbol, SYMBOL_SWAP_SHORT); // For short positions
    Print("Swap Rate Long: ", swapRateLong);
    Print("Swap Rate Short: ", swapRateShort);
    if (swapRateLong < 0)
    {
      neg_swap_long = TRUE;
    }
    else if (swapRateShort < 0)
    {
      neg_swap_short = TRUE;
    }

    // Print the elements of the array to the console
    // for(int i = 0; i < ArraySize(sma_array); i++) {
    //     Print("sma_array[", i, "] = ", sma_array[i]);
    // }

   return(0);
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
    // Print("Bars: ", Bars);
    // Print("BarCount: ", BarCount);

    if (EachTickMode && Bars != BarCount) TickCheck = False;
    // Print("Bars: ", Bars);    
    if((EachTickMode && !TickCheck) || (!EachTickMode && (Bars != BarCount))
      )
    {
      Print("Account Protection: ", acct_protection);
        
        // call our on bar function if acct_protection is False
        if (!acct_protection)
        {
          Print("Entering into OnBar");
          OnBar();
        }
    }

    // Is the order still open?
    if (gBuyTicket > 0)
    {
      if (OrderSelect(gBuyTicket, SELECT_BY_TICKET))
      {
        // is the ticket still open? 
        if (OrderCloseTime() != 0)
        {
          gBuyTicket = 0;
        }
      }
    }
    else if (gSellTicket > 0)
    {
      if (OrderSelect(gSellTicket, SELECT_BY_TICKET))
      {
        // is the ticket still open? 
        if (OrderCloseTime() != 0)
        {
          gSellTicket = 0;
        }
      }
    }


    // update the max equity
    if (AccountInfoDouble(ACCOUNT_EQUITY) > acct_equity) {
        acct_equity = AccountInfoDouble(ACCOUNT_EQUITY);
    }

    // Check if the current server time corresponds to 5 PM EST (00:00 market time)
    if((TimeHour(TimeCurrent()) == 0) && (TimeMinute(TimeCurrent())) == 0 && (TimeSeconds(TimeCurrent()) == 0)) 
    {
        balanceAtLastClose = AccountInfoDouble(ACCOUNT_BALANCE);
        if (acct_protection) acct_protection = False;
        if (swap_protection_long) swap_protection_long = FALSE;
        if (swap_protection_short) swap_protection_short = FALSE;
        Print("New Account Balance at Reset: ", balanceAtLastClose);
        acct_equity = AccountInfoDouble(ACCOUNT_EQUITY);
        Print("New Account Equity at Reset: ", acct_equity);
    }

    // find the current balance/equity and calculate the drawdown
    double AcctBalDrawdown = (balanceAtLastClose - AccountInfoDouble(ACCOUNT_BALANCE)) / balanceAtLastClose;
    double AcctEquityDrawdown = (acct_equity - AccountInfoDouble(ACCOUNT_EQUITY)) / acct_equity;
    

    // check if the max daily drawdown has been reached or if the time is 12 pm on Friday
    if (AcctBalDrawdown >= max_Daily_Drawdown_Perc ||
       AcctEquityDrawdown >= max_Daily_Drawdown_Perc ||
       ((Max_Payout_Bool) && ((AccountInfoDouble(ACCOUNT_BALANCE) - Initial_Acct_Size) >= Max_Payout_Amt)) ||
        ((TimeHour(TimeCurrent()) >= 23) && DayOfWeek() == 5) || 
        ((((gBuyTicket > 0) && neg_swap_long) || ((gSellTicket > 0) && neg_swap_short)) && 
        (TimeHour(TimeCurrent()) == 23) && (TimeMinute(TimeCurrent()) == 55))){
          // more_trades = False;
          acct_protection = True;    
          if ((gBuyTicket > 0))
          {
            Print("Closing Buy Ticket for Acct Protection Ticket #: ", gBuyTicket);
            CloseOrder(gBuyTicket, 1);
          }
          else if ((gSellTicket > 0))
          {
            Print("Closing Sell Ticket for Acct Protection Ticket #: ", gSellTicket);
            CloseOrder(gSellTicket, 1);
          }
      }
    if (!EachTickMode) BarCount = Bars;
  }
  
  
//+------------------------------------------------------------------+
void OnBar()
  {
      more_trades = True;
      // determine if there is any news happening right now
      if (!News_Trading_Allowed)
      {
        Print("Checking for news");
        more_trades = getnewsfromcsv();
      }
      // if the current time is not > Start_Hour and < End_Hour, don't execute any trades
      if ((TimeHour(TimeCurrent()) < Start_Hour) || (TimeHour(TimeCurrent()) > End_Hour))
      {
        Print("Time: ", TimeHour(TimeCurrent()));
        Print("No Trades Allowed Because we are not within the Start and End Hours");
        more_trades = false;
      }
      else if (((TimeHour(TimeCurrent()) >= Start_Hour) && (TimeHour(TimeCurrent()) <= End_Hour)) && (more_trades))
      {
        more_trades = True;
      }
      // figure out the lot size based upon the account average
      int totalOrders = OrdersHistoryTotal(); // Get total number of closed orders
      double totalLotSize = 0.0; // Initialize total lot size
      double symbol_order_total = 0; // itialize total number of orders for the symbol

      for(int i = 0; i < totalOrders; i++) 
      {
        if(OrderSelect(i, SELECT_BY_POS, MODE_HISTORY)) 
        { // Select each order by position
          // Check if the order's symbol matches the current symbol
          if(OrderSymbol() == Symbol()) 
          {
            symbol_order_total += 1;
            totalLotSize += OrderLots();
          } // Add the lot size of the current order to the total
        }
      }
      if (symbol_order_total == 0)
      {
        totalLotSize = 1;
        symbol_order_total = 1;
      }

      double averageLotSize = totalLotSize / symbol_order_total; // Calculate average lot size
      // Print("Average Lot Size: ", averageLotSize);
      double max_lot_size = 2 * averageLotSize;
      // round max_lot_size to the nearest 0.01
      max_lot_size = MathRound(max_lot_size * 100) / 100;
      // Print("Max Lot Size: ", max_lot_size);

      // determine how large the lot size should be based upon the account balance and the upper limit of the max_lot_size
      double acct_balance = AccountInfoDouble(ACCOUNT_BALANCE);
      double todays_drawdown_limit = max_Daily_Drawdown_Perc * balanceAtLastClose;
      // find out which is less, the AcctBalDrawdown 
      double CurrentBalanceDrawdown = acct_balance - todays_drawdown_limit;
      double TotalDrawdownDiff = acct_balance - max_Total_Drawdown_Amt;
      if (TotalDrawdownDiff < 0)
      {
        TotalDrawdownDiff = 999999999;
      }
      // find out which is less, the AbsBalanceDrawdown or the TotalDrawdownDiff
      double min_drawdown = MathMin(CurrentBalanceDrawdown, TotalDrawdownDiff);
      // Print("Min Drawdown: ", min_drawdown);
      // divide the min_drawdown by the number of trades wanted before the acct_drawdown is reached
      double risk_per_trade = min_drawdown / trades_in_runway;
      // Print("Risk Per Trade: ", risk_per_trade);
      // find the lot size based upon the risk per trade
      if (risk_per_trade == 0) risk_per_trade = 1;
      // find the open price of the current bar
      double current_bar_open = iOpen(NULL, 0, Current);
      double current_bar_close = iClose(NULL, 0, Current);
      Print("Current Bar Close: ", current_bar_close);
      stoploss = MathAbs(current_bar_open - current_bar_close)*SMA_length;
      if (!Renko_bars) stoploss = 0.0003 * 2;
      // Print("Stoploss: ", stoploss);
      double risk_lot_size = MathRound((risk_per_trade / stoploss) * 100) / 100;
      // Print("Risk Lot Size: ", risk_lot_size);
      // find the minimum of the max_lot_size and the risk_lot_size
      double calculated_lot_size = MathMin(max_lot_size, risk_lot_size);
      Print("Calculated Lot Size: ", calculated_lot_size);

      // if more_trades is False, don't execute any trades
      Print("more_trades: ", more_trades);


      //+------------------------------------------------------------------+
      //| ENTRY CONDITIONS FROM INDICATORS                                 |
      //+------------------------------------------------------------------+
      double previous_bar_close = iClose(NULL, 0, Current + 1);
      // Print("previous_bar_close: ", previous_bar_close);

      // sma calculation
      // for the length of sma_length, find the close, open, high, and low of the bar
      double ohlc4_calc = ohlc4(0, SMA_length);

      double sma = ohlc4_calc / SMA_length;
      // Print("SMA: ", sma);
      // store the sma value into an array
      ArrayResize(sma_array, smoothing_sma_length + 1);
      // Shift all existing elements to the right
      for(int i = ArraySize(sma_array) - 1; i > 0; i--) 
      {
          sma_array[i] = sma_array[i - 1];
      }
      sma_array[0] = sma;
      // Resize the array to remove the last element
      ArrayResize(sma_array, ArraySize(sma_array) - 1);

      // Print the elements of the array to the console
      // for(int i = 0; i < ArraySize(sma_array); i++) {
      //     Print("sma_array[", i, "] = ", sma_array[i]);
      // }


      // sma smoothing calculation
      double sma_smooth = 0;
      sma_smooth = iMAOnArray(sma_array, 0, smoothing_sma_length, 0, MODE_SMA, 0);
      // Print("SMA Smooth: ", sma_smooth);
      // store the sma value into an array
      ArrayResize(smoothing_sma_array, 4);
      // Shift all existing elements to the right
      for(int i = ArraySize(smoothing_sma_array) - 1; i > 0; i--) 
      {
          smoothing_sma_array[i] = smoothing_sma_array[i - 1];
      }
      smoothing_sma_array[0] = sma_smooth;
      // Resize the array to remove the last element
      ArrayResize(smoothing_sma_array, ArraySize(smoothing_sma_array) - 1);
      
      // Print the elements of the array to the console
      // for(int i = 0; i < ArraySize(smoothing_sma_array); i++) {
      //     Print("smoothing_sma_array[", i, "] = ", smoothing_sma_array[i]);
      // }


      // compare each of the values in sma_array to the sma_smooth_array value
      for (int i = 0; i < ArraySize(compare_sma_array); i++)
      {
        if (sma_array[i] > smoothing_sma_array[i])
        {
          compare_sma_array[i] = 1;
        }
        else if (sma_array[i] < smoothing_sma_array[i])
        {
          compare_sma_array[i] = -1;
        }
        else if (sma_array[i] == smoothing_sma_array[i])
        {
          compare_sma_array[i] = 0;
        }
      }
        // print the compare_sma_array
      for(int i = 0; i < ArraySize(compare_sma_array); i++) {
          Print("compare_sma_array[", i, "] = ", compare_sma_array[i]);
      }

      bool sma_long = False;
      bool sma_short = False;
      if ((compare_sma_array[0] == 1 && compare_sma_array[1] == -1) ||
          (compare_sma_array[0] == 1 && compare_sma_array[1] == 0 && compare_sma_array[2] == -1))
      {
        sma_long = True;
        if (gSellTicket > 0)
        {
          Print("Closing Sell Ticket: ", gSellTicket);
          CloseOrder(gSellTicket, 1);
        }
      }
      else if ((compare_sma_array[0] == -1 && compare_sma_array[1] == 1) ||
               (compare_sma_array[0] == -1 && compare_sma_array[1] == 0 && compare_sma_array[2] == 1))
      {
        sma_short = True;
        if (gBuyTicket > 0)
        {
          Print("Closing Buy Ticket: ", gBuyTicket);
          CloseOrder(gBuyTicket, 1);
        }
      }

      // create a bool that is True if the number of open orders is 0
      bool no_active_trades = (gSellTicket == 0) && (gBuyTicket == 0);
      // if (no_active_trades) Print("no_active_trades: ", no_active_trades);

      if (input_lot_size != 0)
      {
        lot_size = input_lot_size;
        if (calculated_lot_size < lot_size)
        {
          lot_size = calculated_lot_size;
        }
      }
      else
      {
        lot_size = calculated_lot_size;
      }

      if  (TimeHour(TimeCurrent()) == 23 && neg_swap_long) 
      {
        swap_protection_long = True;
      }
      else if (TimeHour(TimeCurrent()) == 23 && neg_swap_short)
      {
        swap_protection_short = True;
      }

      // if indicators give the signal, sell
      if(sma_short && no_active_trades && more_trades && !swap_protection_short) // 
      {
        Print("SELL");
        // if (takeprofit != 0) order_takeprofit = Bid - (takeprofit);
        // else order_takeprofit = 0;
        gSellTicket = OrderSend(_Symbol, OP_SELL, lot_size, Bid, 100, Bid + (stoploss), 0, "double_sma", 59726816, 0, Yellow);
        Print("New Ticket #: ", gSellTicket);

        if (gSellTicket == -1) 
        {
          Print("Unable to place order"); 
          gSellTicket = 0;
        }
        // if (initial_TP_bool) take_half_cond = True;
      }
      // if indicators give the signal, buy
      else if(sma_long && no_active_trades && more_trades && !swap_protection_long)// 
      {
        
        Print("BUY");
        // Comment("BUY");
        // if (takeprofit != 0) order_takeprofit = Ask + (takeprofit);
        // else order_takeprofit = 0;
        gBuyTicket = OrderSend(_Symbol, OP_BUY, lot_size, Ask, 100, Ask - (stoploss), 0, "double_sma", 59726816, 0, Green);
        Print("New Ticket #: ", gBuyTicket);
        if (gBuyTicket == -1) 
        {
          Print("Unable to place order"); 
          gBuyTicket = 0;
        }
        // if (initial_TP_bool) take_half_cond = True;
      }
      else 
      {
          Print("NO TRADE");
      }

      //+------------------------------------------------------------------+
      //| TRAILING STOPLOSS                                                |
      //+------------------------------------------------------------------+

      if ((gBuyTicket > 0) && (trailing_stoploss))
      {
          if(OrderSelect(gBuyTicket, SELECT_BY_TICKET))
        {
          if (Bid > OrderOpenPrice())
          {
            modify_stoploss(gBuyTicket);
          }
        }
      }
      if ((gSellTicket > 0) && (trailing_stoploss))
      {
          if(OrderSelect(gSellTicket, SELECT_BY_TICKET))
        {
          if (Ask < OrderOpenPrice())
          {
            modify_stoploss(gSellTicket);
          }
        }
      }

}

//+------------------------------------------------------------------+

void CloseOrder(int pTicket, int take_half)
{
   while(IsTradeContextBusy())
   {
      Sleep(50);
   }
   
   if(OrderSelect(pTicket, SELECT_BY_TICKET))
   {
      double lots = OrderLots();
      double price = 0;
      lots = lots / take_half;
      if(OrderType() == OP_BUY)
      { 
        price = Bid;
      }
      else if(OrderType() == OP_SELL) 
      {
        price = Ask;
      }
      Print("Ticket #: ", pTicket);
      bool closed = OrderClose(pTicket, lots, price, 100, Red);

      if(!closed) Alert("Trade not closed: ", pTicket);

      if((OrderType() == OP_BUY) && (take_half == 1))
      { 
        gBuyTicket = 0;
      }
      else if((OrderType() == OP_SELL) && (take_half == 1)) 
      {
        gSellTicket = 0;
      }
   }
}




//+------------------------------------------------------------------+

 
bool getnewsfromcsv()
  {
    // get the date of the past Sunday
    datetime today = TimeCurrent();
    int day_of_week = DayOfWeek();
    int days_to_subtract = 0;
    if (day_of_week == 0)
    {
      days_to_subtract = 0;
    }
    else
    {
      days_to_subtract = day_of_week;
    }
    datetime past_sunday = today - days_to_subtract * 86400;
    // get the month, day, and year of the past Sunday
    int month = TimeMonth(past_sunday);
    int day = TimeDay(past_sunday);
    int year = TimeYear(past_sunday);
    // convert the month, day, and year to a string
    string month_str = IntegerToString(month);
    string day_str = IntegerToString(day);
    string year_str = IntegerToString(year);
    // get the news from the csv file
    string filename = "calendar_statement_" + year_str + "_" + month_str + "_" + day_str + ".csv";
    //Print("Filename: ", filename);
    int file_handle = FileOpen(filename, FILE_READ | FILE_CSV);
    int any_news = 0;
    
    if (file_handle != INVALID_HANDLE) {

        // size = (int)FileReadNumber(file_handle);
        // Print("Size: ", size);
        while (!FileIsEnding(file_handle)) {
            datetime date = FileReadDatetime(file_handle); // Assuming "Date" is the first column
            if (TimeDay(date) == TimeDay(TimeCurrent()))
            {
              // find how many hours and minutes until the news event
              int hours = TimeHour(date) - TimeHour(TimeCurrent());
              if (hours == 0)
              {
                int minutes = TimeMinute(date) - TimeMinute(TimeCurrent());
                Print("News in ", minutes, " minutes");
                if ((!News_Trading_Allowed) && (MathAbs(minutes) <= 10))
                {
                  Print("News event happening in ", minutes, " minutes! No Trades Allowed");
                  any_news = 1;
                  break;
                }
                // if the news event happened within the past hour, don't allow any trades
                else if ((minutes > -60) && (minutes < -10))
                {
                  Print("News event happened within the past hour!");
                  break;
                }
                // however, if there are no events happening in 5 minutes, allow trades
                else if (minutes > 10)
                {
                  Print("News event happening within an hour");
                  break;
                }
              }
            }
        }
        FileClose(file_handle);

    } else {
        Print("Error opening file: ", GetLastError());
    }
    if (any_news == 1)
    {
      return false;
    }
    else
    {
      return true;
    }


  }

void modify_stoploss(int pTicket)
{
  // trailing stoploss
  // find the absolute value of the difference between the stoploss and the current price
  if(OrderSelect(pTicket, SELECT_BY_TICKET))
  {
      double price =  0;
      if(OrderType() == OP_BUY)
      {  
        price = Bid;
      }
      else if(OrderType() == OP_SELL)  
      {
        price = Ask;
      }
    double stoploss_diff = MathAbs(MathAbs(OrderOpenPrice() - stoploss) - price);
    // Print("Stoploss Diff: ", stoploss_diff);
    // if the difference is greater than stoploss, modify the stoploss to be the current price - stoploss
    if (stoploss_diff > stoploss)
    {
      // find the current stoploss
      double current_stoploss = OrderStopLoss();
      double order_takeprofit = 0;
      if (OrderType() == OP_BUY)
      {
        if (current_stoploss < price - stoploss)
        {
        // if (takeprofit != 0)
        // {
        //   order_takeprofit = OrderOpenPrice() + (takeprofit);
        // }
        // else 
        // {
        //   order_takeprofit = 0;
        // }
          bool modified = OrderModify(pTicket, OrderOpenPrice(), price - stoploss, order_takeprofit, 0, 0);
          if (!modified)
          {
            Alert("Trade not modified: ", pTicket);
          }
        }
      }
      else if (OrderType() == OP_SELL)
      {
        if (current_stoploss > price + stoploss)
        {
        // if (takeprofit != 0)
        // {
        //   order_takeprofit = OrderOpenPrice() - (takeprofit);
        // }
        // else 
        // {
        //   order_takeprofit = 0;
        // }

          bool modified = OrderModify(pTicket, OrderOpenPrice(), price + stoploss, order_takeprofit, 0, 0);
          if (!modified)
            {
              Alert("Trade not modified: ", pTicket);
            }
        }

      }
    }

  }
}


double ohlc4(int start, int length)
{
  double ohlc4 = 0;
  for (int i = start; i < (start + length); i++)
      {
        double sma_sum = 0;
        sma_sum += iClose(NULL, 0, Current + i);
        sma_sum += iOpen(NULL, 0, Current + i);
        sma_sum += iHigh(NULL, 0, Current + i);
        sma_sum += iLow(NULL, 0, Current + i);
        ohlc4 += sma_sum / 4;
      }
  return ohlc4;
}



