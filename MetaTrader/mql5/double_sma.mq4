//+------------------------------------------------------------------+
//|                                                         DAX.mq5  |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property strict
//--- input parameters
input string   indicators= "----- Indicators settings -----";
input string   ATR_Inputs="--- ATR settings ---";
input int      ATR_length=14;

input string   Order_Inputs="----- Order settings -----";
input double   input_lot_size=1;
input int      trades_in_runway=10;
input double   takeprofit=1.5;
input double   stoploss=1.5;
input int      Start_Hour=1;
input int      End_Hour=23;
input string   Account_Inputs="--- Account Protection settings ---";
input double   max_Daily_Drawdown_Perc=0.05;
input int      Initial_Acct_Size=400000;
input double   max_Total_Drawdown_Perc=0.12;
input bool     Max_Payout_Bool=False;
input int      Max_Payout_Amt=12000;
input string   News_Inputs ="--- News settings ---";
input bool     News_Trading_Allowed=True;
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
bool     neg_swap_short = FALSE;
bool     neg_swap_long = FALSE;
bool     swap_protection_long = FALSE;
bool     swap_protection_short = FALSE;

int      ATR_handle;
double   ATR_buffer[];
#define EXPERT_MAGIC 59726816 // MagicNumber of the expert
MqlCalendarValue all_news_events

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {

  int BarCount = Bars(_Symbol, PERIOD_CURRENT);

  int Current = EachTickMode ? 0 : 1;
  if (balanceAtLastClose == 0) balanceAtLastClose = AccountInfoDouble(ACCOUNT_BALANCE);

  // Determine which direction "buy" or "sell" the negative swap is in
  double swapRateLong = SymbolInfoDouble(_Symbol, SYMBOL_SWAP_LONG);  // For long positions
  double swapRateShort = SymbolInfoDouble(_Symbol, SYMBOL_SWAP_SHORT);  // For short positions
  PrintFormat("Swap Rate Long: %.5f", swapRateLong);
  PrintFormat("Swap Rate Short: %.5f", swapRateShort);

  bool neg_swap_long = false;
  bool neg_swap_short = false;

  if (swapRateLong < 0)
  {
      neg_swap_long = true;
  }
  else if (swapRateShort < 0)
  {
      neg_swap_short = true;
  }

    // Initialize the ATR indicator
    ATR_handle = iATR(_Symbol, PERIOD_CURRENT, ATR_length);
    
    if (ATR_handle == INVALID_HANDLE)
    {
        Print("Failed to create ATR indicator. Error code:", GetLastError());
        return(INIT_FAILED);
    }
    
    // Set up the ATR buffer
    ArraySetAsSeries(ATR_buffer, true);

    // get the news for the symbol
    all_news_events = getnewsevents()

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
    // Print("Bars: ", Bars);
    // Print("BarCount: ", BarCount);

    if (EachTickMode && Bars != BarCount) TickCheck = False;
    // Print("Bars: ", Bars);    
  if ((EachTickMode && !TickCheck) || (!EachTickMode && (Bars(_Symbol, PERIOD_CURRENT) != BarCount)))
  {
      Print("Account Protection: ", acct_protection);
      
      // Call our on bar function if acct_protection is False
      if (!acct_protection)
      {
          Print("Entering into OnBar");
          OnBar();
      }
  }

  // Is the order still open?
  if (gBuyTicket > 0)
  {
      if (PositionSelectByTicket(gBuyTicket))
      {
          // Is the position still open?
          if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
          {
              // Position is still open
          }
          else
          {
              gBuyTicket = 0;
          }
      }
      else
      {
          gBuyTicket = 0;
      }
  }
  else if (gSellTicket > 0)
  {
      if (PositionSelectByTicket(gSellTicket))
      {
          // Is the position still open?
          if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
          {
              // Position is still open
          }
          else
          {
              gSellTicket = 0;
          }
      }
      else
      {
          gSellTicket = 0;
      }
  }



  // Update the max equity
  if (AccountInfoDouble(ACCOUNT_EQUITY) > acct_equity) {
      acct_equity = AccountInfoDouble(ACCOUNT_EQUITY);
  }

  // Check if the current server time corresponds to 5 PM EST (00:00 market time)
  MqlDateTime current_time;
  TimeToStruct(TimeCurrent(), current_time);
  if (current_time.hour == 0 && current_time.min == 0 && current_time.sec == 0) 
  {
      balanceAtLastClose = AccountInfoDouble(ACCOUNT_BALANCE);
      if (acct_protection) acct_protection = false;
      if (swap_protection_long) swap_protection_long = false;
      if (swap_protection_short) swap_protection_short = false;
      Print("New Account Balance at Reset: ", balanceAtLastClose);
      acct_equity = AccountInfoDouble(ACCOUNT_EQUITY);
      Print("New Account Equity at Reset: ", acct_equity);
  }

  // Find the current balance/equity and calculate the drawdown
  double AcctBalDrawdown = (balanceAtLastClose - AccountInfoDouble(ACCOUNT_BALANCE)) / balanceAtLastClose;
  double AcctEquityDrawdown = (acct_equity - AccountInfoDouble(ACCOUNT_EQUITY)) / acct_equity;

  // Check if the max daily drawdown has been reached or if the time is 12 pm on Friday
  if (AcctBalDrawdown >= max_Daily_Drawdown_Perc ||
      AcctEquityDrawdown >= max_Daily_Drawdown_Perc ||
      ((Max_Payout_Bool) && ((AccountInfoDouble(ACCOUNT_BALANCE) - Initial_Acct_Size) >= Max_Payout_Amt)) ||
      ((current_time.hour >= 23) && current_time.day_of_week == FRIDAY) || 
      ((((gBuyTicket > 0) && neg_swap_long) || ((gSellTicket > 0) && neg_swap_short)) && 
      (current_time.hour == 23) && (current_time.min == 55))) {
      // more_trades = false;
      acct_protection = true;    
      if (gBuyTicket > 0)
      {
          Print("Closing Buy Position for Acct Protection Position #: ", gBuyTicket);
          CloseOrder(gBuyTicket);
      }
      else if (gSellTicket > 0)
      {
          Print("Closing Sell Position for Acct Protection Position #: ", gSellTicket);
          CloseOrder(gSellTicket);
      }
  }

  if (!EachTickMode) BarCount = Bars(_Symbol, PERIOD_CURRENT);


  // Get the current time
  datetime currentTime = TimeCurrent();
  // Get the day of the week and the hour
  int dayOfWeek = DayOfWeek(currentTime); // 0 = Sunday, 1 = Monday, ..., 6 = Saturday
  int current_hour = Hour(currentTime); // 0 - 23 hours

  // Check if it's Monday before 5 AM
  if (dayOfWeek == 1 && current_hour < 5)
  {
    // Assume MqlCalendarValue all_news_events[] is already defined and populated
    datetime maxTime = 0; // Variable to hold the maximum time value

    // Iterate through the array to find the maximum time value
    for (int i = 0; i < ArraySize(all_news_events); i++)
    {
        if (all_news_events[i].time > maxTime)
        {
            maxTime = all_news_events[i].time; // Update maxTime if the current time is greater
        }
    }

    // Check if the max time is less than the current time
    if (maxTime < currentTime)
    {
        all_news_events = getnewsevents()
    }
  }



  
  
//+------------------------------------------------------------------+
void OnBar()
  {
    bool more_trades = true;
    // Determine if there is any news happening right now
    if (!News_Trading_Allowed)
    {
        Print("Checking for news");
        for (int i = 0; i < ArraySize(all_news_events); i++)
        {
            // Calculate the time difference
            long timeDifference = MathAbs(all_news_events[i].time - currentTime); // Get the absolute value of the difference

            // Check if the difference is less than or equal to 15 minutes (15 * 60 seconds = 900 seconds)
            if (timeDifference <= 900) // 15 minutes in seconds
            {
                more_trades = false;
                break; // Optional: Break if you only need to find one match
            }
        }
    }

    // Get current time
    MqlDateTime current_time;
    TimeToStruct(TimeCurrent(), current_time);

    // If the current time is not > Start_Hour and < End_Hour, don't execute any trades
    if ((current_time.hour < Start_Hour) || (current_time.hour > End_Hour))
    {
        Print("Time: ", current_time.hour);
        Print("No Trades Allowed Because we are not within the Start and End Hours");
        more_trades = false;
    }
    else if (((current_time.hour >= Start_Hour) && (current_time.hour <= End_Hour)) && (more_trades))
    {
        more_trades = true;
    }

    // Figure out the lot size based upon the account average
    HistorySelect(0, TimeCurrent());
    int totalOrders = HistoryOrdersTotal(); // Get total number of closed orders
    double totalLotSize = 0.0; // Initialize total lot size
    double symbol_order_total = 0; // Initialize total number of orders for the symbol

    for(int i = 0; i < totalOrders; i++) 
    {
        ulong ticket = HistoryOrderGetTicket(i);
        if(ticket > 0) 
        {
            // Check if the order's symbol matches the current symbol
            if(HistoryOrderGetString(ticket, ORDER_SYMBOL) == _Symbol) 
            {
                symbol_order_total += 1;
                totalLotSize += HistoryOrderGetDouble(ticket, ORDER_VOLUME_INITIAL);
            }
        }
    }
    if (symbol_order_total == 0)
    {
        totalLotSize = 1;
        symbol_order_total = 1;
    }

    double averageLotSize = totalLotSize / symbol_order_total; // Calculate average lot size
    double max_lot_size = 2 * averageLotSize;
    max_lot_size = NormalizeDouble(max_lot_size, 2); // Round max_lot_size to the nearest 0.01

    double acct_balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double todays_drawdown_limit = max_Daily_Drawdown_Perc * balanceAtLastClose;
    double CurrentBalanceDrawdown = acct_balance - todays_drawdown_limit;
    double TotalDrawdownDiff = acct_balance - max_Total_Drawdown_Amt;
    if (TotalDrawdownDiff < 0)
    {
        TotalDrawdownDiff = 999999999;
    }

    double min_drawdown = MathMin(CurrentBalanceDrawdown, TotalDrawdownDiff);
    double risk_per_trade = min_drawdown / trades_in_runway;
    if (risk_per_trade == 0) risk_per_trade = 1;

    double current_bar_open = iOpen(_Symbol, PERIOD_CURRENT, Current);
    double current_bar_close = iClose(_Symbol, PERIOD_CURRENT, Current);
    Print("Current Bar Close: ", current_bar_close);

    double risk_lot_size = NormalizeDouble((risk_per_trade / stoploss), 2);
    double calculated_lot_size = MathMin(max_lot_size, risk_lot_size);
    Print("Calculated Lot Size: ", calculated_lot_size);

    Print("more_trades: ", more_trades);


      //+------------------------------------------------------------------+
      //| ENTRY CONDITIONS FROM INDICATORS                                 |
      //+------------------------------------------------------------------+
      bool dax_long = False;
      bool dax_short = False;
      bool long_c1 = iHigh(_Symbol, PERIOD_CURRENT, shift) > iHigh(_Symbol, PERIOD_CURRENT, shift+1);
      bool long_c2 = iHigh(_Symbol, PERIOD_CURRENT, shift+1) > iLow(_Symbol, PERIOD_CURRENT, shift);
      bool long_c3 = iLow(_Symbol, PERIOD_CURRENT, shift) > iHigh(_Symbol, PERIOD_CURRENT, shift+2);
      bool long_c4 = iHigh(_Symbol, PERIOD_CURRENT, shift+2) > iLow(_Symbol, PERIOD_CURRENT, shift+1);
      bool long_c5 = iLow(_Symbol, PERIOD_CURRENT, shift+1) > iHigh(_Symbol, PERIOD_CURRENT, shift+3);
      bool long_c6 = iHigh(_Symbol, PERIOD_CURRENT, shift+3) > iLow(_Symbol, PERIOD_CURRENT, shift+2);
      bool long_c7 = iLow(_Symbol, PERIOD_CURRENT, shift+2) > iLow(_Symbol, PERIOD_CURRENT, shift+3);
      bool short_c1 = iLow(_Symbol, PERIOD_CURRENT, shift) < iLow(_Symbol, PERIOD_CURRENT, shift+1);
      bool short_c2 = iLow(_Symbol, PERIOD_CURRENT, shift+1) < iHigh(_Symbol, PERIOD_CURRENT, shift);
      bool short_c3 = iHigh(_Symbol, PERIOD_CURRENT, shift) < iLow(_Symbol, PERIOD_CURRENT, shift+2);
      bool short_c4 = iLow(_Symbol, PERIOD_CURRENT, shift+2) < iHigh(_Symbol, PERIOD_CURRENT, shift+1);
      bool short_c5 = iHigh(_Symbol, PERIOD_CURRENT, shift+1) < iLow(_Symbol, PERIOD_CURRENT, shift+3);
      bool short_c6 = iLow(_Symbol, PERIOD_CURRENT, shift+3) < iHigh(_Symbol, PERIOD_CURRENT, shift+2);
      bool short_c7 = iHigh(_Symbol, PERIOD_CURRENT, shift+2) < iHigh(_Symbol, PERIOD_CURRENT, shift+3);

      if (long_c1 && long_c2 && long_c3 && long_c4 && long_c5 && long_c6 && long_c7)
      {
          dax_long = true;
          if (gSellTicket > 0)
          {
              Print("Closing Sell Position: ", gSellTicket);
              CloseOrder(gSellTicket);
          }
      }
      else if (short_c1 && short_c2 && short_c3 && short_c4 && short_c5 && short_c6 && short_c7)
      {
          dax_short = true;
          if (gBuyTicket > 0)
          {
              Print("Closing Buy Position: ", gBuyTicket);
              CloseOrder(gBuyTicket);
          }
      }
      else
      {
        if (gBuyTicket > 0)
        {
            Print("Closing Buy Position: ", gBuyTicket);
            CloseOrder(gBuyTicket);
        }
        if (gSellTicket > 0)
        {
            Print("Closing Sell Position: ", gSellTicket);
            CloseOrder(gBuyTicket);
        }
      }

      // Create a bool that is true if the number of open positions is 0
      bool no_active_trades = (gSellTicket == 0) && (gBuyTicket == 0);

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

      MqlDateTime current_time;
      TimeToStruct(TimeCurrent(), current_time);

      if (current_time.hour == 23 && neg_swap_long) 
      {
          swap_protection_long = true;
      }
      else if (current_time.hour == 23 && neg_swap_short)
      {
          swap_protection_short = true;
      }


      // If indicators give the signal, sell
      if(dax_short && no_active_trades && more_trades && !swap_protection_short)
      {
          Print("SELL");
          gSellTicket = 0;
        //--- declare and initialize the trade request and result of trade request
          MqlTradeRequest request={};
          MqlTradeResult  result={};
        //--- parameters of request
          request.action   =TRADE_ACTION_DEAL;                     // type of trade operation
          request.symbol   =Symbol();                              // symbol
          request.volume   =lot_size;                                   // volume of 0.2 lot
          request.type     =ORDER_TYPE_SELL;                       // order type
          request.price    =SymbolInfoDouble(Symbol(),SYMBOL_BID); // price for opening
          request.deviation=5;                                     // allowed deviation from the price
          request.magic    =EXPERT_MAGIC;                          // MagicNumber of the order
          request.tp       =SymbolInfoDouble(Symbol(),SYMBOL_BID) - (takeprofit * ATR_Buffer[0])
          request.sl       =SymbolInfoDouble(Symbol(),SYMBOL_BID) + (stoploss * ATR_Buffer[0])
        //--- send the request
          if(OrderSend(request, result))
          {
              // Check the return code
              if(result.retcode == 0) // Assuming 0 indicates success
              {
                  // Order was placed successfully
                  gSellTicket = result.order;
                  Print("Order placed successfully: ", result.order);
              }
              else
              {
                  // An error occurred
                  Print("Order placement failed: ", result.comment);
              }
          }
          else
          {
              Print("Failed to send the request");
          }
          
      // If indicators give the signal, buy
      else if(dax_long && no_active_trades && more_trades && !swap_protection_long)
      {
          Print("BUY");
         gBuyTicket = 0;
        //--- declare and initialize the trade request and result of trade request
          MqlTradeRequest request={};
          MqlTradeResult  result={};
        //--- parameters of request
          request.action   =TRADE_ACTION_DEAL;                     // type of trade operation
          request.symbol   =Symbol();                              // symbol
          request.volume   =lot_size;                                   // volume of 0.2 lot
          request.type     =ORDER_TYPE_BUY;                       // order type
          request.price    =SymbolInfoDouble(Symbol(),SYMBOL_BID); // price for opening
          request.deviation=5;                                     // allowed deviation from the price
          request.magic    =EXPERT_MAGIC;                          // MagicNumber of the order
          request.tp       =SymbolInfoDouble(Symbol(),SYMBOL_BID) + (takeprofit * ATR_Buffer[0])
          request.sl       =SymbolInfoDouble(Symbol(),SYMBOL_BID) - (stoploss * ATR_Buffer[0])
        //--- send the request
          if(OrderSend(request, result))
          {
              // Check the return code
              if(result.retcode == 0) // Assuming 0 indicates success
              {
                  // Order was placed successfully
                  gBuyTicket = result.order;
                  Print("Order placed successfully: ", result.order);
              }
              else
              {
                  // An error occurred
                  Print("Order placement failed: ", result.comment);
              }
          }
          else
          {
              Print("Failed to send the request");
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

MqlCalendarValue getnewsevents()
{
    // Get current symbol
    string currentSymbol = Symbol(); 
    
    // Extract currency codes (e.g., EURUSD -> EUR and USD)
    string currencyCode_1 = StringSubstr(currentSymbol, 0, 3); // Base currency (e.g., EUR)
    string currencyCode_2 = StringSubstr(currentSymbol, 3, 3); // Quote currency (e.g., USD)

    // Get current week range
    datetime dateFrom = iTime(currentSymbol, PERIOD_W1, 0); // Start of the current week 
    datetime dateTo = dateFrom + PeriodSeconds(PERIOD_W1) - 1; // End of the current week

    // Array to hold fetched event values for both currencies
    MqlCalendarValue eventValues_1[];
    MqlCalendarValue eventValues_2[];
    
    // Fetch calendar events for the first currency
    if (!CalendarValueHistory(eventValues_1, dateFrom, dateTo, currencyCode_1))
    {
        Print("Failed to get calendar events for ", currencyCode_1, ": Error code: ", GetLastError());
    }

    // Fetch calendar events for the second currency
    if (!CalendarValueHistory(eventValues_2, dateFrom, dateTo, currencyCode_2))
    {
        Print("Failed to get calendar events for ", currencyCode_2, ": Error code: ", GetLastError());
    }

    // Combined array to hold high impact events for both currencies
    MqlCalendarValue highImpactEvents[];
    
    // Merge high-impact events from the first currency
    for (int i = 0; i < ArraySize(eventValues_1); i++)
    {
        if (eventValues_1[i].importance == CALENDAR_IMPORTANCE_HIGH)
        {
            ArrayPush(highImpactEvents, eventValues_1[i]);
        }
    }

    // Merge high-impact events from the second currency
    for (int i = 0; i < ArraySize(eventValues_2); i++)
    {
        if (eventValues_2[i].importance == CALENDAR_IMPORTANCE_HIGH)
        {
            ArrayPush(highImpactEvents, eventValues_2[i]);
        }
    }

    // Return the combined high-impact events array
    return highImpactEvents;
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

void CloseOrder(ulong ticket) {
    // Declare and initialize trade request and result
    MqlTradeRequest request;
    MqlTradeResult result;

    // Zero memory for request and result
    ZeroMemory(request);
    ZeroMemory(result);

    // Set the trade request parameters
    request.action = TRADE_ACTION_DEAL; // Type of trade operation
    request.position = ticket; // Ticket of the position to close
    request.magic = EXPERT_MAGIC; // Magic number of the position

    // Get the position type to determine order type and price
    ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
    string symbol = PositionGetString(POSITION_SYMBOL); // Get the position symbol
    double volume = PositionGetDouble(POSITION_VOLUME); // Get the position volume
    int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS); // Number of decimal places

    // Set price and order type based on position type
    if (type == POSITION_TYPE_BUY) {
        request.price = SymbolInfoDouble(symbol, SYMBOL_BID); // Price for closing a Buy position
        request.type = ORDER_TYPE_SELL; // Set order type to Sell
    } else {
        request.price = SymbolInfoDouble(symbol, SYMBOL_ASK); // Price for closing a Sell position
        request.type = ORDER_TYPE_BUY; // Set order type to Buy
    }

    // Send the request to close the position
    if (!OrderSend(request, result)) {
        PrintFormat("OrderSend error %d", GetLastError()); // Output the error code if unable to send
    } else {
        PrintFormat("Position closed: retcode=%u deal=%I64u order=%I64u", result.retcode, result.deal, result.order);
    }
}

