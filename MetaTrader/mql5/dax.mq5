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
input bool     Max_Payout_Bool=false;
input int      Max_Payout_Amt=12000;
input bool     acct_swap_protection=false;
input string   News_Inputs ="--- News settings ---";
input bool     News_Trading_Allowed=true;
input bool     trailing_stoploss=false;
input int      magic_number=59726816; 

// GLOBAL VARIABLES
ulong      gBuyTicket;
ulong      gSellTicket;
#define  EXPERT_MAGIC magic_number; // MagicNumber of the expert
double   acct_equity = AccountInfoDouble(ACCOUNT_EQUITY);
double   balanceAtLastClose = 0;
bool     acct_protection = false;

double   lot_size = 0;
bool     swap_protection_long = false;
bool     swap_protection_short = false;
datetime maxTime;
int      ATR_handle;
double   ATR_Buffer[];
bool neg_swap_long;
bool neg_swap_short;
MqlCalendarValue all_news_events[];
datetime lastBarTime;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
    
    lastBarTime = 0;


    if (balanceAtLastClose == 0) balanceAtLastClose = AccountInfoDouble(ACCOUNT_BALANCE);

    // Determine which direction "buy" or "sell" the negative swap is in
    double swapRateLong = SymbolInfoDouble(_Symbol, SYMBOL_SWAP_LONG);  // For long positions
    double swapRateShort = SymbolInfoDouble(_Symbol, SYMBOL_SWAP_SHORT);  // For short positions
    PrintFormat("Swap Rate Long: %.5f", swapRateLong);
    PrintFormat("Swap Rate Short: %.5f", swapRateShort);

    neg_swap_long = false;
    neg_swap_short = false;
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
    ArraySetAsSeries(ATR_Buffer, true);

    // get the news for the symbol
    getnewsevents(all_news_events);
    // Iterate through the array to find the maximum time value
    for (int i = 0; i < ArraySize(all_news_events); i++)
    {
        if (all_news_events[i].time > maxTime)
        {
            maxTime = all_news_events[i].time; // Update maxTime if the current time is greater
        }
    }
    for (int i = 0; i < ArraySize(all_news_events); i++)
    {
        Print("--- News Event ", i + 1, " ---");
        Print("ID: ", all_news_events[i].id);
        Print("Time: ", TimeToString(all_news_events[i].time, TIME_DATE | TIME_SECONDS));
        Print("-----------------------");
    }

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
    datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0); // Time of the current bar (index 0)
    datetime currentTime = TimeCurrent();

    if ((currentBarTime > lastBarTime) && (!acct_protection))
    {
        bool more_trades = true;
        // Determine if there is any news happening right now
        if (!News_Trading_Allowed)
        {
            Print("Checking for news");

            if (ArraySize(all_news_events) != 0)
            {
                for (int i = 0; i < ArraySize(all_news_events); i++)
                {
                    // Calculate the time difference
                    long timeDifference = MathAbs(all_news_events[i].time - currentTime); // Get the absolute value of the difference

                    // Check if the difference is less than or equal to 15 minutes (15 * 60 seconds = 900 seconds)
                    if (timeDifference <= 900) // 15 minutes in seconds
                    {
                        more_trades = false;
                        break; // Break since I only need to find one match
                    }
                }
            }
            else
            {
                Print("No news events found or there was an error fetching them.");
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
        double max_Total_Drawdown_Amt = max_Total_Drawdown_Perc * Initial_Acct_Size;
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

        double current_bar_open = iOpen(_Symbol, PERIOD_CURRENT, 1);
        double current_bar_close = iClose(_Symbol, PERIOD_CURRENT, 1);
        // Print("Current Bar Close: ", current_bar_close);

        double risk_lot_size = NormalizeDouble((risk_per_trade / stoploss), 2);
        double calculated_lot_size = MathMin(max_lot_size, risk_lot_size);
        

        Print("more_trades: ", more_trades);


        //+------------------------------------------------------------------+
        //| ENTRY CONDITIONS FROM INDICATORS                                 |
        //+------------------------------------------------------------------+
        bool dax_long = false;
        bool dax_short = false;
        int shift = 1;
        bool long_c1 = iHigh(_Symbol, 0, shift) > iHigh(_Symbol, 0, shift+1);
        bool long_c2 = iHigh(_Symbol, 0, shift+1) > iLow(_Symbol, 0, shift);
        bool long_c3 = iLow(_Symbol, 0, shift) > iHigh(_Symbol, 0, shift+2);
        bool long_c4 = iHigh(_Symbol, 0, shift+2) > iLow(_Symbol, 0, shift+1);
        bool long_c5 = iLow(_Symbol, 0, shift+1) > iHigh(_Symbol, 0, shift+3);
        bool long_c6 = iHigh(_Symbol, 0, shift+3) > iLow(_Symbol, 0, shift+2);
        bool long_c7 = iLow(_Symbol, 0, shift+2) > iLow(_Symbol, 0, shift+3);
        bool short_c1 = iLow(_Symbol, 0, shift) < iLow(_Symbol, 0, shift+1);
        bool short_c2 = iLow(_Symbol, 0, shift+1) < iHigh(_Symbol, 0, shift);
        bool short_c3 = iHigh(_Symbol, 0, shift) < iLow(_Symbol, 0, shift+2);
        bool short_c4 = iLow(_Symbol, 0, shift+2) < iHigh(_Symbol, 0, shift+1);
        bool short_c5 = iHigh(_Symbol, 0, shift+1) < iLow(_Symbol, 0, shift+3);
        bool short_c6 = iLow(_Symbol, 0, shift+3) < iHigh(_Symbol, 0, shift+2);
        bool short_c7 = iHigh(_Symbol, 0, shift+2) < iHigh(_Symbol, 0, shift+3);

        if (long_c1 && long_c2 && long_c3 && long_c4 && long_c5 && long_c6 && long_c7)
        {
            dax_long = true;
            Print("Dax Long", dax_long);
            if (gSellTicket > 0)
            {
                Print("Closing Sell Position: ", gSellTicket);
                CloseOrder(gSellTicket);
            }
        }
        else if (short_c1 && short_c2 && short_c3 && short_c4 && short_c5 && short_c6 && short_c7)
        {
            dax_short = true;
            Print("Dax Short", dax_short);
            if (gBuyTicket > 0)
            {
                Print("Closing Buy Position: ", gBuyTicket);
                CloseOrder(gBuyTicket);
            }
        }
        else
        {
            Print("No Signal");
            // if (gBuyTicket > 0)
            // {
            //     Print("Closing Buy Position: ", gBuyTicket);
            //     CloseOrder(gBuyTicket);
            // }
            // if (gSellTicket > 0)
            // {
            //     Print("Closing Sell Position: ", gSellTicket);
            //     CloseOrder(gSellTicket);
            // }
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

        TimeToStruct(TimeCurrent(), current_time);

        if (current_time.hour == 23 && neg_swap_long) 
        {
            swap_protection_long = true;
        }
        else if (current_time.hour == 23 && neg_swap_short)
        {
            swap_protection_short = true;
        }
        Print("Calculated Lot Size: ", calculated_lot_size, " Lot Size: ", lot_size);

        // updating the ATR_Buffer
        CopyBuffer(ATR_handle, 0, 0, 1, ATR_Buffer);
        // If indicators give the signal, sell
        if (dax_short && no_active_trades && more_trades && !swap_protection_short)
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
            request.tp       =SymbolInfoDouble(Symbol(),SYMBOL_BID) - (takeprofit * ATR_Buffer[0]);
            request.sl       =SymbolInfoDouble(Symbol(),SYMBOL_BID) + (stoploss * ATR_Buffer[0]);
            //--- send the request
            if(OrderSend(request, result))
            {
                // Check the return code
                if (result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED) // Assuming 0 indicates success
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
            request.price    =SymbolInfoDouble(Symbol(),SYMBOL_ASK); // price for opening
            request.deviation=5;                                     // allowed deviation from the price
            request.magic    =EXPERT_MAGIC;                          // MagicNumber of the order
            request.tp       =SymbolInfoDouble(Symbol(),SYMBOL_ASK) + (takeprofit * ATR_Buffer[0]);
            request.sl       =SymbolInfoDouble(Symbol(),SYMBOL_ASK) - (stoploss * ATR_Buffer[0]);
            //--- send the request
            if(OrderSend(request, result))
            {
                // Check the return code
                if (result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED) // Assuming 0 indicates success
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
        }
        //+------------------------------------------------------------------+
        //| TRAILING STOPLOSS                                                |
        //+------------------------------------------------------------------+

        if (trailing_stoploss)
        { // Check if trailing stop is enabled only once

            if (gBuyTicket > 0)
            {
                if (PositionSelectByTicket(gBuyTicket)) // Select the buy position
                {
                    if (SymbolInfoDouble(_Symbol, SYMBOL_BID) > PositionGetDouble(POSITION_PRICE_OPEN))
                    {
                        modify_stoploss(gBuyTicket);
                    }
                }
                else
                {
                    Print("Failed to select buy position: ", GetLastError());
                }
            }

            if (gSellTicket > 0)
            {
                if (PositionSelectByTicket(gSellTicket)) // Select the sell position
                {
                    if (SymbolInfoDouble(_Symbol, SYMBOL_ASK) < PositionGetDouble(POSITION_PRICE_OPEN))
                    {
                        modify_stoploss(gSellTicket);
                    }
                }
                else
                {
                    Print("Failed to select sell position: ", GetLastError());
                }
            }
        }
        lastBarTime = currentBarTime;
    }
  // Is the order still open?
  if (gBuyTicket > 0)
  {
      if (!PositionSelectByTicket(gBuyTicket))
      {
        gBuyTicket = 0;
      }
  }
  if (gSellTicket > 0)
  {
      if (!PositionSelectByTicket(gSellTicket))
      {
        gSellTicket = 0;
      }
  }



  // Update the max equity
  if (AccountInfoDouble(ACCOUNT_EQUITY) > acct_equity) 
  {
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
      (current_time.hour == 23) && (current_time.min == 55) && (acct_swap_protection))) 
    {
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

  // Get the day of the week and the hour
  int dayOfWeek = current_time.day_of_week; // 0 = Sunday, 1 = Monday, ..., 6 = Saturday

  // Check if it's Monday and max Time from the events table is less than the current time
  if ((dayOfWeek == 1) && (maxTime < TimeCurrent()))
  {
    getnewsevents(all_news_events);

    // Iterate through the array to find the maximum time value
    for (int i = 0; i < ArraySize(all_news_events); i++)
    {
        if (all_news_events[i].time > maxTime)
        {
            maxTime = all_news_events[i].time; // Update maxTime if the current time is greater
        }
    }
  }
}

//+------------------------------------------------------------------+

void getnewsevents(MqlCalendarValue &highImpactEvents[])
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
    if (!CalendarValueHistory(eventValues_1, dateFrom, dateTo, NULL, currencyCode_1))
    {
        Print("Failed to get calendar events for ", currencyCode_1, ": Error code: ", GetLastError());
    }

    // Fetch calendar events for the second currency
    if (!CalendarValueHistory(eventValues_2, dateFrom, dateTo, NULL, currencyCode_2))
    {
        Print("Failed to get calendar events for ", currencyCode_2, ": Error code: ", GetLastError());
    }

    // Combined array to hold high impact events for both currencies
    ArrayFree(highImpactEvents);
    MqlCalendarEvent event_1;

    // Merge high-impact events from the first currency
    for (int i = 0; i < ArraySize(eventValues_1); i++)
    {
        CalendarEventById(eventValues_1[i].event_id, event_1);
        if (event_1.importance == CALENDAR_IMPORTANCE_HIGH)
        {
            PushValueToMQLCalendarArray(highImpactEvents, eventValues_1[i]);
        }
    }
    MqlCalendarEvent event_2;
    // Merge high-impact events from the second currency
    for (int i = 0; i < ArraySize(eventValues_2); i++)
    {
        CalendarEventById(eventValues_2[i].event_id, event_2);
        if (event_2.importance == CALENDAR_IMPORTANCE_HIGH)
        {
            PushValueToMQLCalendarArray(highImpactEvents, eventValues_2[i]);
        }
    }
}

// void CloseOrder(ulong ticket) {
//     // Declare and initialize trade request and result
//     MqlTradeRequest request;
//     MqlTradeResult result;

//     // Zero memory for request and result
//     ZeroMemory(request);
//     ZeroMemory(result);

//     // Set the trade request parameters
//     request.action = TRADE_ACTION_DEAL; // Type of trade operation
//     request.position = ticket; // Ticket of the position to close
//     request.magic = EXPERT_MAGIC; // Magic number of the position

//     // Get the position type to determine order type and price
//     ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
//     string symbol = PositionGetString(POSITION_SYMBOL); // Get the position symbol
//     double volume = PositionGetDouble(POSITION_VOLUME); // Get the position volume
//     int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS); // Number of decimal places

//     // Set price and order type based on position type
//     if (type == POSITION_TYPE_BUY) {
//         request.price = SymbolInfoDouble(symbol, SYMBOL_BID); // Price for closing a Buy position
//         request.type = ORDER_TYPE_SELL; // Set order type to Sell
//     } else {
//         request.price = SymbolInfoDouble(symbol, SYMBOL_ASK); // Price for closing a Sell position
//         request.type = ORDER_TYPE_BUY; // Set order type to Buy
//     }

//     // Send the request to close the position
//     if (!OrderSend(request, result)) 
//     {
//         PrintFormat("OrderSend error %d", GetLastError()); // Output the error code if unable to send
//     } else {
//         PrintFormat("Position closed: retcode=%u deal=%I64u order=%I64u", result.retcode, result.deal, result.order);
//     }
// }
void CloseOrder(ulong ticket)
{
  MqlTradeRequest request;
  MqlTradeResult result;
  ZeroMemory(request);
  ZeroMemory(result);

  if (!PositionSelectByTicket(ticket)) {
    Print("Error: Could not select position with ticket: ", ticket);
    return;
  }

  string symbol = PositionGetString(POSITION_SYMBOL);
  double volume = PositionGetDouble(POSITION_VOLUME);
  ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
  int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
  
  request.action = TRADE_ACTION_DEAL;
  request.position = ticket;
  request.symbol = symbol;
  request.volume = volume;
  request.deviation = 5; // Set your desired slippage tolerance
  request.magic = EXPERT_MAGIC;
  
  if (type == POSITION_TYPE_BUY)
  {
    request.price = NormalizeDouble(SymbolInfoDouble(symbol, SYMBOL_BID), digits);
    request.type = ORDER_TYPE_SELL;
  }
  else if (type == POSITION_TYPE_SELL)
  {
    request.price = NormalizeDouble(SymbolInfoDouble(symbol, SYMBOL_ASK), digits);
    request.type = ORDER_TYPE_BUY;
  }
  else 
  {
    Print("Error: Invalid position type for ticket: ", ticket);
    return;
  }

  Print("Request: action=", request.action,
        ", position=", request.position,
        ", symbol=", request.symbol,
        ", volume=", request.volume,
        ", price=", request.price,
        ", deviation=", request.deviation,
        ", magic=", request.magic);

  if (!OrderSend(request, result))
  {
    PrintFormat("OrderSend error %d", GetLastError());
  }
  else
  {
    PrintFormat("Position closed: retcode=%u deal=%I64u order=%I64u", result.retcode, result.deal, result.order);
  }
}

bool modify_stoploss(ulong pTicket)
{
    // --- Get position information using PositionSelectByTicket
    if (!PositionSelectByTicket(pTicket))
    {
        Print("modify_stoploss(): Failed to select position by ticket: ", pTicket, " Error: ", GetLastError());
        return (false);
    }

    // --- Get required position properties
    ENUM_POSITION_TYPE positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
    double positionOpenPrice = PositionGetDouble(POSITION_PRICE_OPEN);
    double currentStopLoss = PositionGetDouble(POSITION_SL);
    double positionTakeProfit = PositionGetDouble(POSITION_TP); // Get current TP
    double currentPrice = 0;

    // --- Determine current price based on position type
    if (positionType == POSITION_TYPE_BUY)
    {
        currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    }
    else if (positionType == POSITION_TYPE_SELL)
    {
        currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    }
    else
    {
        Print("modify_stoploss(): Invalid position type for ticket: ", pTicket);
        return (false);
    }

    // --- Calculate new stop-loss level
    double newStopLoss = 0;
    if (positionType == POSITION_TYPE_BUY)
    {
        newStopLoss = currentPrice - (stoploss * ATR_Buffer[0]); // stoploss is your input parameter
        if (currentStopLoss == 0 || newStopLoss > currentStopLoss)
        { // Update SL if it's not set or if the new SL is higher (better for us)
            newStopLoss = NormalizeDouble(newStopLoss, _Digits); // Normalize to the correct number of digits
        }
        else
        {
            // No need to modify stop-loss
            return (true);
        }
    }
    else // positionType == POSITION_TYPE_SELL
    {
        newStopLoss = currentPrice + (stoploss * ATR_Buffer[0]); // stoploss is your input parameter
        if (currentStopLoss == 0 || newStopLoss < currentStopLoss)
        { // Update SL if it's not set or if the new SL is lower (better for us)
            newStopLoss = NormalizeDouble(newStopLoss, _Digits); // Normalize to the correct number of digits
        }
        else
        {
            // No need to modify stop-loss
            return (true);
        }
    }

    // --- Modify the position
    MqlTradeRequest request;
    MqlTradeResult result;
    ZeroMemory(request);
    ZeroMemory(result);

    request.action = TRADE_ACTION_SLTP; // Modify Stop Loss and Take Profit
    request.position = pTicket;
    request.sl = newStopLoss;
    request.tp = positionTakeProfit; // Keep the existing TP
    request.symbol = _Symbol;
    // request.type = positionType;

    if (OrderSend(request, result))
    {
        if (result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED)
        {
            Print("modify_stoploss(): Stop-loss for ticket ", pTicket, " modified successfully. New SL: ", newStopLoss);
            return (true);
        }
        else
        {
            Print("modify_stoploss(): Failed to modify stop-loss for ticket ", pTicket, ". Retcode: ", result.retcode, ", Comment: ", result.comment);
            return (false);
        }
    }
    else
    {
        Print("modify_stoploss(): OrderSend failed for ticket ", pTicket, ". Error: ", GetLastError());
        return (false);
    }
}

// Function to push a value to an MqlCalendarValue array
void PushValueToMQLCalendarArray(MqlCalendarValue &arr[], MqlCalendarValue &value) {
    // Resize the array to add one more element
    int currentSize = ArraySize(arr);
    ArrayResize(arr, currentSize + 1);

    // Assign the value to the last index
    arr[currentSize] = value;
}