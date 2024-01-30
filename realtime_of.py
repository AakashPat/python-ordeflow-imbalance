import websocket
import threading
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import sys
from rich.console import Console
from rich.table import Table
from playsound import playsound

console = Console()


# Set the maximum number of ticks to keep in the DataFrames
max_ticks = 1000

tick_size = 10 #0.05 * 200
precision = 10
lot_size = 100

global cumulative_sell_df, cumulative_buy_df, cumulative_df, combined_candle_df

cumulative_sell_df = pd.DataFrame(columns=["T", "v", "p", "S"])
cumulative_buy_df = pd.DataFrame(columns=["T", "v", "p", "S"])
cumulative_df = pd.DataFrame(columns=["T", "v", "p", "S"])
combined_candle_df = pd.DataFrame(columns=["T", "price_bin", "buy_volume", "sell_volume"])


def speakMessage(message):
    speech_thread = threading.Thread(target=speak, args=(message,)).start()

def speak(message):
    print('SPEAKING MESSAGE: ', message)
    playsound(message)

    if message == 'up.wav':
        console.print("[bold green]Buying Imbalance detected at the current price_bin ![/bold green]")
    elif message == 'down.wav':
         console.print("[bold red]Selling Imbalance detected at the current price_bin![/bold red]")

    
def clear_line():
    sys.stdout.write("\033[F")  # Move the cursor up one line
    sys.stdout.write("\033[K")  # Clear the line

def print_updated_dataframe(df, timeframe):
    clear_line()
    print(f"\rUpdated Combined Candle DataFrame - {timeframe} Intervals:")
    # print(df[['Selling_Imbalance', 'v_sell', 'v_buy', 'Buying_Imbalance']])
    # print(df[['SI', 'v_sell', 'v_buy', 'BI', 'DELTA', 'CVD', 'CVD_Candle']])
    print(df[['SI', 'v_sell', 'v_buy', 'prev_v_sell', 'prev_v_buy', 'BI', 'DELTA', 'CVD', 'CVD_Candle']])
    # print(df.iloc[-1]['BI'], df.iloc[-1]['SI'])

    # Print BI found at the current price_bin
    if df.iloc[-1]['BI'] == 1.0:
        speakMessage('up.wav')  # Speak the Buying Imbalance message
        # console.print("[bold green]Buying Imbalance detected at the current price_bin ![/bold green]")
    # Print SI found at the current price_bin
    if df.iloc[-1]['SI'] == 1.0:
        speakMessage('down.wav')  # Speak the Selling Imbalance message
        # console.print("[bold red]Selling Imbalance detected at the current price_bin![/bold red]")

def process_tick(tick_data):
    # print('PROCESSING TICK!')
    # print(tick_data)
    global cumulative_sell_df, cumulative_buy_df, cumulative_df

    # Convert tick data to DataFrame
    tick_df = pd.DataFrame([tick_data], columns=["T", "v", "p", "S"]).dropna()
    tick_df["T"] = pd.to_datetime(tick_df["T"], unit="ms")  # Convert timestamp to datetime

    # Convert volume and price columns to numeric types using NumPy
    tick_df["v"] = np.array(pd.to_numeric(tick_df["v"])/100)
    # print(tick_df)


    # print(pd.to_numeric(tick_df["v"])/100)
    tick_df["p"] = np.array(pd.to_numeric(tick_df["p"]))

    if not tick_df.dropna().empty:
        cumulative_df = pd.concat([cumulative_df, tick_df], ignore_index=True, sort=False, verify_integrity=True)

         # Trim the DataFrames to keep only the last 'max_ticks' entries
        cumulative_df = cumulative_df.tail(max_ticks)
        cumulative_sell_df = cumulative_sell_df.tail(max_ticks)
        cumulative_buy_df = cumulative_buy_df.tail(max_ticks)

        # Check if it's a buy or sell tick
        if tick_data["S"] == "Sell":
            cumulative_sell_df = pd.concat([cumulative_sell_df, tick_df], ignore_index=True, sort=False)
        elif tick_data["S"] == "Buy":
            cumulative_buy_df = pd.concat([cumulative_buy_df, tick_df], ignore_index=True, sort=False)
    calculate_and_print_aggregates(timeframe='1Min')
    print('Current LTP: ', tick_data["p"])


def calculate_and_print_aggregates(timeframe='1Min'):
    try:
        global cumulative_sell_df, cumulative_buy_df, combined_candle_df

        min_price = cumulative_sell_df["p"].min().round(0)
        max_price = cumulative_sell_df["p"].max().round(0)

        # Calculate bin edges using floating-point division and rounding
        adjusted_min_price = (min_price // tick_size) * tick_size
        adjusted_max_price = ((max_price // tick_size) + 1) * tick_size

        bin_edges = np.arange((adjusted_min_price / tick_size) * tick_size,
                            (adjusted_max_price / tick_size) * tick_size + tick_size,
                            tick_size)

        # Define bin edges
        # bin_edges = range(int(cumulative_sell_df["p"].min() // tick_size) * tick_size, int(cumulative_sell_df["p"].max() // tick_size + 2) * tick_size, tick_size)

        cumulative_sell_df["price_bin"] = pd.cut(cumulative_sell_df["p"], bins=bin_edges, labels=False) * tick_size + bin_edges[0]
        cumulative_buy_df["price_bin"] = pd.cut(cumulative_buy_df["p"], bins=bin_edges, labels=False) * tick_size + bin_edges[0]

        # Example: Round price bin values to 2 decimal places
        # cumulative_sell_df["price_bin"] = cumulative_sell_df["price_bin"].round(2)
        # cumulative_buy_df["price_bin"] = cumulative_buy_df["price_bin"].round(2)

        # Select the appropriate aggregation frequency based on the timeframe parameter
        if timeframe == '1Min':
            frequency = '1Min'
        elif timeframe == '5Min':
            frequency = '5Min'
        elif timeframe == '15Min':
            frequency = '15Min'
        else:
            raise ValueError("Invalid timeframe. Supported values: '1Min', '5Min', '15Min'.")

        # Aggregate data over 1-minute intervals using NumPy operations
        cumulative_sell_aggregated = cumulative_sell_df.groupby([pd.Grouper(key='T', freq=frequency), 'price_bin']).agg({'v': 'sum'})
        cumulative_buy_aggregated = cumulative_buy_df.groupby([pd.Grouper(key='T', freq=frequency), 'price_bin']).agg({'v': 'sum'})

        # Merge the aggregated DataFrames
        combined_candle_df = pd.merge(cumulative_sell_aggregated, cumulative_buy_aggregated, how='outer', left_index=True, right_index=True, suffixes=('_sell', '_buy'))

        # Fill NaN values with 0
        combined_candle_df = combined_candle_df.fillna(0)

        # Calculate Imbalance
        combined_candle_df['prev_v_sell'] = combined_candle_df['v_sell'].shift(1)
        combined_candle_df['prev_v_buy'] = combined_candle_df['v_buy'].shift(-1)
        
        # Calculate Selling and Buying imbalances using NumPy operations
        combined_candle_df['SI'] = np.where(combined_candle_df['v_sell'] > 18 * combined_candle_df['prev_v_buy'], 1, 0)
        combined_candle_df['BI'] = np.where(combined_candle_df['v_buy'] > 18 * combined_candle_df['prev_v_sell'], 1, 0)

        # Calculate Delta between v_buy and v_sell for the same price_bin
        combined_candle_df['DELTA'] = combined_candle_df['v_buy'] - combined_candle_df['v_sell']

        combined_candle_df['CVD_Candle'] = combined_candle_df.groupby(level=0)['DELTA'].cumsum()

        
        # Drop unnecessary columns
        # combined_candle_df.drop(columns=['prev_v_sell', 'prev_v_buy'], inplace=True)

        # Calculate cumulative delta for the entire candle
        combined_candle_df['CVD'] = combined_candle_df.groupby(level=0)['DELTA'].cumsum()

        # Calculate OHLC values for each candle
        # ohlc_candle_df = combined_candle_df.groupby(level=0)['price_bin'].agg([('Open', 'first'), ('High', 'max'), ('Low', 'min'), ('Close', 'last')])

        # Reindex to include only the bins within the calculated High and Low values
        # selected_bins = range(int(ohlc_candle_df['Low'].min()), int(ohlc_candle_df['High'].max()) + 1)
        # combined_candle_df = combined_candle_df.reindex(pd.MultiIndex.from_product([combined_candle_df.index.levels[0], selected_bins], names=['T', 'price_bin']), fill_value=0)


         # Reindex to include all bins and fill NaN values with 0
        # all_bins = pd.Index(bin_edges, name='price_bin')
        # combined_candle_df = combined_candle_df.reindex(pd.MultiIndex.from_product([combined_candle_df.index.levels[0], all_bins], names=['T', 'price_bin']), fill_value=0)

        # Sort price_bin in descending order within each timestamp group
        combined_candle_df.sort_values(by=['T', 'price_bin'], ascending=[True, False], inplace=True)

        # Trim the combined_candle_df to keep only the last 2 entries
        combined_candle_df = combined_candle_df.tail(100)
        
        # Print combined DataFrame
        print(f"\nCombined Candle DataFrame - {timeframe} Intervals:")
        print_updated_dataframe(combined_candle_df, timeframe)
    except Exception as e:
        # Do nothing
        pass
        # print(e)

def createWebsocketConnection():
    global cumulative_sell_df, cumulative_buy_df, combined_candle_df

    def on_message(ws, message):
        tick_data = json.loads(message)
        # Check if the received message is a tick update
        if "topic" in tick_data and tick_data["topic"] == "publicTrade.BTCUSD" and "data" in tick_data:
            for tick in tick_data["data"]:
                process_tick(tick)
            # process_tick(tick_data["data"][0])

    def on_open(ws):
        try:
            print("WEBSOCKET STARTED!")
            ws.send('{"op":"subscribe","args":["publicTrade.BTCUSD"]}')

        except Exception as e:
            print("WEBSOCKET ERROR inside on open!")
            print(e)
    def on_error(ws, err):
        print("WEBSOCKET ERROR!")
        print(err)
        # Restart the WebSocket connection after a 2-second delay
        time.sleep(2)
        # start_thread()

    def on_close(ws, close_status_code, close_msg):
        print("WEBSOCKET CLOSED!")
        # Restart the WebSocket connection after a 2-second delay
        time.sleep(2)
        start_thread()

    try:
        web_url = f'wss://stream.bybit.com/v5/public/inverse'
        print(web_url)
        ws = websocket.WebSocketApp(web_url)
        ws.on_message = on_message
        ws.on_open = on_open
        ws.on_error = on_error
        ws.on_close = on_close
        ws.run_forever()
    except Exception as e:
        print(e)

# Start a thread to create a websocket connection
def start_thread():
    t = threading.Thread(target=createWebsocketConnection, args=()).start()

# Start threads
start_thread()

