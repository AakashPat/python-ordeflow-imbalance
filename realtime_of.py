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

# Set the maximum number of ticks to keep in the DataFrames
max_ticks = 1000

tick_size = 10
global cumulative_sell_df, cumulative_buy_df, cumulative_df, combined_candle_df

cumulative_sell_df = pd.DataFrame(columns=["T", "v", "p", "S"])
cumulative_buy_df = pd.DataFrame(columns=["T", "v", "p", "S"])
cumulative_df = pd.DataFrame(columns=["T", "v", "p", "S"])
combined_candle_df = pd.DataFrame(columns=["T", "price_bin", "buy_volume", "sell_volume"])



console = Console()

# Create a Table
table = Table(show_header=True, header_style="bold magenta")

# Define table columns
table.add_column("Timestamp", style="dim", width=25)
table.add_column("Price Bin", style="dim", width=15)
table.add_column("v_sell", justify="right", width=10)
table.add_column("v_buy", justify="right", width=10)
table.add_column("prev_v_buy", justify="right", width=10)
table.add_column("SI", justify="right", width=10)
table.add_column("BI", justify="right", width=10)
table.add_column("DELTA", justify="right", width=10)
table.add_column("CVD", justify="right", width=10)

def display_data(df):
    # console.print(table)
    print(df)

    # Print the table rows with the latest values from the DataFrame
    console.print(df[['SI', 'v_sell', 'v_buy', 'BI', 'DELTA', 'CVD']])

def clear_line():
    sys.stdout.write("\033[F")  # Move the cursor up one line
    sys.stdout.write("\033[K")  # Clear the line

def print_updated_dataframe(df, timeframe):
    clear_line()
    print(f"\rUpdated Combined Candle DataFrame - {timeframe} Intervals:")
    # print(df[['Selling_Imbalance', 'v_sell', 'v_buy', 'Buying_Imbalance']])
    print(df[['SI', 'v_sell', 'v_buy', 'BI', 'DELTA', 'CVD']])



def process_tick(tick_data):
    # print('PROCESSING TICK!')
    # print(tick_data)
    global cumulative_sell_df, cumulative_buy_df, cumulative_df

    # Convert tick data to DataFrame
    tick_df = pd.DataFrame([tick_data], columns=["T", "v", "p", "S"]).dropna()
    tick_df["T"] = pd.to_datetime(tick_df["T"], unit="ms")  # Convert timestamp to datetime

    # Convert volume and price columns to numeric types using NumPy
    tick_df["v"] = np.array(pd.to_numeric(tick_df["v"])/100)
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


def calculate_and_print_aggregates(timeframe='1Min'):
    try:
        global cumulative_sell_df, cumulative_buy_df, combined_candle_df

        # Define bin edges
        bin_edges = range(int(cumulative_sell_df["p"].min() // tick_size) * tick_size, int(cumulative_sell_df["p"].max() // tick_size + 2) * tick_size, tick_size)

        cumulative_sell_df["price_bin"] = pd.cut(cumulative_sell_df["p"], bins=bin_edges, labels=False) * tick_size + bin_edges[0]
        cumulative_buy_df["price_bin"] = pd.cut(cumulative_buy_df["p"], bins=bin_edges, labels=False) * tick_size + bin_edges[0]

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
        combined_candle_df['prev_v_buy'] = combined_candle_df['v_buy'].shift(1)


        # combined_candle_df['SI'] = (combined_candle_df['v_sell'] > 3 * combined_candle_df['v_buy']).astype(int)
        # combined_candle_df['BI'] = (combined_candle_df['v_buy'] > 3 * combined_candle_df['prev_v_sell']).astype(int)
        
        # Calculate Selling and Buying imbalances using NumPy operations
        combined_candle_df['SI'] = np.where(combined_candle_df['v_sell'] > 3 * combined_candle_df['prev_v_buy'], 1, 0)
        combined_candle_df['BI'] = np.where(combined_candle_df['v_buy'] > 3 * combined_candle_df['prev_v_sell'], 1, 0)


        # Check for consecutive imbalances
        if combined_candle_df['SI'].iloc[-1] == 1:
            consecutive_sell_imbalance += 1
            consecutive_buy_imbalance = 0
        elif combined_candle_df['BI'].iloc[-1] == 1:
            consecutive_buy_imbalance += 1
            consecutive_sell_imbalance = 0
        else:
            consecutive_sell_imbalance = 0
            consecutive_buy_imbalance = 0

        # Raise alert if there are 3 consecutive imbalances
        if consecutive_sell_imbalance == 2:
            console.print("[bold red]SELL ALERT: Three consecutive selling imbalances detected![/bold red]")
        elif consecutive_buy_imbalance == 2:
            console.print("[bold green]BUY ALERT: Three consecutive buying imbalances detected![/bold green]")

        # Calculate Delta between v_buy and v_sell for the same price_bin
        combined_candle_df['DELTA'] = combined_candle_df['v_buy'] - combined_candle_df['v_sell']
        
        combined_candle_df.drop(columns=['prev_v_sell'], inplace=True)

        # Sort price_bin in descending order within each timestamp group
        combined_candle_df.sort_values(by=['T', 'price_bin'], ascending=[True, False], inplace=True)

        # Calculate cumulative delta for the entire candle
        combined_candle_df['CVD'] = combined_candle_df.groupby(level=0)['DELTA'].cumsum()

        # Trim the combined_candle_df to keep only the last 2 entries
        combined_candle_df = combined_candle_df.tail(10)

        # Print combined DataFrame
        print(f"\nCombined Candle DataFrame - {timeframe} Intervals:")

        print_updated_dataframe(combined_candle_df, timeframe)
        # Display updated data
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
    def on_close(ws):
        print("WEBSOCKET CLOSED!")

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