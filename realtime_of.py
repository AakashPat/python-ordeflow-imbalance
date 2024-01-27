import websocket
import threading
import json
import pandas as pd
from datetime import datetime, timedelta
import time

tick_size = 10
global cumulative_sell_df, cumulative_buy_df, cumulative_df, combined_candle_df

cumulative_sell_df = pd.DataFrame(columns=["T", "v", "p", "S"])
cumulative_buy_df = pd.DataFrame(columns=["T", "v", "p", "S"])
cumulative_df = pd.DataFrame(columns=["T", "v", "p", "S"])
combined_candle_df = pd.DataFrame(columns=["T", "price_bin", "buy_volume", "sell_volume"])

def process_tick(tick_data):
    print('PROCESSING TICK!')
    print(tick_data)
    global cumulative_sell_df, cumulative_buy_df, cumulative_df

    # Convert tick data to DataFrame
    tick_df = pd.DataFrame([tick_data], columns=["T", "v", "p", "S"])
    tick_df["T"] = pd.to_datetime(tick_df["T"], unit="ms")  # Convert timestamp to datetime

    # Convert data types
    tick_df["v"] = pd.to_numeric(tick_df["v"])/100  # Volume
    tick_df["p"] = pd.to_numeric(tick_df["p"])  # Price

    cumulative_df = pd.concat([cumulative_df, tick_df], ignore_index=True)

    # Check if it's a buy or sell tick
    if tick_data["S"] == "Sell":
        cumulative_sell_df = pd.concat([cumulative_sell_df, tick_df], ignore_index=True)
        # Perform cumulative calculations or other processing for sell data
    elif tick_data["S"] == "Buy":
        cumulative_buy_df = pd.concat([cumulative_buy_df, tick_df], ignore_index=True)

def calculate_and_print_aggregates(timeframe='1Min'):
    try:
        print(f'CALCULATING AGGREGATES FOR {timeframe} CANDLES!')
        global cumulative_sell_df, cumulative_buy_df, combined_candle_df

        # Group cumulative buy and sell data based on price levels with a tick size of 10
        cumulative_sell_df["price_bin"] = (cumulative_sell_df["p"] // tick_size) * tick_size
        cumulative_buy_df["price_bin"] = (cumulative_buy_df["p"] // tick_size) * tick_size

        # Select the appropriate aggregation frequency based on the timeframe parameter
        if timeframe == '1Min':
            frequency = '1Min'
        elif timeframe == '5Min':
            frequency = '5Min'
        elif timeframe == '15Min':
            frequency = '15Min'
        else:
            raise ValueError("Invalid timeframe. Supported values: '1Min', '5Min', '15Min'.")

        # Aggregate data based on the specified timeframe
        cumulative_sell_aggregated = cumulative_sell_df.groupby([pd.Grouper(key='T', freq=frequency), 'price_bin']).agg({'v': 'sum'})
        cumulative_buy_aggregated = cumulative_buy_df.groupby([pd.Grouper(key='T', freq=frequency), 'price_bin']).agg({'v': 'sum'})

        # Merge the aggregated DataFrames
        combined_candle_df = pd.merge(cumulative_sell_aggregated, cumulative_buy_aggregated, how='outer', left_index=True, right_index=True, suffixes=('_sell', '_buy'))

        # Fill NaN values with 0
        combined_candle_df = combined_candle_df.fillna(0)

        # Calculate Imbalance
        combined_candle_df['prev_v_sell'] = combined_candle_df['v_sell'].shift(1)
        combined_candle_df['prev_v_buy'] = combined_candle_df['v_buy'].shift(1)
        combined_candle_df['Selling_Imbalance'] = (combined_candle_df['v_sell'] > 3 * combined_candle_df['v_buy']).astype(int)
        combined_candle_df['Buying_Imbalance'] = (combined_candle_df['v_buy'] > 3 * combined_candle_df['prev_v_sell']).astype(int)
        combined_candle_df.drop(columns=['prev_v_sell'], inplace=True)

        # Sort price_bin in descending order within each timestamp group
        combined_candle_df.sort_values(by=['T', 'price_bin'], ascending=[True, False], inplace=True)

        # Print combined DataFrame
        print(f"\nCombined Candle DataFrame - {timeframe} Intervals:")
        print(combined_candle_df[['Selling_Imbalance', 'v_sell', 'v_buy', 'Buying_Imbalance']])

    except Exception as e:
        print(e)

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
    def on_error(err):
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

# Start a thread to periodically print aggregates
def start_aggregate_thread():
    while True:

        # If the current time is not a multiple of 1 minute, wait till the next minute
        # time.sleep(60 - datetime.now().second)

        # Calculate and print aggregates
        calculate_and_print_aggregates('1Min')
        # Wait for 1 second by default
        time.sleep(1)

# Start threads
start_thread()
threading.Thread(target=start_aggregate_thread, args=()).start()
