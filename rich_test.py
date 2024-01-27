from rich.console import Console
from rich.table import Table
import pandas as pd
import threading
import time

console = Console()

# Example DataFrame
data = {
    'v_sell': [11.79],
    'v_buy': [10.00],
    'prev_v_buy': [None],
    'SI': [0.00],
    'BI': [0.00],
    'DELTA': [-1.79],
    'CVD': [-1.79],
}

index = pd.MultiIndex.from_tuples([(pd.Timestamp('2024-01-27 09:05:00'), 41450.0)], names=['T', 'price_bin'])

df = pd.DataFrame(data, index=index)

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

def display_table():
    while True:
        # Print the table header
        console.clear()
        console.print(table)

        # Print the table rows with the latest values from the DataFrame
        for index, row in df.iterrows():
            console.print(
                f"{index[0]}\t{index[1]}\t{row['v_sell']:.2f}\t{row['v_buy']:.2f}\t{'' if pd.isna(row['prev_v_buy']) else row['prev_v_buy']:.2f}\t{row['SI']:.2f}\t{row['BI']:.2f}\t{row['DELTA']:.2f}\t{row['CVD']:.2f}"
            )

        time.sleep(1)  # Adjust the sleep interval as needed

# Start the display thread
threading.Thread(target=display_table).start()

# Simulate changes in the DataFrame
for i in range(5):
    time.sleep(2)
    # Simulate new values in the DataFrame
    df['v_sell'] += 1.0
    df['v_buy'] += 1.0
    df['prev_v_buy'] = df['v_buy'].shift(1)
    df['SI'] = (df['v_sell'] > 3 * df['v_buy']).astype(int)
    df['BI'] = (df['v_buy'] > 3 * df['prev_v_buy']).astype(int)
    df['DELTA'] = df['v_sell'] - df['v_buy']
    df['CVD'] = df['DELTA'].cumsum()

# Keep the main thread running
while True:
    time.sleep(1)
