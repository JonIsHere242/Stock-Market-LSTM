import backtrader as bt
from ib_insync import IB, Forex
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading

# IB configuration
CurrentWSLip = "192.168.1.9"
CurrentBackTestingPort = 7497
SessionID = 1
localhost = "127.0.0.1"

# Create a new IB app
app = IB()

try:
    print("Attempting to connect to TWS...")
    app.connect(CurrentWSLip, CurrentBackTestingPort, SessionID)
    Connected = True
except Exception as e:
    print(f"Exception during connect: {e}")
    Connected = False

# Sleep for a while to allow connection to complete
time.sleep(5)

if Connected:
    print("Connected to TWS.")
else:
    print("Failed to connect to TWS. Please check settings and try again.")
    exit()

# Verify connection
if not app.isConnected():
    print("Connection verification failed: Not connected to TWS.")
else:
    print("Connection verification successful: Connected to TWS.")

# Disconnect
app.disconnect()
print("Disconnected from TWS.")
