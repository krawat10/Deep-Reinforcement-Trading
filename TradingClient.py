import decimal

from dotenv import dotenv_values
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from enum import Enum


class OrderType(Enum):
    LONG = 'LONG'
    SHORT = 'SHORT'
    NONE = 'NONE'


class TradingManager:
    def __init__(self):
        api_key = dotenv_values('.env')['api-key']
        secret_key = dotenv_values('.env')['secret-key']
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.closing_position = False

    def close_all_positions(self):
        # Get all open positions
        positions = self.trading_client.get_all_positions()

        # Iterate through positions and close them
        for position in positions:
            symbol = position.symbol
            qty_to_close = abs(float(position.qty))  # Ensure positive quantity
            side = OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY  # Determine the side to close

            # Prepare market order to close the position
            market_order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty_to_close,
                side=side,
                time_in_force=TimeInForce.GTC
            )

            # Submit the market order to close the position
            market_order = self.trading_client.submit_order(
                order_data=market_order_data
            )

        # Print a message or take any additional actions as needed
        print("All open positions have been closed.")

    def check_order_type(self, ticket: str):
        # Check if there are any open positions for the given symbol
        positions = self.trading_client.get_all_positions()
        for position in positions:
            if position.symbol == ticket:
                if float(position.qty) > 0:
                    return OrderType.LONG
                elif float(position.qty) < 0:
                    return OrderType.SHORT

        # If there are no positions, return NONE
        return OrderType.NONE

    def close_short_positions(self, ticket: str):
        if self.closing_position:
            return  # Prevent recursion
        self.closing_position = True
        # Close short positions for the given symbol
        positions = self.trading_client.get_all_positions()
        for position in positions:
            if position.symbol == ticket and float(position.qty) < 0:
                # Calculate the quantity to close the short position
                qty_to_close = abs(float(position.qty))

                # Submit a market order to close the short position
                self.submit_long(ticket, qty_to_close)
        self.closing_position = False

    def submit_long(self, ticket: str, qty=1.0):
        # Check the existing order type
        existing_order_type = self.check_order_type(ticket)

        # If there are existing short positions, close them before going long
        if existing_order_type == OrderType.SHORT:
            self.close_short_positions(ticket)

        # If there are no existing long positions, open a new long position
        if existing_order_type != OrderType.LONG:
            # Preparing market order
            market_order_data = MarketOrderRequest(
                symbol=ticket,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )

            # Submit the long market order
            market_order = self.trading_client.submit_order(
                order_data=market_order_data
            )

    def close_long_positions(self, ticket: str):
        if self.closing_position:
            return  # Prevent recursion
        self.closing_position = True
        # Close long positions for the given symbol
        positions = self.trading_client.get_all_positions()
        for position in positions:
            if position.symbol == ticket and float(position.qty) > 0:
                # Calculate the quantity to close the long position
                qty_to_close = abs(float(position.qty))

                # Submit a market order to close the long position
                self.submit_short(ticket, qty_to_close)
        self.closing_position = False

    def submit_short(self, ticket: str, qty=1.0):
        # Check the existing order type
        existing_order_type = self.check_order_type(ticket)

        # If there are existing long positions, close them before going short
        if existing_order_type == OrderType.LONG:
            self.close_long_positions(ticket)

        # If there are no existing short positions, open a new short position
        if existing_order_type != OrderType.SHORT:
            # Preparing market order
            market_order_data = MarketOrderRequest(
                symbol=ticket,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )

            # Submit the short market order
            market_order = self.trading_client.submit_order(
                order_data=market_order_data
            )
