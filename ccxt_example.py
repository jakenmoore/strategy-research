import ccxtpro
import asyncio

async def main():
    exchange = ccxtpro.binance({'enableRateLimit': True})
    while True:
        orderbook = await exchange.watch_order_book('ETH/BTC')
        print(orderbook['asks'][0], orderbook['bids'][0])

asyncio.get_event_loop().run_until_complete(main())
