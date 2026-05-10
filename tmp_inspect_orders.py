import asyncio
from collections import Counter
from app.clients.backend import BackendExportClient, crawl_all
from datetime import datetime, timedelta, timezone
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

async def main():
    client = BackendExportClient()
    from_date = (datetime.now(timezone.utc) - timedelta(days=60)).strftime('%Y-%m-%d')
    raw = await crawl_all(client, '/internal/ai/export/order-items', extra_params={'fromDate': from_date}, max_pages=200, page_size=200, timeout_s=10.0)
    print('raw items:', len(raw))
    orders = {}
    for item in raw:
        order_id = str(item.get('orderId') or item.get('order_id') or '')
        raw_mid = item.get('menuItemId') or item.get('menu_item_id')
        if not order_id or raw_mid is None:
            continue
        try:
            mid = int(raw_mid)
        except Exception:
            continue
        orders.setdefault(order_id, set()).add(mid)
    transactions = [tx for tx in orders.values() if len(tx) >= 2]
    print('unique orders total:', len(orders))
    print('multi-item transactions:', len(transactions))
    # distribution of items per tx
    dist = Counter(len(tx) for tx in transactions)
    print('transaction size distribution:', dist.most_common())
    all_items = sorted({mid for tx in transactions for mid in tx})
    print('unique items in transactions:', len(all_items))
    if len(transactions) < 2:
        return
    records = [{mid: (mid in tx) for mid in all_items} for tx in transactions]
    df = pd.DataFrame(records, columns=all_items)
    # try multiple support thresholds
    for s in [0.05, 0.03, 0.01, 0.005, 0.001]:
        try:
            fi = fpgrowth(df, min_support=s, use_colnames=True)
            print(f'support {s}: freq_itemsets count =', len(fi))
            if not fi.empty:
                print(fi.head(10))
        except Exception as e:
            print('fpgrowth error', e)

asyncio.run(main())
