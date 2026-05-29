"""Quick test to confirm poolHourDatas API returns data."""
import sys
sys.path.insert(0, '.')
from dex_utils import POOL, START_TS, END_TS, run_query

QUERY = """
query($pool: String!, $start: Int!, $end: Int!, $first: Int!, $lastId: String!) {
  poolHourDatas(
    first: $first orderBy: id orderDirection: asc
    where: { pool: $pool periodStartUnix_gte: $start periodStartUnix_lte: $end id_gt: $lastId }
  ) { id periodStartUnix volumeUSD feesUSD tvlUSD txCount }
}
"""
print("Testing poolHourDatas query...", flush=True)
data = run_query(QUERY, {"pool": POOL, "start": START_TS, "end": END_TS, "first": 3, "lastId": ""})
rows = data["poolHourDatas"]
print(f"Got {len(rows)} rows", flush=True)
if rows:
    print(f"First: {rows[0]}", flush=True)
    print(f"Last:  {rows[-1]}", flush=True)
print("DONE")
