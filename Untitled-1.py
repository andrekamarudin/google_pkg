# %%
from google_api.bigquery import BigQuery

bq = BigQuery(project="fairprice-bigquery")

# %%
# writing to df is slow.
# limit row_cnt to avoid vs code being stuck forever
# trying to get millions of rows into a df
bq.q(
    r"""
    SELECT *
    FROM cdm_grocery.sales_line
    WHERE 1=1 
    AND date_of_order >= "2024-10-01"
    """,
    row_limit=500,
)

# %%
bq.q(
    r"""
SELECT *
FROM _1d306bd5c7c9a2bf2cef9aa7fab9a9466ca6d7e1.anon9ba0fc6a91a95e05e588796904d948b1f499b4071139453340811f755f565739
"""
)
# %%
bq.q(
    r"""
SELECT COUNT(*)
FROM _1d306bd5c7c9a2bf2cef9aa7fab9a9466ca6d7e1.anon9ba0fc6a91a95e05e588796904d948b1f499b4071139453340811f755f565739
"""
)
# %%
