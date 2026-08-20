import pandas as pd

chunks = pd.read_parquet(
    "processed/legal_chunks_v2.parquet"
)

bad = []

for doc_id, g in chunks.groupby("doc_id"):

    indexes = sorted(
        g["chunk_index"].tolist()
    )

    expected = list(range(len(indexes)))

    if indexes != expected:
        bad.append(
            {
                "doc_id": doc_id,
                "indexes": indexes[:20],
                "expected": expected[:20]
            }
        )

print(
    "Documents with bad chunk_index:",
    len(bad)
)

if bad:
    print(bad[:10])

print(
    "Total chunks:",
    len(chunks)
)

print(
    "Unique chunk_id:",
    chunks["chunk_id"].nunique()
)

print(
    "Duplicate chunk_id:",
    chunks["chunk_id"].duplicated().sum()
)