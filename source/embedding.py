"""
Embedding các chunk văn bản pháp lý và lưu thành FAISS index.

Input : processed/legal_chunks.parquet  (cột: chunk_id, doc_id, chunk_index,
                                          part_index, articles, text)
Output: processed/faiss.index          (FAISS index, cosine similarity)
        processed/chunk_meta.parquet   (metadata map theo thứ tự vector trong index)

Cài đặt thư viện cần thiết:
    pip install faiss-cpu sentence-transformers pandas --break-system-packages
"""

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------
# Cấu hình
# ----------------------------------------------------------------------
CHUNK_PATH = "processed/legal_chunks.parquet"
INDEX_PATH = "processed/faiss.index"
META_PATH = "processed/chunk_meta.parquet"

# Model embedding tiếng Việt (đa ngôn ngữ, chất lượng tốt cho văn bản pháp lý).
# Có thể đổi sang "intfloat/multilingual-e5-base" hoặc
# "bkai-foundation-models/vietnamese-bi-encoder" tuỳ nhu cầu.
MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

BATCH_SIZE = 64
EMBED_DIM = None  # sẽ tự lấy từ model


# ----------------------------------------------------------------------
# Hàm chính
# ----------------------------------------------------------------------
def load_chunks(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    return df


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode + chuẩn hoá L2 để dùng cosine similarity qua inner product."""
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # chuẩn hoá sẵn -> IndexFlatIP = cosine
    )
    return embeddings.astype("float32")


def build_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]

    # Với vài chục nghìn chunk trở xuống: Flat index (chính xác 100%) là đủ nhanh.
    # Nếu số lượng chunk lớn (>100k), có thể đổi sang IVF/HNSW để tăng tốc:
    #   quantizer = faiss.IndexFlatIP(dim)
    #   index = faiss.IndexIVFFlat(quantizer, dim, nlist=100, faiss.METRIC_INNER_PRODUCT)
    #   index.train(embeddings)
    index = faiss.IndexFlatIP(dim)

    # Bọc thêm IndexIDMap để gán ID tuỳ ý (dùng chunk_index) thay vì ID mặc định 0..n-1
    index = faiss.IndexIDMap(index)
    return index


def main():
    print(f"Đang load chunks từ {CHUNK_PATH} ...")
    df = load_chunks(CHUNK_PATH)
    print(f"Tổng số chunk: {len(df)}")

    print(f"Đang load model embedding: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print("Đang encode văn bản thành vector ...")
    embeddings = embed_texts(model, df["text"].tolist())

    print("Đang build FAISS index ...")
    index = build_index(embeddings)

    # ID = vị trí dòng trong df (0..n-1), dùng để tra ngược metadata sau khi search
    ids = np.arange(len(df)).astype("int64")
    index.add_with_ids(embeddings, ids)

    print(f"Lưu index vào {INDEX_PATH} ...")
    faiss.write_index(index, INDEX_PATH)

    # Lưu metadata theo đúng thứ tự ID để tra cứu sau khi search (search trả về id)
    meta_df = df[["chunk_id", "doc_id", "chunk_index", "part_index", "articles", "text"]].copy()
    meta_df["vector_id"] = ids
    meta_df.to_parquet(META_PATH, index=False)
    print(f"Lưu metadata vào {META_PATH} ...")

    print("Hoàn tất!")
    print(f"  - Số vector trong index : {index.ntotal}")
    print(f"  - Chiều vector          : {embeddings.shape[1]}")


# ----------------------------------------------------------------------
# Hàm tiện ích: search thử lại index
# ----------------------------------------------------------------------
def search(query: str, top_k: int = 5):
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)
    meta_df = pd.read_parquet(META_PATH)

    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, top_k)

    results = []
    for score, vid in zip(scores[0], ids[0]):
        if vid == -1:
            continue
        row = meta_df[meta_df["vector_id"] == vid].iloc[0]
        results.append({
            "score": float(score),
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "articles": row["articles"],
            "text": row["text"],
        })
    return results


if __name__ == "__main__":
    main()

    # Ví dụ test nhanh sau khi build xong (bỏ comment để chạy thử):
    # for r in search("điều kiện được nghỉ phép năm", top_k=3):
    #     print(r["score"], r["doc_id"], r["articles"])
    #     print(r["text"][:200])
    #     print("---")