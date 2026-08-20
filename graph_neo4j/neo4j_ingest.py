import argparse
import sys
import re
import unicodedata
from typing import Iterable, List

import pandas as pd
from neo4j import GraphDatabase, Driver
from tqdm import tqdm

from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    BATCH_SIZE,
)


# ============================================================================
# CONFIG
# ============================================================================

METADATA_COLUMNS = [
    "id",
    "title",
    "so_ky_hieu",
    "ngay_ban_hanh",
    "loai_van_ban",
    "ngay_co_hieu_luc",
    "ngay_het_hieu_luc",
    "nguon_thu_thap",
    "ngay_dang_cong_bao",
    "nganh",
    "linh_vuc",
    "co_quan_ban_hanh",
    "chuc_danh",
    "nguoi_ky",
    "pham_vi",
    "thong_tin_ap_dung",
    "tinh_trang_hieu_luc",
]


CHUNK_COLUMNS = [
    "chunk_id",
    "doc_id",
    "chunk_index",
    "part_index",
    "articles",
    "text",
]


# ============================================================================
# UTILITY
# ============================================================================

def clean_records(df: pd.DataFrame) -> List[dict]:
    """
    Chuyển DataFrame -> list[dict].

    NaN/NaT được đổi thành None để Neo4j driver chấp nhận.
    """
    df = df.where(pd.notnull(df), None)
    return df.to_dict("records")


def batched(records: List[dict], size: int) -> Iterable[List[dict]]:
    for i in range(0, len(records), size):
        yield records[i:i + size]


def normalize_id(value):
    """
    Chuẩn hóa ID về string.

    Quan trọng:
        Neo4j phân biệt:
            72
        và
            "72"

    Vì vậy Document.id và Chunk.doc_id phải cùng kiểu.
    """

    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if pd.isna(value):
        return None

    # Tránh trường hợp pandas đọc ID số thành 72.0
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value).strip()


def sanitize_rel_type(raw: str) -> str:
    """
    Chuẩn hóa chuỗi quan hệ.

    Ví dụ:
        'sửa đổi bởi'
        ->
        SUA_DOI_BOI

        'hết hiệu lực bởi'
        ->
        HET_HIEU_LUC_BOI
    """

    s = unicodedata.normalize("NFKD", str(raw))
    s = "".join(
        c for c in s
        if not unicodedata.combining(c)
    )

    s = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        s
    ).strip("_").upper()

    if not s:
        s = "RELATED_TO"

    if s[0].isdigit():
        s = "R_" + s

    return s


def parse_articles(value) -> list:
    """
    Cột articles:

        "12,13a"

    ->

        ["12", "13a"]
    """

    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if pd.isna(value):
        return []

    # Nếu đã là list thì giữ nguyên
    if isinstance(value, list):
        return [
            str(x).strip()
            for x in value
            if x is not None and str(x).strip()
        ]

    return [
        a.strip()
        for a in str(value).split(",")
        if a.strip()
    ]


# ============================================================================
# STEP 1: SCHEMA
# ============================================================================

def setup_schema(driver: Driver):

    statements = [

        # --------------------------------------------------------------------
        # UNIQUE CONSTRAINT
        # --------------------------------------------------------------------

        """
        CREATE CONSTRAINT doc_id_unique IF NOT EXISTS
        FOR (d:Document)
        REQUIRE d.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
        FOR (c:Chunk)
        REQUIRE c.id IS UNIQUE
        """,

        # --------------------------------------------------------------------
        # DOCUMENT INDEX
        # --------------------------------------------------------------------

        """
        CREATE INDEX doc_so_ky_hieu IF NOT EXISTS
        FOR (d:Document)
        ON (d.so_ky_hieu)
        """,

        """
        CREATE INDEX doc_loai_van_ban IF NOT EXISTS
        FOR (d:Document)
        ON (d.loai_van_ban)
        """,

        """
        CREATE INDEX doc_tinh_trang IF NOT EXISTS
        FOR (d:Document)
        ON (d.tinh_trang_hieu_luc)
        """,

        # --------------------------------------------------------------------
        # CHUNK INDEX
        # --------------------------------------------------------------------

        """
        CREATE INDEX chunk_doc_id IF NOT EXISTS
        FOR (c:Chunk)
        ON (c.doc_id)
        """,

        """
        CREATE INDEX chunk_part_index IF NOT EXISTS
        FOR (c:Chunk)
        ON (c.part_index)
        """,

        """
        CREATE INDEX chunk_articles IF NOT EXISTS
        FOR (c:Chunk)
        ON (c.articles)
        """,

        # --------------------------------------------------------------------
        # FULLTEXT
        # --------------------------------------------------------------------

        """
        CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
        FOR (c:Chunk)
        ON EACH [c.text]
        """,

        """
        CREATE FULLTEXT INDEX doc_title_fulltext IF NOT EXISTS
        FOR (d:Document)
        ON EACH [d.title]
        """,
    ]

    with driver.session(database=NEO4J_DATABASE) as session:

        for stmt in statements:
            session.run(stmt)

    print("[OK] Đã tạo constraints & indexes.")


# ============================================================================
# VECTOR INDEX
# ============================================================================

def setup_vector_index(driver: Driver, dim: int):

    stmt = f"""
    CREATE VECTOR INDEX chunk_embedding_idx IF NOT EXISTS
    FOR (c:Chunk)
    ON (c.embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(stmt)

    print(
        f"[OK] Đã tạo vector index cho Chunk.embedding "
        f"(dim={dim})."
    )


# ============================================================================
# STEP 2: INGEST DOCUMENTS
# ============================================================================

def ingest_documents(
    driver: Driver,
    metadata_df: pd.DataFrame
):

    missing = set(METADATA_COLUMNS) - set(metadata_df.columns)

    if missing:

        print(
            f"[WARN] metadata thiếu các cột: "
            f"{missing} (sẽ set None)"
        )

        for col in missing:
            metadata_df[col] = None

    df = metadata_df.copy()

    # ------------------------------------------------------------------------
    # QUAN TRỌNG:
    # Chuẩn hóa Document.id thành string.
    # ------------------------------------------------------------------------

    df["id"] = df["id"].apply(normalize_id)

    # Loại bỏ Document không có ID
    missing_id = df["id"].isna()

    if missing_id.any():

        count = int(missing_id.sum())

        print(
            f"[WARN] Bỏ qua {count} Document không có id."
        )

        df = df.loc[~missing_id].copy()

    records = clean_records(
        df[METADATA_COLUMNS]
    )

    query = """
    UNWIND $rows AS row

    MERGE (d:Document {id: row.id})

    SET
        d.title = row.title,
        d.so_ky_hieu = row.so_ky_hieu,
        d.ngay_ban_hanh = row.ngay_ban_hanh,
        d.loai_van_ban = row.loai_van_ban,
        d.ngay_co_hieu_luc = row.ngay_co_hieu_luc,
        d.ngay_het_hieu_luc = row.ngay_het_hieu_luc,
        d.nguon_thu_thap = row.nguon_thu_thap,
        d.ngay_dang_cong_bao = row.ngay_dang_cong_bao,
        d.nganh = row.nganh,
        d.linh_vuc = row.linh_vuc,
        d.co_quan_ban_hanh = row.co_quan_ban_hanh,
        d.chuc_danh = row.chuc_danh,
        d.nguoi_ky = row.nguoi_ky,
        d.pham_vi = row.pham_vi,
        d.thong_tin_ap_dung = row.thong_tin_ap_dung,
        d.tinh_trang_hieu_luc = row.tinh_trang_hieu_luc
    """

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        for batch in tqdm(
            list(batched(records, BATCH_SIZE)),
            desc="Documents"
        ):

            session.run(
                query,
                rows=batch
            )

    print(
        f"[OK] Đã nạp {len(records)} Document."
    )


# ============================================================================
# STEP 3: INGEST DOCUMENT RELATIONSHIPS
# ============================================================================

def ingest_relationships(
    driver: Driver,
    rels_df: pd.DataFrame
):

    rels_df = rels_df.copy()

    # ------------------------------------------------------------------------
    # Chuẩn hóa ID
    # ------------------------------------------------------------------------

    rels_df["doc_id"] = (
        rels_df["doc_id"]
        .apply(normalize_id)
    )

    rels_df["other_doc_id"] = (
        rels_df["other_doc_id"]
        .apply(normalize_id)
    )

    # ------------------------------------------------------------------------
    # Relationship type
    # ------------------------------------------------------------------------

    rels_df["rel_type"] = (
        rels_df["relationship"]
        .map(sanitize_rel_type)
    )

    records = clean_records(
        rels_df[
            [
                "doc_id",
                "other_doc_id",
                "rel_type",
                "relationship",
            ]
        ]
    )

    # ------------------------------------------------------------------------
    # Gom theo relationship type
    # ------------------------------------------------------------------------

    by_type = {}

    for r in records:

        if (
            r["doc_id"] is None
            or r["other_doc_id"] is None
        ):
            continue

        by_type.setdefault(
            r["rel_type"],
            []
        ).append(r)

    total = 0

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        for rel_type, rows in by_type.items():

            query = f"""
            UNWIND $rows AS row

            MATCH (a:Document {{id: row.doc_id}})
            MATCH (b:Document {{id: row.other_doc_id}})

            MERGE (a)-[r:{rel_type}]->(b)

            SET r.raw_label = row.relationship
            """

            for batch in tqdm(
                list(
                    batched(
                        rows,
                        BATCH_SIZE
                    )
                ),
                desc=f"Rel:{rel_type}"
            ):

                session.run(
                    query,
                    rows=batch
                )

            total += len(rows)

    print(
        f"[OK] Đã nạp {total} relationship, "
        f"gồm {len(by_type)} loại: "
        f"{list(by_type.keys())}"
    )


# ============================================================================
# STEP 4A: INGEST CHUNKS
# ============================================================================

def ingest_chunks(
    driver: Driver,
    chunks_df: pd.DataFrame
):

    """
    Nạp Chunk từ legal_chunks.parquet.

    Expected columns:

        chunk_id
        doc_id
        chunk_index
        part_index
        articles
        text

    Logic quan trọng:

        Nếu doc_id bị null:

            72_c0
              ↓
            72

        Nếu chunk_index bị null:

            72_c0
              ↓
            0

    Sau đó tạo:

        (:Chunk)-[:PART_OF]->(:Document)
    """

    # ------------------------------------------------------------------------
    # CHECK COLUMNS
    # ------------------------------------------------------------------------

    missing = (
        set(CHUNK_COLUMNS)
        - set(chunks_df.columns)
    )

    if missing:

        raise ValueError(
            f"legal_chunks thiếu "
            f"cột bắt buộc: {missing}"
        )

    df = chunks_df.copy()

    # ------------------------------------------------------------------------
    # CHUNK ID
    # ------------------------------------------------------------------------

    df["chunk_id"] = (
        df["chunk_id"]
        .apply(normalize_id)
    )



    df["doc_id"] = (
        df["doc_id"]
        .apply(normalize_id)
    )

    missing_doc_id = df["doc_id"].isna()

    if missing_doc_id.any():

        print(
            f"[WARN] Có "
            f"{int(missing_doc_id.sum())} chunks "
            f"thiếu doc_id."
        )

        print(
            "[INFO] Đang suy ra doc_id từ chunk_id..."
        )

        derived_doc_id = (
            df.loc[missing_doc_id, "chunk_id"]
            .astype(str)
            .str.split("_c")
            .str[0]
        )

        df.loc[
            missing_doc_id,
            "doc_id"
        ] = derived_doc_id

    # ------------------------------------------------------------------------
    # CHUNK INDEX
    #
    # 72_c0 -> 0
    # 72_c1 -> 1
    # 72_c2 -> 2
    # ------------------------------------------------------------------------

    chunk_index_from_id = pd.to_numeric(
        df["chunk_id"]
        .astype(str)
        .str.extract(
            r"_c(\d+)$"
        )[0],
        errors="coerce"
    )

    df["chunk_index"] = pd.to_numeric(
        df["chunk_index"],
        errors="coerce"
    )

    missing_chunk_index = (
        df["chunk_index"].isna()
    )

    if missing_chunk_index.any():

        print(
            f"[WARN] Có "
            f"{int(missing_chunk_index.sum())} chunks "
            f"thiếu chunk_index."
        )

        print(
            "[INFO] Đang suy ra chunk_index từ chunk_id..."
        )

        df.loc[
            missing_chunk_index,
            "chunk_index"
        ] = chunk_index_from_id[
            missing_chunk_index
        ]

    # ------------------------------------------------------------------------
    # PART INDEX
    # ------------------------------------------------------------------------

    df["part_index"] = pd.to_numeric(
        df["part_index"],
        errors="coerce"
    )

    # Nếu part_index null thì mặc định 0
    df["part_index"] = (
        df["part_index"]
        .fillna(0)
        .astype(int)
    )

    # ------------------------------------------------------------------------
    # ARTICLES
    # ------------------------------------------------------------------------

    df["articles"] = (
        df["articles"]
        .apply(parse_articles)
    )

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    bad_chunk_id = df["chunk_id"].isna()

    if bad_chunk_id.any():

        raise ValueError(
            f"Có {int(bad_chunk_id.sum())} chunks "
            f"không có chunk_id."
        )

    bad_doc_id = df["doc_id"].isna()

    if bad_doc_id.any():

        sample = (
            df.loc[
                bad_doc_id,
                "chunk_id"
            ]
            .head(20)
            .tolist()
        )

        raise ValueError(
            "Không thể xác định doc_id cho "
            f"{int(bad_doc_id.sum())} chunks.\n"
            f"Ví dụ: {sample}"
        )

    bad_chunk_index = (
        df["chunk_index"].isna()
    )

    if bad_chunk_index.any():

        sample = (
            df.loc[
                bad_chunk_index,
                "chunk_id"
            ]
            .head(20)
            .tolist()
        )

        raise ValueError(
            "Không thể xác định chunk_index cho "
            f"{int(bad_chunk_index.sum())} chunks.\n"
            f"Ví dụ: {sample}"
        )

    # ------------------------------------------------------------------------
    # LOG SAMPLE
    # ------------------------------------------------------------------------

    print("\n[DEBUG] Sample chunks sau khi chuẩn hóa:")

    print(
        df[
            [
                "chunk_id",
                "doc_id",
                "chunk_index",
                "part_index",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------------
    # RECORDS
    # ------------------------------------------------------------------------

    records = clean_records(
        df[CHUNK_COLUMNS]
    )

    for r in records:

        if r["articles"] is None:
            r["articles"] = []

    # ------------------------------------------------------------------------
    # CYPHER
    # ------------------------------------------------------------------------

    query = """
    UNWIND $rows AS row

    MERGE (c:Chunk {id: row.chunk_id})

    SET
        c.text = row.text,
        c.doc_id = row.doc_id,
        c.chunk_index = row.chunk_index,
        c.part_index = row.part_index,
        c.articles = row.articles

    WITH c, row

    MATCH (d:Document {id: row.doc_id})

    MERGE (c)-[:PART_OF]->(d)
    """

    # ------------------------------------------------------------------------
    # INGEST
    # ------------------------------------------------------------------------

    imported = 0

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        for batch in tqdm(
            list(
                batched(
                    records,
                    BATCH_SIZE
                )
            ),
            desc="Chunks"
        ):

            result = session.run(
                query,
                rows=batch
            )

            # Consume để Neo4j thực thi query ngay
            result.consume()

            imported += len(batch)

    print(
        f"[OK] Đã nạp {imported} Chunk."
    )

    # ------------------------------------------------------------------------
    # VERIFY
    # ------------------------------------------------------------------------

    verify_chunk_relationships(
        driver
    )


# ============================================================================
# STEP 4B: FALLBACK CONTENT -> CHUNKS
# ============================================================================

def ingest_content_as_chunks(
    driver: Driver,
    content_df: pd.DataFrame
):

    """
    Fallback:

        mỗi Document = 1 Chunk

    Ví dụ:

        Document.id = 72

        Chunk.id = 72_c0
        Chunk.doc_id = 72
        Chunk.chunk_index = 0
        Chunk.part_index = 0

    Relationship:

        Chunk -[:PART_OF]-> Document
    """

    df = content_df.copy()

    # ------------------------------------------------------------------------
    # NORMALIZE DOCUMENT ID
    # ------------------------------------------------------------------------

    df["id"] = (
        df["id"]
        .apply(normalize_id)
    )

    # ------------------------------------------------------------------------
    # CREATE CHUNK FIELDS
    # ------------------------------------------------------------------------

    df["chunk_id"] = (
        df["id"]
        .astype(str)
        + "_c0"
    )

    df["doc_id"] = df["id"]

    df["chunk_index"] = 0

    df["part_index"] = 0

    df["articles"] = [
        []
        for _ in range(len(df))
    ]

    # ------------------------------------------------------------------------
    # RECORDS
    # ------------------------------------------------------------------------

    records = clean_records(
        df[CHUNK_COLUMNS]
    )

    for r in records:

        if r["articles"] is None:
            r["articles"] = []

    # ------------------------------------------------------------------------
    # CYPHER
    # ------------------------------------------------------------------------

    query = """
    UNWIND $rows AS row

    MERGE (c:Chunk {id: row.chunk_id})

    SET
        c.text = row.text,
        c.doc_id = row.doc_id,
        c.chunk_index = row.chunk_index,
        c.part_index = row.part_index,
        c.articles = row.articles

    WITH c, row

    MATCH (d:Document {id: row.doc_id})

    MERGE (c)-[:PART_OF]->(d)
    """

    # ------------------------------------------------------------------------
    # INGEST
    # ------------------------------------------------------------------------

    imported = 0

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        for batch in tqdm(
            list(
                batched(
                    records,
                    BATCH_SIZE
                )
            ),
            desc="Chunks"
        ):

            result = session.run(
                query,
                rows=batch
            )

            result.consume()

            imported += len(batch)

    print(
        f"[OK] Đã nạp {imported} Chunk "
        f"(fallback: 1 văn bản = 1 chunk)."
    )

    verify_chunk_relationships(
        driver
    )


# ============================================================================
# VERIFY CHUNK -> DOCUMENT
# ============================================================================

def verify_chunk_relationships(
    driver: Driver
):

    """
    Kiểm tra:

        (:Chunk)-[:PART_OF]->(:Document)

    và tìm các Chunk bị orphan.
    """

    query = """
    MATCH (c:Chunk)

    OPTIONAL MATCH (c)-[r:PART_OF]->(d:Document)

    RETURN
        count(c) AS total_chunks,

        count(
            CASE
                WHEN r IS NOT NULL
                THEN 1
            END
        ) AS chunks_with_part_of,

        count(
            CASE
                WHEN r IS NULL
                THEN 1
            END
        ) AS orphan_chunks
    """

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        record = session.run(
            query
        ).single()

    total = record["total_chunks"]
    connected = record["chunks_with_part_of"]
    orphan = record["orphan_chunks"]

    print("\n" + "=" * 70)
    print("VERIFY CHUNK -> DOCUMENT")
    print("=" * 70)

    print(
        f"Total Chunk       : {total}"
    )

    print(
        f"Chunk có PART_OF  : {connected}"
    )

    print(
        f"Chunk bị orphan   : {orphan}"
    )

    if orphan > 0:

        print(
            "\n[WARN] Vẫn còn Chunk chưa có "
            "PART_OF -> Document."
        )

        sample_query = """
        MATCH (c:Chunk)

        OPTIONAL MATCH (c)-[r:PART_OF]->(d:Document)

        WITH c, d
        WHERE d IS NULL

        RETURN
            c.id AS chunk_id,
            c.doc_id AS doc_id,
            c.chunk_index AS chunk_index

        LIMIT 20
        """

        with driver.session(
            database=NEO4J_DATABASE
        ) as session:

            rows = session.run(
                sample_query
            )

            print(
                "\n[DEBUG] Orphan chunks:"
            )

            for row in rows:
                print(
                    f"  chunk_id={row['chunk_id']} "
                    f"doc_id={row['doc_id']} "
                    f"chunk_index={row['chunk_index']}"
                )

    else:

        print(
            "\n[OK] Tất cả Chunk đều có "
            "PART_OF -> Document."
        )

    print("=" * 70)


# ============================================================================
# DEBUG: SHOW GRAPH RELATIONSHIPS
# ============================================================================

def show_relationship_summary(
    driver: Driver
):

    """
    Hiển thị toàn bộ loại relationship hiện tại.
    """

    query = """
    MATCH (a)-[r]->(b)

    RETURN
        labels(a) AS from_labels,
        type(r) AS relationship,
        labels(b) AS to_labels,
        count(*) AS count

    ORDER BY count DESC
    """

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        rows = session.run(
            query
        )

        print("\n" + "=" * 70)
        print("RELATIONSHIP SUMMARY")
        print("=" * 70)

        found = False

        for row in rows:

            found = True

            print(
                f"{row['from_labels']} "
                f"--[{row['relationship']}]--> "
                f"{row['to_labels']} "
                f"count={row['count']}"
            )

        if not found:
            print("[WARN] Không có relationship nào.")

        print("=" * 70)


# ============================================================================
# DEBUG: SHOW SAMPLE CHUNK RELATIONSHIPS
# ============================================================================

def show_sample_chunks(
    driver: Driver,
    limit: int = 20
):

    query = f"""
    MATCH (c:Chunk)
    OPTIONAL MATCH (c)-[r:PART_OF]->(d:Document)

    RETURN
        c.id AS chunk_id,
        c.doc_id AS chunk_doc_id,
        c.chunk_index AS chunk_index,
        c.part_index AS part_index,
        type(r) AS relationship,
        d.id AS document_id

    LIMIT {int(limit)}
    """

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        rows = session.run(
            query
        )

        print("\n" + "=" * 70)
        print("SAMPLE CHUNKS")
        print("=" * 70)

        for row in rows:

            print(
                f"chunk_id={row['chunk_id']} | "
                f"doc_id={row['chunk_doc_id']} | "
                f"chunk_index={row['chunk_index']} | "
                f"part_index={row['part_index']} | "
                f"relationship={row['relationship']} | "
                f"document_id={row['document_id']}"
            )

        print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Nạp dữ liệu pháp luật vào Neo4j KG"
        )
    )

    parser.add_argument(
        "--metadata",
        required=True,
        help="Đường dẫn legal_metadata.parquet"
    )

    parser.add_argument(
        "--relationships",
        required=True,
        help="Đường dẫn legal_relationships.parquet"
    )

    parser.add_argument(
        "--chunks",
        required=False,
        help=(
            "Đường dẫn legal_chunks.parquet "
            "(đã chunk theo Điều/Khoản/Part)"
        )
    )

    parser.add_argument(
        "--content",
        required=False,
        help=(
            "Đường dẫn legal_content.parquet "
            "(fallback: mỗi văn bản = 1 chunk)"
        )
    )

    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Bỏ qua bước tạo constraint/index"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # CHECK CHUNK INPUT
    # ------------------------------------------------------------------------

    if not args.chunks and not args.content:

        print(
            "[WARN] Không truyền --chunks "
            "lẫn --content -> "
            "sẽ KHÔNG nạp Chunk nào cả."
        )

    # Không cho truyền cả 2
    if args.chunks and args.content:

        raise ValueError(
            "Không được truyền đồng thời "
            "--chunks và --content."
        )

    # ------------------------------------------------------------------------
    # CONNECT NEO4J
    # ------------------------------------------------------------------------

    print(
        f"[INFO] Kết nối Neo4j tại "
        f"{NEO4J_URI} ..."
    )

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(
            NEO4J_USER,
            NEO4J_PASSWORD
        )
    )

    driver.verify_connectivity()

    print(
        "[OK] Kết nối thành công."
    )

    try:

        # --------------------------------------------------------------------
        # SCHEMA
        # --------------------------------------------------------------------

        if not args.skip_schema:

            setup_schema(
                driver
            )

        # --------------------------------------------------------------------
        # DOCUMENT
        # --------------------------------------------------------------------

        print(
            "\n[STEP 1] Đọc metadata..."
        )

        metadata_df = pd.read_parquet(
            args.metadata
        )

        print(
            f"[INFO] Metadata rows: "
            f"{len(metadata_df)}"
        )

        ingest_documents(
            driver,
            metadata_df
        )

        # --------------------------------------------------------------------
        # DOCUMENT RELATIONSHIPS
        # --------------------------------------------------------------------

        print(
            "\n[STEP 2] Đọc relationships..."
        )

        rels_df = pd.read_parquet(
            args.relationships
        )

        print(
            f"[INFO] Relationship rows: "
            f"{len(rels_df)}"
        )

        ingest_relationships(
            driver,
            rels_df
        )

        # --------------------------------------------------------------------
        # CHUNKS
        # --------------------------------------------------------------------

        if args.chunks:

            print(
                "\n[STEP 3] Đọc legal_chunks..."
            )

            chunks_df = pd.read_parquet(
                args.chunks
            )

            print(
                f"[INFO] Chunk rows: "
                f"{len(chunks_df)}"
            )

            ingest_chunks(
                driver,
                chunks_df
            )

        # --------------------------------------------------------------------
        # FALLBACK CONTENT
        # --------------------------------------------------------------------

        elif args.content:

            print(
                "\n[STEP 3] Đọc legal_content..."
            )

            content_df = pd.read_parquet(
                args.content
            )

            print(
                f"[INFO] Content rows: "
                f"{len(content_df)}"
            )

            ingest_content_as_chunks(
                driver,
                content_df
            )

        # --------------------------------------------------------------------
        # FINAL DEBUG
        # --------------------------------------------------------------------

        print(
            "\n[STEP 4] Kiểm tra graph..."
        )

        show_sample_chunks(
            driver,
            limit=20
        )

        show_relationship_summary(
            driver
        )

        print(
            "\n✅ Hoàn tất nạp dữ liệu vào Neo4j."
        )

    finally:

        driver.close()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())