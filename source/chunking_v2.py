import re
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ATTACHED_DOC_MARKERS = [
    r'QUY\s*ĐỊNH\s*:',
    r'QUY\s*CHẾ\s*:',
    r'NỘI\s*QUY\s*:',
    r'ĐIỀU\s*LỆ\s*:',
    r'QUY\s*TRÌNH\s*:',
]

BOILERPLATE_CHUNK_PATTERNS = [
    r'^\(?\s*xem\s+file\s+đính\s+kèm\s*\)?$',
    r'^\(?\s*có\s+file\s+đính\s+kèm\s*\)?$',
    r'^\(?\s*có\s+văn\s+bản\s+đính\s+kèm\s*\)?$',
    r'^\(?\s*văn\s+bản\s+đính\s+kèm\s*\)?$',
    r'^\(?\s*văn\s+bản\s+mật\s*\)?$',
]


# ============================================================
# 1. CLEAN TEXT
# ============================================================

def clean_noise(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    text = re.sub(r'\.{4,}', ' [...] ', text)
    text = re.sub(r'_{4,}', ' [...] ', text)

    text = text.replace('\r\n', '\n').replace('\r', '\n')

    text = re.sub(r'[ \t]{2,}', ' ', text)

    text = re.sub(r'\n[ \t]*\n+', '\n\n', text)

    return text.strip()


# ============================================================
# 2. SPLIT DOCUMENT -> PART
# ============================================================

def split_into_parts(text: str):
    """
    Tách văn bản thành các part độc lập.

    Ví dụ:

        Quyết định chính
        QUY ĐỊNH:
        Điều 1...
        Điều 2...

    sẽ thành:

        part 0 = Quyết định chính
        part 1 = QUY ĐỊNH + Điều...
    """

    if not text:
        return []

    pattern = r'(?=' + '|'.join(ATTACHED_DOC_MARKERS) + r')'

    parts = re.split(pattern, text)

    return [
        p.strip()
        for p in parts
        if p and p.strip()
    ]


# ============================================================
# 3. ARTICLE / ĐIỀU
# ============================================================

ARTICLE_PATTERN = (
    r'(?='
    r'(?:^|\n)'
    r'\s*Điều\s+\d+[a-zA-Z]?'
    r'(?:\s*[:.])?'
    r')'
)


def extract_dieu_number(dieu_text: str):
    if not dieu_text:
        return None

    m = re.search(
        r'^\s*Điều\s+(\d+[a-zA-Z]?)\b',
        dieu_text,
        flags=re.IGNORECASE
    )

    return m.group(1) if m else None


def split_by_dieu(text: str):
    """
    Tách theo Điều.

    Quan trọng:
    Không dùng pattern quá cứng kiểu:
        Điều 1:
        Điều 2.

    mà cho phép:
        Điều 1
        Điều 1.
        Điều 1:
        Điều 1a.
        Điều 1a:
    """

    if not text:
        return []

    parts = re.split(ARTICLE_PATTERN, text, flags=re.IGNORECASE)

    return [
        p.strip()
        for p in parts
        if p and p.strip()
    ]


# ============================================================
# 4. SPLIT KHOẢN / ĐIỂM
# ============================================================

KHOAN_PATTERN = (
    r'(?='
    r'\n\s*'
    r'(?:'
    r'\d+[\.)]'
    r'|[a-zA-ZđĐ][\.)]'
    r')'
    r'\s+'
    r')'
)


def split_by_khoan(text: str):
    if not text:
        return []

    parts = re.split(
        KHOAN_PATTERN,
        text
    )

    return [
        p.strip()
        for p in parts
        if p and p.strip()
    ]


# ============================================================
# 5. WORD BOUNDARY FALLBACK
# ============================================================

def split_on_word_boundary(
    text: str,
    max_chars=1500,
    overlap=200
):
    """
    HARD LIMIT:

    Mỗi chunk cố gắng <= max_chars.

    Chỉ có trường hợp một token đơn lẻ dài hơn
    max_chars thì mới có thể vượt giới hạn.
    """

    if not text:
        return []

    if len(text) <= max_chars:
        return [text.strip()]

    chunks = []

    start = 0
    n = len(text)

    while start < n:

        end = min(
            start + max_chars,
            n
        )

        if end < n:

            newline_pos = text.rfind(
                '\n',
                start,
                end
            )

            if (
                newline_pos != -1
                and newline_pos > start
            ):
                end = newline_pos

            else:

                space_pos = text.rfind(
                    ' ',
                    start,
                    end
                )

                if (
                    space_pos != -1
                    and space_pos > start
                ):
                    end = space_pos

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end <= start:
            end = min(
                start + max_chars,
                n
            )

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ============================================================
# 6. ROBUST LONG TEXT SPLITTER
# ============================================================

def split_long_text(
    text,
    max_chars=1500,
    overlap=200
):
    """
    Split nhiều tầng:

    1. đoạn
    2. khoản
    3. word boundary

    Mục tiêu cuối:
        KHÔNG có chunk > max_chars
    """

    if not text:
        return []

    text = text.strip()

    if len(text) <= max_chars:
        return [text]

    # --------------------------------------------------------
    # Tầng 1: paragraph
    # --------------------------------------------------------

    paragraphs = re.split(
        r'\n\s*\n+',
        text
    )

    paragraphs = [
        p.strip()
        for p in paragraphs
        if p and p.strip()
    ]

    # --------------------------------------------------------
    # Tầng 2: khoản
    # --------------------------------------------------------

    units = []

    for paragraph in paragraphs:

        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue

        khoans = split_by_khoan(
            paragraph
        )

        if len(khoans) <= 1:
            units.append(paragraph)
        else:
            units.extend(khoans)

    # --------------------------------------------------------
    # Tầng 3: hard split
    # --------------------------------------------------------

    result = []

    for unit in units:

        if len(unit) <= max_chars:
            result.append(unit)
        else:
            result.extend(
                split_on_word_boundary(
                    unit,
                    max_chars=max_chars,
                    overlap=overlap
                )
            )

    return [
        x.strip()
        for x in result
        if x and x.strip()
    ]


# ============================================================
# 7. MERGE SMALL CHUNKS
# ============================================================

def merge_small_chunks(
    items,
    max_chars=1500,
    min_chars=50
):
    """
    Merge chunk nhỏ nhưng:

        KHÔNG BAO GIỜ > max_chars
    """

    merged = []

    for item in items:

        text = item["text"]

        if not text:
            continue

        if not merged:

            merged.append({
                "text": text,
                "articles": list(
                    item.get("articles", [])
                )
            })

            continue

        previous = merged[-1]

        candidate = (
            previous["text"]
            + "\n"
            + text
        ).strip()

        # chỉ merge nếu vẫn <= max_chars
        if (
            len(previous["text"]) < min_chars
            and len(candidate) <= max_chars
        ):

            previous["text"] = candidate

            previous["articles"].extend(
                item.get("articles", [])
            )

        else:

            merged.append({
                "text": text,
                "articles": list(
                    item.get("articles", [])
                )
            })

    return merged


# ============================================================
# 8. BUILD CHUNKS FOR PART
# ============================================================

def build_chunks_for_part(
    part_text,
    max_chars=1500,
    overlap=200,
    min_chars=50
):
    """
    Chunk trong một PART.

    Có Điều:
        Điều -> khoản -> word boundary

    Không có Điều:
        paragraph -> khoản -> word boundary
    """

    if not part_text:
        return []

    # --------------------------------------------------------
    # thử tách Điều
    # --------------------------------------------------------

    dieu_texts = split_by_dieu(
        part_text
    )

    items = []

    # --------------------------------------------------------
    # Có nhiều Điều
    # --------------------------------------------------------

    if len(dieu_texts) > 1:

        for dieu_text in dieu_texts:

            article_no = extract_dieu_number(
                dieu_text
            )

            sub_chunks = split_long_text(
                dieu_text,
                max_chars=max_chars,
                overlap=overlap
            )

            for sc in sub_chunks:

                items.append({
                    "text": sc,
                    "articles": (
                        [article_no]
                        if article_no
                        else []
                    )
                })

    # --------------------------------------------------------
    # Không nhận diện được Điều
    # --------------------------------------------------------

    else:

        sub_chunks = split_long_text(
            part_text,
            max_chars=max_chars,
            overlap=overlap
        )

        for sc in sub_chunks:

            items.append({
                "text": sc,
                "articles": []
            })

    # --------------------------------------------------------
    # merge nhỏ
    # --------------------------------------------------------

    return merge_small_chunks(
        items,
        max_chars=max_chars,
        min_chars=min_chars
    )


# ============================================================
# 9. BOILERPLATE
# ============================================================

def is_boilerplate_chunk(text: str) -> bool:

    if not text:
        return True

    normalized = re.sub(
        r'\s+',
        ' ',
        text
    ).strip().lower()

    if len(normalized) > 40:
        return False

    if re.search(r'\d', normalized):
        return False

    return any(
        re.match(
            pattern,
            normalized,
            flags=re.IGNORECASE
        )
        for pattern
        in BOILERPLATE_CHUNK_PATTERNS
    )


# ============================================================
# 10. FINAL SAFETY SPLIT
# ============================================================

def enforce_max_chars(
    rows,
    max_chars=1500,
    overlap=200
):
    """
    FINAL SAFETY NET.

    Nếu bất kỳ chunk nào > max_chars,
    bắt buộc split lại.

    Đây là lớp bảo vệ để không bao giờ
    tái diễn chunk 654,113 chars.
    """

    final_rows = []

    for row in rows:

        text = row["text"]

        if len(text) <= max_chars:

            final_rows.append(row)

            continue

        # split lại
        pieces = split_on_word_boundary(
            text,
            max_chars=max_chars,
            overlap=overlap
        )

        for piece in pieces:

            new_row = row.copy()
            new_row["text"] = piece

            final_rows.append(
                new_row
            )

    # đánh lại chunk_index
    for idx, row in enumerate(final_rows):

        row["chunk_index"] = idx
        row["chunk_id"] = (
            f'{row["doc_id"]}_{idx}'
        )

    return final_rows


# ============================================================
# 11. CHUNK DOCUMENT
# ============================================================

def chunk_document(
    doc_id,
    text,
    max_chars=1500,
    overlap=200,
    min_chars=50
):

    text = clean_noise(text)

    if not text:
        return []

    parts = split_into_parts(
        text
    )

    rows = []

    for part_index, part_text in enumerate(parts):

        chunks = build_chunks_for_part(
            part_text,
            max_chars=max_chars,
            overlap=overlap,
            min_chars=min_chars
        )

        for c in chunks:

            chunk_text = c["text"].strip()

            if not chunk_text:
                continue

            if is_boilerplate_chunk(
                chunk_text
            ):
                continue

            rows.append({

                "chunk_id": None,

                "doc_id": doc_id,

                "chunk_index": None,

                "part_index": part_index,

                "articles": (
                    ",".join(
                        c["articles"]
                    )
                    if c["articles"]
                    else None
                ),

                "text": chunk_text,
            })

    # --------------------------------------------------------
    # FINAL HARD SAFETY
    # --------------------------------------------------------

    rows = enforce_max_chars(
        rows,
        max_chars=max_chars,
        overlap=overlap
    )

    return rows


# ============================================================
# 12. DATAFRAME
# ============================================================

def chunk_dataframe(
    df,
    id_col="id",
    text_col="text",
    max_chars=1500,
    overlap=200
):

    rows = []

    total = len(df)

    for i, (_, row) in enumerate(
        df.iterrows()
    ):

        doc_id = row[id_col]
        text = row[text_col]

        doc_rows = chunk_document(
            doc_id,
            text,
            max_chars=max_chars,
            overlap=overlap
        )

        rows.extend(doc_rows)

        # progress
        if (
            i % 1000 == 0
            or i == total - 1
        ):
            print(
                f"\rProcessed "
                f"{i + 1:,}/{total:,} documents",
                end=""
            )

    print()

    return pd.DataFrame(
        rows,
        columns=[
            "chunk_id",
            "doc_id",
            "chunk_index",
            "part_index",
            "articles",
            "text"
        ]
    )


# ============================================================
# 13. MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("LOAD DOCUMENTS")
    print("=" * 70)

    df = pd.read_parquet(
        "raw/legal_content.parquet"
    )

    print(
        f"Documents: {len(df):,}"
    )

    print("=" * 70)
    print("CHUNKING")
    print("=" * 70)

    chunk_df = chunk_dataframe(
        df,
        max_chars=1500,
        overlap=200
    )

    print()
    print("=" * 70)
    print("VERIFY CHUNKS")
    print("=" * 70)

    print(
        f"Total chunks: "
        f"{len(chunk_df):,}"
    )

    if len(chunk_df) > 0:

        lengths = chunk_df["text"].str.len()

        print(
            f"Min chars: "
            f"{lengths.min():,}"
        )

        print(
            f"Max chars: "
            f"{lengths.max():,}"
        )

        print(
            f"Avg chars: "
            f"{lengths.mean():.2f}"
        )

        print(
            f"> max_chars: "
            f"{(lengths > 1500).sum():,}"
        )

        print(
            f"> 10000 chars: "
            f"{(lengths > 10000).sum():,}"
        )

    # --------------------------------------------------------
    # hard assertion
    # --------------------------------------------------------

    oversized = chunk_df[
        chunk_df["text"].str.len() > 1500
    ]

    if len(oversized) > 0:

        print()
        print(
            "[ERROR] Có chunk > 1500 chars!"
        )

        print(
            oversized[
                [
                    "chunk_id",
                    "doc_id",
                    "chunk_index",
                    "part_index"
                ]
            ].head(20)
        )

        raise RuntimeError(
            "Chunking failed: "
            "found chunks > max_chars"
        )

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAVE")
    print("=" * 70)

    chunk_df.to_parquet(
        "processed/legal_chunks_v2.parquet",
        index=False
    )

    print(
        "Saved: "
        "processed/legal_chunks_v2.parquet"
    )