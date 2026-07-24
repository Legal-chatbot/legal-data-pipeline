import re
import pandas as pd

ATTACHED_DOC_MARKERS = [
    r'QUY\s*ĐỊNH\s*:', r'QUY\s*CHẾ\s*:', r'NỘI\s*QUY\s*:',
    r'ĐIỀU\s*LỆ\s*:', r'QUY\s*TRÌNH\s*:',
]

BOILERPLATE_CHUNK_PATTERNS = [
    r'^\(?\s*xem\s+file\s+đính\s+kèm\s*\)?$',
    r'^\(?\s*có\s+file\s+đính\s+kèm\s*\)?$',
    r'^\(?\s*có\s+văn\s+bản\s+đính\s+kèm\s*\)?$',
    r'^\(?\s*văn\s+bản\s+đính\s+kèm\s*\)?$',
    r'^\(?\s*văn\s+bản\s+mật\s*\)?$',
]

def clean_noise(text: str) -> str:
    text = re.sub(r'\.{4,}', ' [...] ', text)
    text = re.sub(r'_{4,}', ' [...] ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n[ \t]*\n+', '\n\n', text)
    return text.strip()

def split_into_parts(text: str):
    """Tách văn bản gốc thành các 'phần' độc lập: Quyết định chính +
    (các) văn bản đính kèm (Quy định/Quy chế/Nội quy...).
    Mỗi phần có đánh số Điều riêng -> không được merge/tính chung."""
    pattern = r'(?=' + '|'.join(ATTACHED_DOC_MARKERS) + r')'
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]

def extract_dieu_number(dieu_text: str):
    """Lấy số Điều để lưu metadata, tránh mất thông tin khi merge."""
    m = re.match(r'Điều\s+(\d+[a-zA-Z]?)\b', dieu_text)
    return m.group(1) if m else None

def split_by_dieu(text: str):
    pattern = r'(?=Điều\s+\d+[a-zA-Z]?(?!\.\d)\s*[:.])'
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]

def split_on_word_boundary(text, max_chars=1500, overlap=200):
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            back = text.rfind(' ', start, end)
            if back != -1 and back > start:
                end = back
        chunks.append(text[start:end].strip())
        new_start = end - overlap
        start = new_start if new_start > start else end
    return [c for c in chunks if c]

def sub_split_long_chunk(text, max_chars=1500, overlap=200):
    if len(text) <= max_chars:
        return [text]
    khoan_parts = re.split(r'(?=\n\s*(?:\d+[\.)]|[a-zA-ZđĐ][\.)])\s)', text)
    khoan_parts = [p.strip() for p in khoan_parts if p.strip()]
    result = []
    for part in khoan_parts:
        if len(part) <= max_chars:
            result.append(part)
        else:
            result.extend(split_on_word_boundary(part, max_chars, overlap))
    merged, current = [], ""
    for part in result:
        if len(current) + len(part) <= max_chars:
            current = (current + "\n" + part).strip()
        else:
            if current:
                merged.append(current)
            current = part
    if current:
        merged.append(current)
    return merged

def is_boilerplate_chunk(text: str) -> bool:
    normalized = re.sub(r'\s+', ' ', text).strip().lower()
    if len(normalized) > 40:
        return False
    if re.search(r'\d', normalized):
        return False
    return any(re.match(pattern, normalized, flags=re.IGNORECASE) for pattern in BOILERPLATE_CHUNK_PATTERNS)

def build_chunks_for_part(part_text, max_chars, overlap, min_chars):
    """Chunk trong nội bộ 1 'phần' (part) -- KHÔNG bao giờ merge ra ngoài part."""
    dieu_texts = split_by_dieu(part_text)
    items = []  # list of {"text":..., "articles": [..]}

    for dc in dieu_texts:
        article_no = extract_dieu_number(dc)
        sub_chunks = sub_split_long_chunk(dc, max_chars, overlap)
        for sc in sub_chunks:
            items.append({"text": sc, "articles": [article_no] if article_no else []})

    # merge các item nhỏ liên tiếp TRONG CÙNG PART, cộng dồn article numbers
    merged = []
    for it in items:
        if merged and len(merged[-1]["text"]) < min_chars:
            merged[-1]["text"] = merged[-1]["text"] + "\n" + it["text"]
            merged[-1]["articles"] = merged[-1]["articles"] + it["articles"]
        else:
            merged.append(it)
    return merged

def chunk_document(doc_id, text, max_chars=1500, overlap=200, min_chars=50):
    text = clean_noise(text)
    parts = split_into_parts(text)

    rows = []
    idx = 0
    for part_index, part_text in enumerate(parts):
        chunks = build_chunks_for_part(part_text, max_chars, overlap, min_chars)
        for c in chunks:
            if is_boilerplate_chunk(c["text"]):
                continue
            rows.append({
                "chunk_id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "chunk_index": idx,
                "part_index": part_index,          # 0 = Quyết định chính, 1+ = văn bản đính kèm
                "articles": ",".join(c["articles"]) if c["articles"] else None,
                "text": c["text"],
            })
            idx += 1
    return rows

def chunk_dataframe(df, id_col="id", text_col="text", max_chars=1500, overlap=200):
    rows = []
    for _, row in df.iterrows():
        rows.extend(chunk_document(row[id_col], row[text_col], max_chars, overlap))
    return pd.DataFrame(rows, columns=["chunk_id", "doc_id", "chunk_index", "part_index", "articles", "text"])


df = pd.read_parquet("raw/legal_content.parquet")
chunk_df = chunk_dataframe(df, max_chars=1500, overlap=200)
chunk_df.to_parquet("processed/legal_chunks.parquet", index=False)