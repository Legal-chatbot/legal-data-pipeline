# BÁO CÁO: QUY TRÌNH CHUNKING DỮ LIỆU VĂN BẢN PHÁP LUẬT

## 1. Bối cảnh và mục tiêu

**Đầu vào:** `legal_content.parquet` — mỗi dòng là một văn bản pháp luật hoàn chỉnh (Sắc lệnh, Quyết định, Chỉ thị...) với cấu trúc gồm phần mở đầu (căn cứ pháp lý), phần nội dung chia theo "Điều", và có thể có phụ lục/biểu mẫu/văn bản đính kèm.

**Đầu ra mong muốn:** `chunk_id, doc_id, chunk_index, text` — các đoạn nhỏ giữ nguyên ranh giới ngữ nghĩa pháp lý, sẵn sàng đưa vào embedding cho hệ thống retrieval (RAG).

**Mục tiêu chất lượng của chunking:**
- Mỗi chunk phải là một đơn vị ngữ nghĩa hoàn chỉnh (không cắt giữa câu, giữa từ, giữa Điều/Khoản có liên quan).
- Không sinh ra chunk "rác" (không mang thông tin pháp lý).
- Không trộn lẫn nội dung của hai văn bản/văn bản-con khác nhau vào một chunk.
- Giữ được khả năng truy vết chunk về đúng Điều/Khoản nguồn.

Qua quá trình kiểm thử trên dữ liệu thực tế, đã phát hiện **3 nhóm vấn đề chính**, được trình bày và xử lý tuần tự dưới đây.

---

## 2. Các vấn đề phát hiện được

### Vấn đề 1 — Cắt giữa từ khi fallback theo số ký tự

**Hiện tượng:** Khi một "Điều" quá dài và không có cấu trúc Khoản rõ ràng, thuật toán fallback về cắt cứng theo `max_chars` ký tự. Việc cắt theo chỉ số ký tự tuyệt đối không quan tâm ranh giới từ, dẫn đến:

```
...trách nhiệm in ấn, cấp phát, quản lý...
```
bị tách thành hai chunk:
```
chunk N:   "...trách nhi"
chunk N+1: "ệm in ấn, cấp phát, quản lý..."
```

**Nguyên nhân gốc:** `text[start:end]` cắt theo offset ký tự cố định, không kiểm tra ký tự tại vị trí cắt có phải khoảng trắng hay không.

**Tác động:** Từ bị vỡ đôi làm hỏng token hóa của embedding model, giảm chất lượng vector, có thể khiến cả hai chunk liên quan đều bị đánh giá thấp về độ liên quan khi truy vấn.

---

### Vấn đề 2 — Chunk "rác" từ phần biểu mẫu/khoảng trống điền tay

**Hiện tượng:** Một số văn bản (ví dụ Quyết định ban hành biểu mẫu biên bản) chứa các đoạn form trống với dấu chấm lặp để điền tay:

```
Họ và tên:......................................................
```

Vì các đoạn này không có "Điều" hoặc "Khoản" nào để làm điểm cắt, toàn bộ đoạn form rơi vào fallback cắt cứng theo ký tự, sinh ra các chunk mà phần lớn nội dung chỉ là dấu `.` hoặc `_` lặp lại, gần như không mang thông tin ngữ nghĩa.

**Tác động:** 
- Lãng phí ngân sách embedding (token hóa cho chuỗi dấu chấm không có giá trị).
- Làm loãng không gian vector — các chunk rác này có thể vô tình được retrieval trả về do độ dài/mật độ ký tự đặc biệt, gây nhiễu kết quả tìm kiếm.

---

### Vấn đề 3 — Gộp chunk ngắn xuyên ranh giới văn bản độc lập (nghiêm trọng nhất)

**Hiện tượng:** Nhiều văn bản pháp luật có cấu trúc **kép**: một Quyết định chính (thường chỉ có 2-3 Điều ngắn: "Điều 1: Ban hành kèm theo...", "Điều 2: Có hiệu lực...", "Điều 3: Giao ai thi hành...") **kèm theo** một văn bản phụ lục độc lập (Quy định / Quy chế / Nội quy / Điều lệ) có **hệ thống đánh số Điều riêng, bắt đầu lại từ Điều 1**.

Ví dụ thực tế (doc `4205`):

```
[QUYẾT ĐỊNH]
  Điều 1: Ban hành kèm theo Quyết định này quy định về chức năng...
  Điều 2: Quyết định này có hiệu lực kể từ ngày ký.
  Điều 3: Thủ trưởng các cơ quan... căn cứ quyết định thi hành.
QUY ĐỊNH:  (văn bản đính kèm — SỐ ĐIỀU RESET)
  Điều 1. Chức năng: Sở Văn hoá - Thông tin là...
  Điều 2. Nhiệm vụ quyền hạn: ...
  Điều 3. Tổ chức bộ máy: ...
```

Vì các Điều 1/2/3 của phần "QUYẾT ĐỊNH" đều ngắn (dưới ngưỡng `min_chars`), thuật toán gộp chunk ngắn với chunk liền kề đã vô tình gộp **Điều 3 (của Quyết định)** với **tiêu đề "QUY ĐỊNH:" mở đầu văn bản đính kèm** vào chung một chunk.

**Tác động — đây là lỗi nghiêm trọng nhất trong 3 lỗi:**
- Hai đơn vị pháp lý độc lập, có đánh số Điều **trùng nhau** (đều có "Điều 1", "Điều 2", "Điều 3" nhưng nội dung hoàn toàn khác) bị nhập nhằng trong cùng một chunk hoặc bị mất metadata phân biệt.
- Nếu người dùng hỏi "Điều 1 quy định nội dung gì", hệ thống không thể phân biệt được đây là Điều 1 của Quyết định (nói về việc ban hành) hay Điều 1 của Quy định đính kèm (nói về chức năng của Sở) — dẫn đến trích dẫn sai văn bản nguồn.
- Về bản chất, đây là lỗi **ranh giới tài liệu** (document boundary), nghiêm trọng hơn lỗi ranh giới câu/từ vì nó làm sai lệch cả nguồn trích dẫn, không chỉ chất lượng câu chữ.

---

## 3. Giải pháp chi tiết

### 3.1. Giải pháp cho Vấn đề 1: Cắt theo ranh giới từ

Thay cắt cứng `text[start:start+max_chars]` bằng việc lùi điểm cắt về khoảng trắng gần nhất trước ngưỡng:

```python
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
        start = new_start if new_start > start else end  # tránh lặp vô hạn
    return [c for c in chunks if c]
```

Đây chỉ nên là **lớp fallback cuối cùng**, ưu tiên tách theo Điều → Khoản → (nếu là biểu mẫu) theo "Mẫu số" trước khi rơi vào cắt cứng.

### 3.2. Giải pháp cho Vấn đề 2: Làm sạch nhiễu trước khi chunk

Nén các chuỗi dấu chấm/gạch dưới lặp lại (chỗ trống điền tay) thành một placeholder ngắn, thực hiện **trước** khi tách Điều/Khoản:

```python
def clean_noise(text: str) -> str:
    text = re.sub(r'\.{4,}', ' [...] ', text)
    text = re.sub(r'_{4,}', ' [...] ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()
```

Đồng thời bổ sung tầng tách theo **"Mẫu số"** cho các đoạn biểu mẫu quá dài không có Khoản, để mỗi biểu mẫu (Mẫu số 1, 2, 3...) là một chunk độc lập có ngữ nghĩa, thay vì bị cắt cứng giữa hai mẫu không liên quan.

> **Khuyến nghị bổ sung:** Với các biểu mẫu hoàn toàn trống (chỉ có nhãn trường như "Họ và tên:", "Địa chỉ:" không có nội dung), nên đánh dấu `is_template=True` hoặc loại khỏi tập embedding, vì các đoạn này không mang giá trị tra cứu ngữ nghĩa mà chỉ có giá trị "biết văn bản này có ban hành mẫu X" — có thể giữ 1 chunk tóm tắt duy nhất "văn bản này ban hành các mẫu 1,2,3,4" thay vì embedding toàn bộ mẫu trống.

### 3.3. Giải pháp cho Vấn đề 3: Tách theo ranh giới văn bản trước khi tách Điều

Đây là thay đổi kiến trúc quan trọng nhất: thêm **một tầng tách ở mức cao hơn Điều**, nhận diện các điểm bắt đầu của văn bản đính kèm (thường là các dòng tiêu đề độc lập: "QUY ĐỊNH:", "QUY CHẾ:", "NỘI QUY:", "ĐIỀU LỆ:", "QUY TRÌNH:"):

```python
ATTACHED_DOC_MARKERS = [
    r'QUY\s*ĐỊNH\s*:', r'QUY\s*CHẾ\s*:', r'NỘI\s*QUY\s*:',
    r'ĐIỀU\s*LỆ\s*:', r'QUY\s*TRÌNH\s*:',
]

def split_into_parts(text: str):
    pattern = r'(?=' + '|'.join(ATTACHED_DOC_MARKERS) + r')'
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]
```

**Nguyên tắc cứng:** việc gộp chunk ngắn (do dưới `min_chars`) chỉ được thực hiện **trong nội bộ một `part`**, không bao giờ được gộp xuyên qua ranh giới `part`. Điều 3 cuối của Quyết định, dù ngắn, sẽ **không còn bị gộp** với tiêu đề mở đầu của văn bản đính kèm.

Đồng thời bổ sung metadata `part_index` và `articles` (số Điều gốc) để mỗi chunk luôn truy vết được:
- Nó thuộc văn bản chính hay văn bản đính kèm nào (`part_index`).
- Nó tương ứng với Điều số mấy trong văn bản đó (`articles`), dù đã bị gộp với Điều liền kề do quá ngắn.

```python
def build_chunks_for_part(part_text, max_chars, overlap, min_chars):
    dieu_texts = split_by_dieu(part_text)
    items = []
    for dc in dieu_texts:
        article_no = extract_dieu_number(dc)
        for sc in sub_split_long_chunk(dc, max_chars, overlap):
            items.append({"text": sc, "articles": [article_no] if article_no else []})

    merged = []
    for it in items:
        if merged and len(merged[-1]["text"]) < min_chars:
            merged[-1]["text"] += "\n" + it["text"]
            merged[-1]["articles"] += it["articles"]
        else:
            merged.append(it)
    return merged   # KHÔNG merge ra ngoài phạm vi part_text
```

---

## 4. Kiến trúc pipeline hoàn chỉnh (4 tầng tách, theo thứ tự ưu tiên)

```
Văn bản gốc (1 dòng trong parquet)
   │
   ├─ 0. clean_noise()              → nén chuỗi dấu chấm/gạch lặp
   │
   ├─ 1. split_into_parts()         → tách Quyết định chính / văn bản đính kèm
   │        (ranh giới: "QUY ĐỊNH:", "QUY CHẾ:", "NỘI QUY:"...)
   │
   ├─ 2. split_by_dieu()            → trong mỗi part, tách theo "Điều X"
   │
   ├─ 3. sub_split_long_chunk()     → nếu 1 Điều quá dài:
   │        3a. thử tách theo "Khoản" (1. 2. 3...)
   │        3b. nếu vẫn dài & là biểu mẫu: tách theo "Mẫu số"
   │        3c. fallback: split_on_word_boundary() (cắt theo ranh giới từ)
   │
   └─ 4. merge chunk ngắn           → CHỈ gộp trong cùng 1 part,
            giữ lại "articles" metadata khi gộp
```

## 5. Bảng đối chiếu trước/sau

| # | Vấn đề | Trước | Sau |
|---|---|---|---|
| 1 | Cắt giữa từ | `"...trách nhi" / "ệm in ấn..."` | Luôn lùi về khoảng trắng gần nhất trước khi cắt |
| 2 | Chunk toàn dấu `.....` | Chunk rác, không có giá trị embedding | Nén thành `[...]`, tách riêng theo "Mẫu số" nếu đủ dài |
| 3 | Gộp xuyên ranh giới văn bản | Điều 3 (Quyết định) dính với tiêu đề "QUY ĐỊNH:" (văn bản đính kèm, Điều 1 khác) | Tách `part_index` trước, không merge xuyên part; giữ `articles` để truy vết |

## 6. Schema đầu ra đề xuất

| Cột | Ý nghĩa |
|---|---|
| `chunk_id` | `{doc_id}_{index}` |
| `doc_id` | ID văn bản gốc |
| `chunk_index` | Thứ tự chunk trong văn bản |
| `part_index` | 0 = văn bản/quyết định chính; 1, 2... = các văn bản đính kèm theo thứ tự xuất hiện |
| `articles` | Danh sách số Điều gốc chứa trong chunk (vd. `"1,2"`), phục vụ trích dẫn chính xác |
| `text` | Nội dung chunk đã làm sạch |


