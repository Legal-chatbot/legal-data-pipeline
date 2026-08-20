import sys
from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def search_by_keyword(keyword: str, top_k: int = 5):
    query = """
    CALL db.index.fulltext.queryNodes('chunk_fulltext', $keyword)
    YIELD node, score
    MATCH (node)-[:PART_OF]->(d:Document)
    RETURN d.id AS doc_id, d.title AS title, d.tinh_trang_hieu_luc AS tinh_trang,
           node.part_index AS part_index, node.chunk_index AS chunk_index,
           node.articles AS articles, node.text AS text, score
    ORDER BY score DESC
    LIMIT $top_k
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        return [r.data() for r in session.run(query, keyword=keyword, top_k=top_k)]


# 2. Tìm chunk liên quan theo vector similarity — dùng khi ĐÃ sinh embedding
#    (xem ghi chú trong neo4j_ingest.py::setup_vector_index)
def search_by_vector(query_embedding: list, top_k: int = 5):
    query = """
    CALL db.index.vector.queryNodes('chunk_embedding_idx', $top_k, $embedding)
    YIELD node, score
    MATCH (node)-[:PART_OF]->(d:Document)
    RETURN d.id AS doc_id, d.title AS title,
           node.part_index AS part_index, node.chunk_index AS chunk_index,
           node.articles AS articles, node.text AS text, score
    ORDER BY score DESC
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        return [r.data() for r in session.run(query, embedding=query_embedding, top_k=top_k)]


# 3. Truy vết quan hệ pháp lý của 1 văn bản — trả lời câu hỏi kiểu
#    "văn bản này còn hiệu lực không, bị sửa đổi/thay thế bởi văn bản nào?"
def get_document_legal_status(doc_id: int):
    query = """
    MATCH (d:Document {id: $doc_id})
    OPTIONAL MATCH (d)-[r]->(other:Document)
    RETURN d.id AS id, d.title AS title, d.tinh_trang_hieu_luc AS tinh_trang,
           collect({rel_type: type(r), other_id: other.id, other_title: other.title}) AS quan_he
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, doc_id=doc_id).single()
        return result.data() if result else None


# 4. Truy vết ngược: những văn bản nào tham chiếu / sửa đổi / hết hiệu lực BỞI văn bản này
def get_incoming_relations(doc_id: int):
    query = """
    MATCH (other:Document)-[r]->(d:Document {id: $doc_id})
    RETURN type(r) AS rel_type, other.id AS other_id, other.title AS other_title
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        return [r.data() for r in session.run(query, doc_id=doc_id)]


# 5. Đường đi giữa 2 văn bản (vd: A sửa đổi B, B bị thay thế bởi C -> hỏi quan hệ A-C)
def find_path_between_documents(doc_id_a: int, doc_id_b: int, max_hops: int = 4):
    query = f"""
    MATCH p = shortestPath((a:Document {{id: $a}})-[*..{max_hops}]-(b:Document {{id: $b}}))
    RETURN [n IN nodes(p) | n.title] AS titles,
           [r IN relationships(p) | type(r)] AS rel_types
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, a=doc_id_a, b=doc_id_b).single()
        return result.data() if result else None


# 6. Hybrid retrieval cho RAG: kết hợp fulltext search + mở rộng ngữ cảnh
#    bằng graph (lấy thêm văn bản liên quan trực tiếp) — đây là điểm mạnh
#    của KG so với vector-only RAG: trả lời được câu hỏi cần suy luận qua
#    nhiều văn bản (multi-hop), ví dụ "văn bản đang hết hiệu lực nào từng
#    hướng dẫn luật X?"
def hybrid_retrieve(keyword: str, top_k: int = 5, expand_hops: int = 1):
    query = f"""
    CALL db.index.fulltext.queryNodes('chunk_fulltext', $keyword)
    YIELD node, score
    WITH node, score LIMIT $top_k
    MATCH (node)-[:PART_OF]->(d:Document)
    OPTIONAL MATCH (d)-[r*1..{expand_hops}]-(related:Document)
    RETURN d.id AS doc_id, d.title AS title,
           node.part_index AS part_index, node.chunk_index AS chunk_index,
           node.articles AS articles, node.text AS chunk_text, score,
           collect(DISTINCT related.title)[..5] AS related_docs
    ORDER BY score DESC
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        return [r.data() for r in session.run(query, keyword=keyword, top_k=top_k)]


# 7. MỚI: Lấy đúng (các) chunk chứa 1 Điều cụ thể của 1 văn bản.
#    Dùng khi user hỏi thẳng "Điều 12 Nghị định X quy định gì" -> khỏi cần
#    fulltext search mò, tra thẳng theo articles (đã index ở setup_schema).
#    LƯU Ý: nếu văn bản có phần đính kèm (Quy định/Quy chế...), truyền
#    part_index để tránh lấy nhầm "Điều 12" của phần khác.
def search_by_article(doc_id: int, article_no: str, part_index: int = None):
    query = """
    MATCH (c:Chunk)-[:PART_OF]->(d:Document {id: $doc_id})
    WHERE $article_no IN c.articles
      AND ($part_index IS NULL OR c.part_index = $part_index)
    RETURN d.title AS title, c.part_index AS part_index,
           c.chunk_index AS chunk_index, c.articles AS articles, c.text AS text
    ORDER BY c.part_index, c.chunk_index
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        return [r.data() for r in session.run(
            query, doc_id=doc_id, article_no=str(article_no), part_index=part_index
        )]


# 8. MỚI: Liệt kê các "phần" (part) của 1 văn bản — hữu ích để biết văn bản
#    có kèm Quy định/Quy chế/Nội quy... hay không, và mỗi phần có bao nhiêu
#    chunk / Điều, trước khi drill-down bằng search_by_article.
def list_document_parts(doc_id: int):
    query = """
    MATCH (c:Chunk)-[:PART_OF]->(d:Document {id: $doc_id})
    RETURN c.part_index AS part_index,
           count(c) AS num_chunks,
           apoc.coll.toSet(apoc.coll.flatten(collect(c.articles))) AS articles
    ORDER BY part_index
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        try:
            return [r.data() for r in session.run(query, doc_id=doc_id)]
        except Exception:
            # Fallback nếu APOC chưa cài: gom thủ công phía Python
            fallback_query = """
            MATCH (c:Chunk)-[:PART_OF]->(d:Document {id: $doc_id})
            RETURN c.part_index AS part_index, c.articles AS articles
            """
            rows = [r.data() for r in session.run(fallback_query, doc_id=doc_id)]
            grouped = {}
            for row in rows:
                pi = row["part_index"]
                grouped.setdefault(pi, {"part_index": pi, "num_chunks": 0, "articles": set()})
                grouped[pi]["num_chunks"] += 1
                grouped[pi]["articles"].update(row["articles"] or [])
            result = list(grouped.values())
            for r in result:
                r["articles"] = sorted(r["articles"])
            return sorted(result, key=lambda r: r["part_index"])


if __name__ == "__main__":
    kw = sys.argv[1] if len(sys.argv) > 1 else "hiệu lực"
    for row in hybrid_retrieve(kw):
        print(row)
    driver.close()
