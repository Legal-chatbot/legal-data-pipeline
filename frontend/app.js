(() => {
  "use strict";

  const config = window.LEGAL_CHAT_CONFIG || {};
  const apiBaseUrl = (config.apiBaseUrl || "http://127.0.0.1:8000").replace(/\/$/, "");
  const state = { history: loadHistory(), sources: [], busy: false };
  const elements = {
    form: document.querySelector("#chat-form"), input: document.querySelector("#query-input"),
    send: document.querySelector("#send-button"), conversation: document.querySelector("#conversation"),
    welcome: document.querySelector("#welcome-block"), history: document.querySelector("#history-list"),
    historyCount: document.querySelector("#history-count"), sourceCount: document.querySelector("#source-count"),
    sourceEmpty: document.querySelector("#sources-empty"), sourceDetail: document.querySelector("#source-detail"),
    apiStatus: document.querySelector("#api-status"), newChat: document.querySelector("#new-chat"), toast: document.querySelector("#toast")
  };

  init();

  function init() {
    elements.form.addEventListener("submit", submitQuestion);
    elements.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); elements.form.requestSubmit(); }
    });
    elements.input.addEventListener("input", autoGrow);
    elements.newChat.addEventListener("click", newConversation);
    document.querySelectorAll("[data-suggestion]").forEach((button) => button.addEventListener("click", () => {
      elements.input.value = button.dataset.suggestion; autoGrow(); elements.input.focus();
    }));
    renderHistory();
    checkHealth();
  }

  async function submitQuestion(event) {
    event.preventDefault();
    const query = elements.input.value.trim();
    if (!query || state.busy) return;
    state.busy = true; setLoading(true);
    elements.welcome.hidden = true;
    appendMessage("user", query);
    elements.input.value = ""; autoGrow();
    const loading = appendLoading();
    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/chat`, {
        method: "POST", headers: { "Content-Type": "application/json", "X-Request-ID": crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) },
        body: JSON.stringify({ query })
      });
      if (!response.ok) throw new Error(await errorMessage(response));
      const payload = await response.json();
      loading.remove(); appendAnswer(payload);
      state.sources = payload.citations || [];
      updateSources(state.sources);
      saveConversation(query, payload);
      setApiStatus(true);
    } catch (error) {
      loading.remove(); appendMessage("assistant", `Không thể kết nối với dịch vụ pháp lý.\n\n${error.message || "Vui lòng thử lại."}`, true);
      showToast("Không nhận được phản hồi từ backend."); setApiStatus(false);
    } finally { state.busy = false; setLoading(false); }
  }

  function appendMessage(role, text, isError = false) {
    const list = getMessageList(); const message = document.createElement("article");
    message.className = `message ${role}`;
    const avatar = document.createElement("div"); avatar.className = "message-avatar"; avatar.textContent = role === "assistant" ? "L" : "BẠN";
    const body = document.createElement("div"); body.className = "message-body";
    const meta = document.createElement("div"); meta.className = "message-meta"; meta.textContent = role === "assistant" ? "LUẬT · TRỢ LÝ PHÁP LÝ" : "CÂU HỎI CỦA BẠN";
    const content = document.createElement("div"); content.className = `message-text${isError ? " error-message" : ""}`; content.textContent = text;
    body.append(meta, content); message.append(avatar, body); list.append(message); scrollToBottom(); return message;
  }

  function appendAnswer(payload) {
    const list = getMessageList(); const message = document.createElement("article"); message.className = "message assistant";
    const avatar = document.createElement("div"); avatar.className = "message-avatar"; avatar.textContent = "L";
    const body = document.createElement("div"); body.className = "message-body";
    const meta = document.createElement("div"); meta.className = "message-meta"; meta.textContent = "LUẬT · TRỢ LÝ PHÁP LÝ";
    const content = document.createElement("div"); content.className = "message-text"; renderAnswer(content, payload.answer || "", payload.citations || []);
    body.append(meta, content); message.append(avatar, body); list.append(message); scrollToBottom();
  }

  function renderAnswer(container, answer, citations) {
    const parts = answer.split(/(\[C\d+\])/g);
    parts.forEach((part) => {
      const match = part.match(/^\[(C\d+)\]$/);
      if (!match) { container.append(document.createTextNode(part)); return; }
      const citation = citations.find((item) => item.citation_id === match[1]);
      const button = document.createElement("button"); button.className = "citation-button"; button.type = "button"; button.textContent = part;
      button.title = citation ? citation.label : "Citation không hợp lệ";
      button.addEventListener("click", () => showSource(citation)); container.append(button);
    });
  }

  function getMessageList() {
    let list = elements.conversation.querySelector(".message-list");
    if (!list) { list = document.createElement("div"); list.className = "message-list"; elements.conversation.append(list); }
    return list;
  }

  function appendLoading() {
    const message = document.createElement("article"); message.className = "message assistant";
    message.innerHTML = '<div class="message-avatar">L</div><div class="message-body"><div class="message-meta">ĐANG TRA CỨU</div><div class="message-text loading-dots"><i></i><i></i><i></i></div></div>';
    getMessageList().append(message); scrollToBottom(); return message;
  }

  function updateSources(citations) {
    elements.sourceCount.textContent = citations.length; elements.sourceEmpty.hidden = citations.length > 0; elements.sourceDetail.hidden = citations.length === 0;
    elements.sourceDetail.innerHTML = ""; citations.forEach((citation, index) => {
      const card = document.createElement("article"); card.className = "source-card"; card.dataset.citationId = citation.citation_id;
      const doc = citation.source_document || {}; const chunk = citation.source_chunk || {};
      card.innerHTML = `<div class="source-index">[${escapeHtml(citation.citation_id || `C${index + 1}`)}]</div><div class="source-title">${escapeHtml(doc.title || citation.label || "Văn bản pháp luật")}</div><div class="source-number">${escapeHtml(doc.document_number || "Chưa có số ký hiệu")}</div><div class="source-meta">${location(citation)} · ${escapeHtml(doc.validity_status || "Chưa rõ hiệu lực")}</div>`;
      card.addEventListener("click", () => showSource(citation)); elements.sourceDetail.append(card);
    });
  }

  function showSource(citation) {
    if (!citation) { showToast("Citation không tồn tại trong context."); return; }
    const card = elements.sourceDetail.querySelector(`[data-citation-id="${citation.citation_id}"]`);
    if (card) { card.scrollIntoView({ behavior: "smooth", block: "center" }); card.classList.add("selected"); setTimeout(() => card.classList.remove("selected"), 900); }
    let context = elements.sourceDetail.querySelector(".source-context"); if (context) context.remove();
    if (card && citation.source_chunk && citation.source_chunk.text) { context = document.createElement("div"); context.className = "source-context"; context.textContent = citation.source_chunk.text; card.append(context); }
  }

  function location(citation) { return [citation.article && `Điều ${citation.article}`, citation.clause && `Khoản ${citation.clause}`, citation.point && `Điểm ${citation.point}`].filter(Boolean).join(" · ") || "Vị trí chưa rõ"; }
  function newConversation() { const list = elements.conversation.querySelector(".message-list"); if (list) list.remove(); elements.welcome.hidden = false; updateSources([]); elements.input.value = ""; elements.input.focus(); }
  function setLoading(loading) { elements.send.disabled = loading; elements.send.querySelector("span").textContent = loading ? "Đang xử lý" : "Gửi câu hỏi"; }
  function autoGrow() { elements.input.style.height = "auto"; elements.input.style.height = `${Math.min(elements.input.scrollHeight, 130)}px`; }
  function scrollToBottom() { requestAnimationFrame(() => elements.conversation.scrollTo({ top: elements.conversation.scrollHeight, behavior: "smooth" })); }
  function setApiStatus(online) { elements.apiStatus.className = `topbar-status ${online ? "online" : "offline"}`; elements.apiStatus.lastElementChild.textContent = online ? "Backend đang hoạt động" : "Backend không khả dụng"; }
  async function checkHealth() { try { const response = await fetch(`${apiBaseUrl}/health`); setApiStatus(response.ok); } catch { setApiStatus(false); } }
  async function errorMessage(response) { try { const data = await response.json(); return data.message || "Backend trả về lỗi."; } catch { return `Backend trả về lỗi (${response.status}).`; } }
  function showToast(text) { elements.toast.textContent = text; elements.toast.classList.add("visible"); setTimeout(() => elements.toast.classList.remove("visible"), 3200); }
  function saveConversation(query, payload) { state.history = [{ query, answer: payload.answer, citations: payload.citations || [] }, ...state.history.filter((item) => item.query !== query)].slice(0, 12); localStorage.setItem("legal-chat-history", JSON.stringify(state.history)); renderHistory(); }
  function loadHistory() { try { return JSON.parse(localStorage.getItem("legal-chat-history") || "[]"); } catch { return []; } }
  function renderHistory() { elements.history.innerHTML = ""; elements.historyCount.textContent = state.history.length; if (!state.history.length) { elements.history.innerHTML = '<div class="history-empty">Chưa có câu hỏi nào</div>'; return; } state.history.forEach((item) => { const button = document.createElement("button"); button.className = "history-item"; button.type = "button"; button.textContent = item.query; button.addEventListener("click", () => { elements.input.value = item.query; autoGrow(); elements.input.focus(); }); elements.history.append(button); }); }
  function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char])); }
})();