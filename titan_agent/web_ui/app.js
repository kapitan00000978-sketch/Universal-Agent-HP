// Extra LLM X Web UI Client Logic

const messagesStream = document.getElementById("messages-stream");
const chatForm = document.getElementById("chat-form");
const userPromptInput = document.getElementById("user-prompt");
const submitBtn = document.getElementById("submit-btn");
const welcomeCard = document.getElementById("welcome-card");
const statusBadge = document.getElementById("agent-status-badge");
const currentProviderSpan = document.getElementById("current-provider");
const currentModelSpan = document.getElementById("current-model");

// Settings modal elements
const settingsModal = document.getElementById("settings-modal");
const openSettingsBtn = document.getElementById("open-settings-btn");
const closeSettingsBtn = document.getElementById("close-settings-btn");
const saveSettingsBtn = document.getElementById("save-settings-btn");
const providerSelect = document.getElementById("provider-select");
const modelInput = document.getElementById("model-input");
const apiKeyInput = document.getElementById("api-key-input");
const baseUrlInput = document.getElementById("base-url-input");

let isStreaming = false;
let currentMode = "fast"; // "fast" | "deep" | "deep_search"
let currentEffort = "medium"; // "low" | "medium" | "high" | "ultra"

// Token throughput guardrail: the browser (Puter) path can never exceed
// 214,000 tokens/second. Token bucket, same contract as the backend limiter.
const TOKEN_RATE_LIMIT = 214000;
const tokenBucket = { tokens: TOKEN_RATE_LIMIT, last: performance.now() };
async function acquireTokens(n) {
  if (n <= 0) return 0;
  const now = performance.now();
  tokenBucket.tokens = Math.min(
    TOKEN_RATE_LIMIT,
    tokenBucket.tokens + ((now - tokenBucket.last) / 1000) * TOKEN_RATE_LIMIT
  );
  tokenBucket.last = now;
  if (n <= tokenBucket.tokens) {
    tokenBucket.tokens -= n;
    return 0;
  }
  const waitMs = ((n - tokenBucket.tokens) / TOKEN_RATE_LIMIT) * 1000;
  tokenBucket.tokens = 0;
  tokenBucket.last = now + waitMs;
  if (waitMs > 0) await new Promise((r) => setTimeout(r, waitMs));
  return waitMs;
}

// ---------- Phase 12: server auth token ----------
// Every /api route requires 'Authorization: Bearer <TITAN_API_KEY>'. The token
// lives in localStorage; on first load we prompt once and cache it. The public
// routes / and /health need no token.
const API_KEY_STORAGE = "titan_api_key";

function getApiKey() {
  try { return localStorage.getItem(API_KEY_STORAGE) || ""; } catch (e) { return ""; }
}

function setApiKey(token) {
  try { localStorage.setItem(API_KEY_STORAGE, token); } catch (e) { /* ignore */ }
}

// Wraps fetch() with the Bearer header + a 401 handler that re-prompts for the key.
async function apiFetch(url, options = {}) {
  const token = getApiKey();
  const headers = Object.assign({}, options.headers || {});
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(url, Object.assign({}, options, { headers }));
  if (res.status === 401) {
    promptForApiKey();
    throw new Error("Unauthorized — API key required.");
  }
  return res;
}

function promptForApiKey() {
  const current = getApiKey();
  const input = window.prompt(
    "EXTRA LLM X — server authentication is on.\n" +
    "Enter the API key (it was printed once in the server console, or set in .env as TITAN_API_KEY / EXTRA_LLM_X_API_KEY).\n" +
    "Leave empty to try the public liveness check only.",
    current
  );
  if (input !== null) {
    const trimmed = input.trim();
    if (trimmed) setApiKey(trimmed);
  }
}

// ---------- Phase 19: live HITL approvals panel ----------
// Polls pending approval requests and lets an operator approve/deny inline.
// A failed poll pauses for 60s instead of re-prompting for the API key on
// every tick (a bad key would otherwise spam dialogs every 5 seconds).
let _hitlNextPoll = 0;

async function fetchHITLPending(force) {
  const now = Date.now();
  if (!force && now < _hitlNextPoll) return;
  const listEl = document.getElementById("hitl-requests");
  const countEl = document.getElementById("hitl-count");
  try {
    const token = getApiKey();
    const headers = {};
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const res = await apiFetch("/api/hitl/pending", { headers });
    if (!res.ok) {
      _hitlNextPoll = now + 60000;
      if (res.status === 401) promptForApiKey();
      throw new Error("HTTP " + res.status);
    }
    _hitlNextPoll = 0;
    const data = await res.json();
    const pending = (data && data.pending) || [];
    if (countEl) countEl.textContent = String(pending.length);
    if (!listEl) return;
    if (!pending.length) {
      listEl.innerHTML = '<div class="empty-hint">No pending approvals</div>';
      return;
    }
    listEl.innerHTML = pending.map((req) => {
      const id = escapeHtml(req.request_id);
      const action = escapeHtml(req.action);
      const resource = escapeHtml(req.resource || "*");
      const reason = escapeHtml(req.reason);
      const details = req.details && typeof req.details === "object"
        ? escapeHtml(JSON.stringify(req.details)).slice(0, 160)
        : "";
      return (
        '<div class="approval-item" data-id="' + id + '">' +
          '<div class="approval-head">' +
            '<span class="approval-action">' + action + '</span>' +
            '<span class="approval-status">pending</span>' +
          '</div>' +
          '<div class="approval-resource" title="' + resource + '">' + resource + '</div>' +
          (reason ? '<div class="approval-reason">' + reason + '</div>' : "") +
          (details ? '<div class="approval-details">' + details + '</div>' : "") +
          '<div class="approval-actions">' +
            '<button class="tiny-btn approve-btn" onclick="decideHITL(' +
            "'" + id + "','approve'" + ')">✓ Approve</button>' +
            '<button class="tiny-btn deny-btn" onclick="decideHITL(' +
            "'" + id + "','deny'" + ')">✕ Deny</button>' +
          '</div>' +
        '</div>'
      );
    }).join("");
  } catch (e) {
    if (listEl) listEl.innerHTML =
      '<div class="empty-hint">Approvals unavailable: ' +
      escapeHtml(String((e && e.message) || e)) + '</div>';
  }
}

async function decideHITL(requestId, decision) {
  try {
    const token = getApiKey();
    const headers = { "Content-Type": "application/json" };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const res = await apiFetch("/api/hitl/decide", {
      method: "POST",
      headers,
      body: JSON.stringify({ decision, request_id: requestId, by: "web-ui" }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      alert("HITL " + decision + " failed: " +
        ((detail && detail.detail) || res.status));
    }
  } catch (e) {
    alert("HITL " + decision + " failed: " + ((e && e.message) || e));
  }
  fetchHITLPending(true);
}

// ---------- Subagent Staff Roster ----------
async function fetchStaffRoles(force) {
  const listEl = document.getElementById("subagents-list");
  const countEl = document.getElementById("staff-count");
  try {
    const res = await apiFetch("/api/staff/roles");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    const roles = (data && data.roles) || [];
    if (countEl) countEl.textContent = String(roles.length);
    if (!listEl) return;
    if (!roles.length) {
      listEl.innerHTML = '<div class="empty-hint">No subagent roles found</div>';
      return;
    }
    listEl.innerHTML = roles.map((r) => {
      const id = escapeHtml(r.id);
      const title = escapeHtml(r.title);
      const desc = escapeHtml(r.description);
      return (
        '<div class="subagent-card" onclick="selectStaffRole(' + "'" + id + "'" + ')" title="' + desc + '">' +
          '<div class="subagent-card-info">' +
            '<span class="subagent-card-title">' + title + '</span>' +
            '<span class="subagent-card-desc">' + desc + '</span>' +
          '</div>' +
          '<span class="subagent-role-pill">' + id + '</span>' +
        '</div>'
      );
    }).join("");
  } catch (e) {
    if (listEl) listEl.innerHTML =
      '<div class="empty-hint">Roster unavailable</div>';
  }
}

function selectStaffRole(roleId) {
  const promptInput = document.getElementById("user-prompt");
  if (!promptInput) return;
  const current = promptInput.value;
  const prefix = `[Delegate to ${roleId}]: `;
  if (!current.startsWith("[Delegate to")) {
    promptInput.value = prefix + current;
  }
  promptInput.focus();
}

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  // Phase 12: if no API key is cached, prompt for it on first load so the very
  // first config fetch is authenticated. A missing key only shows the prompt
  // once; /health stays reachable without it.
  if (!getApiKey()) {
    promptForApiKey();
  }
  fetchConfig();
  fetchMcpTools();
  fetchWorkspaceFiles();
  fetchStaffRoles();
  setupEventListeners();
  fetchHITLPending();                    // Phase 19: live approvals panel
  setInterval(fetchHITLPending, 5000);   // (poll pauses for 60s after failures)
});

// ---------- Model name helpers ----------
// The underlying variant ID (e.g. "...:free") is what the API needs for free usage,
// but the UI should never reveal it. We strip it for display and re-append on send.
function displayModel(id) {
  return id ? id.replace(/:free$/, "") : id;
}
function resolvePuterModel(raw) {
  const v = (raw || "").trim();
  if (!v) return "deepseek/deepseek-v4-pro:free";
  if (/:(free|flex|priority)$/.test(v)) return v;
  return v + ":free";
}

function setupEventListeners() {
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const prompt = userPromptInput.value.trim();
    if (!prompt || isStreaming) return;
    if (handleLocalSlashCommand(prompt)) {
      userPromptInput.value = "";
      return;
    }
    sendMessage(prompt);
  });

  userPromptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Mode selector (Fast / Deep Thinking / Deep Search)
  document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      currentMode = btn.dataset.mode;
      document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("mode-active"));
      btn.classList.add("mode-active");
    });
  });

  // Effort selector (Low / Medium / High / Ultra)
  document.querySelectorAll(".effort-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      currentEffort = btn.dataset.effort;
      document.querySelectorAll(".effort-btn").forEach(b => b.classList.remove("effort-active"));
      btn.classList.add("effort-active");
    });
  });

  // Settings
  openSettingsBtn.addEventListener("click", () => {
    settingsModal.classList.add("open");
    scanLocalModels();
    // When Puter provider is selected, auto-load models
    if (providerSelect.value === "puter") {
      loadPuterModels(false);
    }
  });
  closeSettingsBtn.addEventListener("click", () => settingsModal.classList.remove("open"));
  saveSettingsBtn.addEventListener("click", saveConfig);
  document.getElementById("scan-local-btn").addEventListener("click", scanLocalModels);

  // Quick Model Chips (Puter default; data-provider overrides, e.g. OmniRoute)
  document.querySelectorAll(".model-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      providerSelect.value = btn.dataset.provider || "puter";
      modelInput.value = displayModel(btn.dataset.model);
      apiKeyInput.value = "";
    });
  });

  // Free Provider Hub buttons
  const freePortalInfo = document.getElementById("free-portal-info");
  const freePortalMsg = document.getElementById("free-portal-msg");
  const freePortalLink = document.getElementById("free-portal-link");

  document.querySelectorAll(".free-prov-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const prov = btn.dataset.prov;
      const model = btn.dataset.model;
      const url = btn.dataset.url;
      const portal = btn.dataset.portal;

      providerSelect.value = prov;
      modelInput.value = model;
      baseUrlInput.value = url || "";
      apiKeyInput.focus();

      if (freePortalInfo && freePortalMsg && freePortalLink) {
        freePortalInfo.style.display = "block";
        freePortalMsg.textContent = `${btn.textContent.trim()} tanlandi! Bepul kalit olish:`;
        freePortalLink.href = portal;
        freePortalLink.textContent = `🔑 Portalga o'tish (Bepul) ↗`;
      }
    });
  });

  // When provider changes: open the model browser automatically for Puter
  providerSelect.addEventListener("change", () => {
    if (providerSelect.value === "puter") {
      loadPuterModels(false);
    }
  });

  // Puter Model Browser — load all models via listModels()
  const loadPuterBtn = document.getElementById("load-puter-models-btn");
  const modelSearch = document.getElementById("puter-model-search");
  if (modelSearch) {
    modelSearch.addEventListener("input", () => renderPuterModels());
  }
  document.querySelectorAll(".puter-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".puter-tab").forEach(t => t.classList.remove("puter-tab-active"));
      tab.classList.add("puter-tab-active");
      renderPuterModels();
    });
  });
  if (loadPuterBtn) {
    loadPuterBtn.addEventListener("click", () => loadPuterModels(true));
  }
}

// ---- Puter model browser ----
let allPuterModels = [];
let puterActiveTab = "free"; // "free" | "all"

function normalizePuterModel(m) {
  // listModels() returns an array of objects; some SDK versions may return strings
  if (typeof m === "string") return { id: m, provider: "", name: m };
  return {
    id: (m && m.id) || "",
    provider: (m && m.provider) || "",
    name: (m && m.name) || "",
    context: (m && m.context) || 0,
    max_tokens: (m && m.max_tokens) || 0
  };
}

async function loadPuterModels(force) {
  const loadPuterBtn = document.getElementById("load-puter-models-btn");
  const browserEl = document.getElementById("puter-model-browser");
  const listEl = document.getElementById("puter-model-list");
  if (!loadPuterBtn || !browserEl || !listEl) return;

  if (loadPuterBtn.disabled) return; // load already in progress
  loadPuterBtn.disabled = true;
  loadPuterBtn.textContent = "Loading...";
  listEl.innerHTML = '<div style="padding:12px; font-size:12px; color:var(--text-muted);">Fetching models...</div>';
  browserEl.style.display = "block";

  try {
    if (typeof puter === "undefined" || !puter.ai || !puter.ai.listModels) {
      throw new Error("Puter.js is not loaded. Check your internet connection or reopen the page.");
    }
    const models = await puter.ai.listModels();
    allPuterModels = (Array.isArray(models) ? models : []).map(normalizePuterModel).filter(m => m.id);
    if (allPuterModels.length === 0) {
      listEl.innerHTML = '<div style="padding:12px; font-size:12px; color:var(--text-muted);">No models found.</div>';
      return;
    }
    renderPuterModels();
  } catch (err) {
    listEl.innerHTML = `<div style="padding:12px; font-size:12px; color:var(--accent-rose);">Error: ${escapeHtml(err.message)}</div>`;
  } finally {
    loadPuterBtn.disabled = false;
    loadPuterBtn.textContent = "🌐 All Models";
  }
}

function renderPuterModels() {
  const listEl = document.getElementById("puter-model-list");
  const searchEl = document.getElementById("puter-model-search");
  if (!listEl) return;

  const q = (searchEl ? searchEl.value : "").toLowerCase().trim();
  const activeTab = [...document.querySelectorAll(".puter-tab")].find(t => t.classList.contains("puter-tab-active"));
  puterActiveTab = activeTab ? activeTab.dataset.tab : "free";

  let filtered = allPuterModels;
  if (puterActiveTab === "free") {
    filtered = filtered.filter(m => m.id.endsWith(":free"));
  }
  if (q) {
    filtered = filtered.filter(m =>
      m.id.toLowerCase().includes(q) ||
      (m.provider || "").toLowerCase().includes(q) ||
      (m.name || "").toLowerCase().includes(q)
    );
  }

  if (filtered.length === 0) {
    listEl.innerHTML = '<div style="padding:12px; font-size:12px; color:var(--text-muted);">No models match your search. Try another query or switch tabs.</div>';
    return;
  }

  // Group by provider
  const groups = {};
  for (const m of filtered) {
    const prov = m.provider || "others";
    (groups[prov] = groups[prov] || []).push(m);
  }

  let html = `<div style="padding:6px 10px; font-size:11px; color:var(--text-muted); border-bottom:1px solid var(--border-color);">Total: ${filtered.length} models</div>`;

  for (const [prov, models] of Object.entries(groups)) {
    html += `<div style="padding:8px 10px 2px; font-size:11px; font-weight:700; letter-spacing:0.6px; text-transform:uppercase; color:var(--accent-cyan);">${escapeHtml(prov)} (${models.length})</div>`;
    for (const m of models) {
      const displayId = displayModel(m.id);
      const ctx = m.context ? ` • ${Math.round(m.context / 1000)}k context` : "";
      html += `
        <button type="button" class="puter-model-chip" data-model="${escapeHtml(m.id)}" title="${escapeHtml(displayId)}${ctx}" style="display:flex; align-items:center; gap:8px; width:100%; text-align:left; padding:7px 10px; background:rgba(255,255,255,0.03); border:1px solid transparent; border-radius:6px; cursor:pointer; font-size:12px; color:var(--text-primary);">
          <span style="color:var(--text-muted);">🤖</span>
          <span style="font-family:var(--font-mono); flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(displayId)}</span>
          <span style="font-size:10px; color:var(--text-muted); white-space:nowrap;">${ctx}</span>
        </button>`;
    }
  }
  listEl.innerHTML = html;

  // Selecting a chip applies the full variant internally but shows the clean name
  listEl.querySelectorAll(".puter-model-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      providerSelect.value = "puter";
      modelInput.value = displayModel(chip.dataset.model);
      apiKeyInput.value = "";
      modelInput.style.borderColor = "var(--accent-emerald)";
    });
  });
}

async function scanLocalModels() {
  const chipsContainer = document.getElementById("local-models-chips");
  chipsContainer.innerHTML = '<span style="font-size:11px; color:var(--accent-cyan);">Scanning...</span>';
  try {
    const res = await apiFetch("/api/local-models");
    const data = await res.json();
    chipsContainer.innerHTML = "";

    const ollamaModels = data.ollama || [];
    const lmModels = data.lmstudio || [];

    if (ollamaModels.length === 0 && lmModels.length === 0) {
      chipsContainer.innerHTML = '<span style="font-size:11px; color:var(--text-muted);">(No local models found yet)</span>';
      return;
    }

    ollamaModels.forEach(m => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "tiny-btn";
      chip.style.borderColor = "var(--accent-cyan)";
      chip.style.color = "var(--accent-cyan)";
      chip.textContent = `🦙 Ollama: ${m}`;
      chip.onclick = () => {
        providerSelect.value = "ollama";
        modelInput.value = m;
        apiKeyInput.value = "";
      };
      chipsContainer.appendChild(chip);
    });

    lmModels.forEach(m => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "tiny-btn";
      chip.style.borderColor = "var(--accent-violet)";
      chip.style.color = "var(--accent-violet)";
      chip.textContent = `⚡ LM: ${m}`;
      chip.onclick = () => {
        providerSelect.value = "lmstudio";
        modelInput.value = m;
      };
      chipsContainer.appendChild(chip);
    });
  } catch (err) {
    chipsContainer.innerHTML = '<span style="font-size:11px; color:var(--text-muted);">(Could not check)</span>';
  }
}

function sendQuickPrompt(text) {
  userPromptInput.value = text;
  sendMessage(text);
}

async function fetchConfig() {
  try {
    const res = await apiFetch("/api/config");
    const data = await res.json();
    currentProviderSpan.textContent = data.provider.toUpperCase();
    currentModelSpan.textContent = displayModel(data.model);
    providerSelect.value = data.provider;
    modelInput.value = displayModel(data.model);
    if (data.base_url) baseUrlInput.value = data.base_url;
  } catch (err) {
    console.error("Config fetch error:", err);
  }
}

async function saveConfig() {
  const payload = {
    provider: providerSelect.value,
    model: providerSelect.value === "puter" ? resolvePuterModel(modelInput.value) : modelInput.value.trim(),
    api_key: apiKeyInput.value.trim(),
    base_url: baseUrlInput.value.trim()
  };
  try {
    const res = await apiFetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      settingsModal.classList.remove("open");
      fetchConfig();
    }
  } catch (err) {
    alert("Error saving settings: " + err);
  }
}

async function fetchMcpTools() {
  try {
    const res = await apiFetch("/api/mcp/tools");
    if (!res.ok) throw new Error(`MCP status request failed (${res.status})`);
    const data = await res.json();
    const servers = data.servers || {};
    const serverKeys = Object.keys(servers);
    const connected = serverKeys.filter((key) => servers[key].connected).length;
    const configured = data.configured_count ?? serverKeys.length;
    const mcpCountElem = document.getElementById("mcp-count");
    if (mcpCountElem) {
      const toolCount = serverKeys.reduce((sum, k) => sum + (servers[k].tools_count || 0), 0);
      mcpCountElem.textContent = `${connected}/${configured} MCP servers connected (${toolCount} tools)`;
    }
  } catch (e) {
    console.warn("MCP tools fetch failed", e);
  }
}

async function fetchWorkspaceFiles() {
  try {
    const res = await apiFetch("/api/workspace/files");
    const data = await res.json();
    const tree = document.getElementById("files-tree");
    if (!data.files || data.files.length === 0) {
      tree.innerHTML = '<div class="empty-hint">(Empty)</div>';
      return;
    }
    tree.innerHTML = data.files.map(f => `
      <div class="file-row">
        <span>📄</span>
        <span class="file-name" title="${f.rel_path}">${f.name}</span>
        <span style="margin-left:auto; color:var(--text-muted); font-size:10px;">${f.size} B</span>
      </div>
    `).join("");
  } catch (e) {
    console.warn("Files fetch error", e);
  }
}

function appendUserMessage(text) {
  if (welcomeCard) welcomeCard.style.display = "none";
  const row = document.createElement("div");
  row.className = "message-row user";
  row.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
  messagesStream.appendChild(row);
  scrollToBottom();
}

function createAgentCard() {
  const row = document.createElement("div");
  row.className = "message-row agent";

  const card = document.createElement("div");
  card.className = "agent-response-card";

  const statusLine = document.createElement("div");
  statusLine.className = "status-line";
  statusLine.style.fontSize = "12px";
  statusLine.style.color = "var(--accent-cyan)";
  statusLine.innerHTML = "⚡ Extra LLM X is working...";
  card.appendChild(statusLine);

  row.appendChild(card);
  messagesStream.appendChild(row);
  scrollToBottom();

  return { row, card, statusLine };
}

// ---- Slash commands (mirrors titan_agent/commands.py) ----
const SLASH_COMMANDS = {
  plan: { mode: "deep", effort: "high", template: "Create a detailed, step-by-step implementation plan for the following task. Break it into phases/milestones with clear done-criteria, list the files/tools you would touch, the risks, and end with the single next action to take now.\n\nTASK: {arg}" },
  review: { mode: "deep", effort: "high", template: "Perform a rigorous code review of the following target. Check for bugs, edge cases, error handling, naming, security issues and test coverage. Report findings by severity (Critical / Important / Minor / Nit) with concrete fixes.\n\nREVIEW TARGET: {arg}" },
  "security-scan": { mode: "deep", effort: "ultra", template: "Run a security audit on the following target using the security-ops skill. Threat-model first: check for injection, secrets, path traversal, SSRF, unsafe deserialization, and dependency vulnerabilities. Report by severity with fixes.\n\nSCAN TARGET: {arg}" },
  research: { mode: "deep_search", effort: "high", template: "Research the following topic thoroughly. Find multiple sources, cross-check claims, prefer recently updated primary sources, and cite everything you actually retrieved.\n\nTOPIC: {arg}" },
  explain: { mode: "deep", effort: "medium", template: "Explain the following in depth: what it is, how it works, why it matters, and any caveats. Use concrete examples.\n\nSUBJECT: {arg}" },
  fix: { mode: "deep", effort: "high", template: "Diagnose and fix the following issue. Reproduce or understand the cause, make the smallest correct change, then verify it actually works before reporting.\n\nISSUE: {arg}" },
  test: { mode: "deep", effort: "medium", template: "Write and/or run tests for the following target. Cover the happy path, edge cases, and error paths. Report test results.\n\nTEST TARGET: {arg}" },
  remember: { mode: "fast", effort: "auto", template: "Save the following to memory using memory_save with a clear short key, then confirm it was saved.\n\nFACT: {arg}" },
  handoff: { mode: "fast", effort: "auto", template: "Leave a handoff note using handoff_create summarizing current state, decisions, and next steps. Keep it concise and actionable.\n\nNOTE: {arg}" }
};

function expandSlashCommand(text) {
  if (!text || text[0] !== "/" || text.length < 2) return null;
  const stripped = text.slice(1).trim();
  if (!stripped) return null;
  const idx = stripped.indexOf(" ");
  const name = (idx === -1 ? stripped : stripped.slice(0, idx)).toLowerCase();
  const arg = (idx === -1 ? "" : stripped.slice(idx + 1)).trim();
  const cmd = SLASH_COMMANDS[name];
  if (!cmd) return null;
  return {
    prompt: cmd.template.replace("{arg}", arg || "(no argument provided — ask about the general case)"),
    mode: cmd.mode,
    effort: cmd.effort
  };
}

// Local (no-LLM) slash commands rendered right in the chat area.
async function handleLocalSlashCommand(text) {
  if (!text || text[0] !== "/") return false;
  const stripped = text.slice(1).trim();
  if (!stripped) return false;
  const idx = stripped.indexOf(" ");
  const name = (idx === -1 ? stripped : stripped.slice(0, idx)).toLowerCase();
  const arg = (idx === -1 ? "" : stripped.slice(idx + 1)).trim();

  const appendRaw = (title, bodyHtml) => {
    const { card, statusLine } = createAgentCard();
    card.querySelector(".agent-message-title").textContent = title;
    const div = document.createElement("div");
    div.className = "answer-content";
    div.innerHTML = bodyHtml;
    card.insertBefore(div, statusLine);
    statusLine.style.display = "none";
    scrollToBottom();
  };

  if (name === "help") {
    appendUserMessage(text);
    const lines = [
      "<b>Slash commands:</b>",
      ...Object.keys(SLASH_COMMANDS).map(k => `<code>/${k}</code> — ${k}`),
      "<b>Local commands:</b>",
      "<code>/plan TASK</code>, <code>/review FILE</code>, <code>/security-scan FILE</code> — expert modes",
      "<code>/research TOPIC</code>, <code>/fix ISSUE</code>, <code>/test FILE</code>, <code>/explain SUBJECT</code>",
      "<code>/remember FACT</code>, <code>/handoff NOTE</code>",
      "<code>/status</code> — current provider / model / mode / effort",
      "<code>/skills</code> — list skill playbooks",
      "<code>/handoffs</code> — list open handoff notes",
      "<code>/memory QUERY</code> — search long-term memory"
    ];
    appendRaw("Help", lines.join("<br>"));
    return true;
  }
  if (name === "status") {
    appendUserMessage(text);
    let cfgText = "Loading config...";
    try {
      const res = await apiFetch("/api/config");
      const cfg = await res.json();
      cfgText = `<b>Provider:</b> ${cfg.provider} · <b>Model:</b> ${cfg.model}<br>` +
        `<b>Mode:</b> ${currentMode} · <b>Effort:</b> ${currentEffort} · <b>Workspace:</b> <code>${cfg.workspace}</code>`;
    } catch (e) { cfgText = "Could not load config."; }
    appendRaw("Status", cfgText);
    return true;
  }
  if (name === "skills") {
    appendUserMessage(text);
    let body = "No skills loaded.";
    try {
      const res = await apiFetch("/api/tools/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tool_name: "skills_list", arguments: {} })
      });
      const data = await res.json();
      body = (data.result || "No skills loaded.").replace(/\n/g, "<br>");
    } catch (e) { body = "Could not list skills."; }
    appendRaw("Skill Playbooks", body);
    return true;
  }
  if (name === "handoffs") {
    appendUserMessage(text);
    let body = "No open handoffs.";
    try {
      const res = await apiFetch("/api/memory/handoffs?status=open");
      const data = await res.json();
      const list = data.handoffs || [];
      if (list.length) {
        body = list.map(h => `[${h.id}] <b>${escapeHtml(h.title)}</b> — ${escapeHtml((h.content || "").slice(0, 160))}`).join("<br>");
      }
    } catch (e) { body = "Could not list handoffs."; }
    appendRaw("Open Handoffs", body);
    return true;
  }
  if (name === "memory") {
    appendUserMessage(text);
    if (!arg) { appendRaw("Memory", "Usage: <code>/memory &lt;query&gt;</code>"); return true; }
    let body = "Nothing found.";
    try {
      const res = await apiFetch("/api/memory?query=" + encodeURIComponent(arg));
      const data = await res.json();
      const list = data.knowledge || [];
      if (list.length) {
        body = list.map(f => `[${f.category}] <b>${escapeHtml(f.key)}</b>: ${escapeHtml(f.value)}`).join("<br>");
      }
    } catch (e) { body = "Could not search memory."; }
    appendRaw(`Memory: ${arg}`, body);
    return true;
  }
  return false;
}

async function sendMessage(prompt) {
  if (isStreaming) return;
  isStreaming = true;
  userPromptInput.value = "";
  submitBtn.disabled = true;
  statusBadge.textContent = "Working...";

  // Expand slash commands (/plan, /review, ...) into a full prompt + mode + effort.
  const slash = expandSlashCommand(prompt);
  if (slash) {
    prompt = slash.prompt;
    currentMode = slash.mode;
    currentEffort = slash.effort;
    document.querySelectorAll(".mode-btn").forEach(b => {
      b.classList.toggle("mode-active", b.dataset.mode === currentMode);
    });
    document.querySelectorAll(".effort-btn").forEach(b => {
      b.classList.toggle("effort-active", b.dataset.effort === currentEffort);
    });
  }

  appendUserMessage(prompt);
  const { card, statusLine } = createAgentCard();

  let currentThoughtAccordion = null;
  let finalAnswerDiv = null;

  // If provider is Puter.js, run direct client-side Puter AI
  if (providerSelect.value === "puter") {
    await sendPuterMessage(prompt, card, statusLine);
    isStreaming = false;
    submitBtn.disabled = false;
    statusBadge.textContent = "System Ready";
    statusLine.style.display = "none";
    fetchWorkspaceFiles();
    scrollToBottom();
    return;
  }

  try {
    const response = await apiFetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: prompt, session_id: "web_session", mode: currentMode, effort: currentEffort })
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Chat request failed (${response.status})${detail ? `: ${detail.slice(0, 300)}` : ""}`);
    }
    if (!response.body) throw new Error("Chat response did not include a stream.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // keep partial

      for (const block of lines) {
        if (!block.startsWith("data: ")) continue;
        const jsonStr = block.replace("data: ", "").trim();
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr);
          handleAgentEvent(event, card, statusLine, {
            getThought: () => currentThoughtAccordion,
            setThought: (el) => { currentThoughtAccordion = el; },
            getAnswer: () => finalAnswerDiv,
            setAnswer: (el) => { finalAnswerDiv = el; }
          });
        } catch (err) {
          console.error("Event parse error:", err, jsonStr);
        }
      }
      scrollToBottom();
    }
  } catch (err) {
    statusLine.innerHTML = `<span style="color:var(--accent-rose)">An error occurred: ${err.message}</span>`;
  } finally {
    isStreaming = false;
    submitBtn.disabled = false;
    statusBadge.textContent = "System Ready";
    statusLine.style.display = "none";
    fetchWorkspaceFiles();
    scrollToBottom();
  }
}

async function sendPuterMessage(prompt, card, statusLine) {
  let modelName = displayModel(resolvePuterModel(modelInput.value.trim() || "deepseek/deepseek-v4-pro"));
  statusLine.innerHTML = `⚡ Connecting directly to ${escapeHtml(modelName)}...`;
  try {
    if (typeof puter === "undefined" || !puter.ai) {
      throw new Error("Puter.js is not loaded. Check your internet connection.");
    }

    let answerDiv = document.createElement("div");
    answerDiv.className = "answer-content";
    card.insertBefore(answerDiv, statusLine);

    let modeNote = "";
    if (currentMode === "deep") {
      modeNote = "\n\n### DEEP THINKING MODE (engaged):\n" +
        "- Decompose the problem into explicit sub-problems and reason about each one in detail.\n" +
        "- Consider alternative approaches, edge cases, and failure modes before committing.\n" +
        "- After every step, ask yourself: is there anything unverified, ambiguous, or missing?\n" +
        "- Do not settle for a shallow answer: dig until the result is provably correct and complete.";
    } else if (currentMode === "deep_search") {
      modeNote = "\n\n### DEEP SEARCH MODE (engaged):\n" +
        "- Start with the deep_search tool to build a comprehensive multi-source dossier on the topic.\n" +
        "- Cross-check claims across multiple sources; prefer verifiable, recently updated information.\n" +
        "- Scrape primary pages when a snippet is insufficient (scrape_webpage tool).\n" +
        "- Structure the final answer with sections and cite the sources you actually retrieved.\n" +
        "- If evidence is thin or conflicting, say so explicitly instead of guessing.";
    }

    let effortNote = "";
    if (currentEffort === "low") {
      effortNote = "\n\n### EFFORT LEVEL: LOW\n" +
        "- Prioritize SPEED: use the smallest number of tool calls that completes the task.\n" +
        "- Answer directly and concisely; do not expand scope beyond the request.";
    } else if (currentEffort === "high") {
      effortNote = "\n\n### EFFORT LEVEL: HIGH\n" +
        "- Work like a careful expert: decompose the problem and reason about each part in detail.\n" +
        "- After every step ask yourself: is anything unverified, ambiguous, or missing?\n" +
        "- Use your larger iteration budget deliberately for verification, never for decoration.";
    } else if (currentEffort === "ultra") {
      effortNote = "\n\n### EFFORT LEVEL: ULTRA\n" +
        "- Be exhaustive: cover edge cases, failure modes, and alternative approaches.\n" +
        "- Verify every claim with tools; do not settle for a shallow answer.\n" +
        "- Review the whole request from the user's perspective — keep working until every part is met.";
    }

    const messages = [
      {
        role: "system",
        content: "You are EXTRA LLM X — an ultra-powerful autonomous AI reasoning and execution engine running in a web dashboard.\n\n" +
          "### PLAN-ACT-VERIFY-REPORT + REFLECT:\n" +
          "1. PLAN: briefly outline your strategy inside <thought>...</thought> before using tools.\n" +
          "2. ACT: use tools via <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>. Batch independent calls and run them together.\n" +
          "3. VERIFY: if a tool errors, read the message, fix arguments, retry with an alternative approach — never give up after one failure.\n" +
          "4. REFLECT: after tools run, critically review your own work — did you satisfy the whole request? fix gaps before answering.\n" +
          "5. REPORT: finish with a well-structured markdown final answer in the user's language.\n\n" +
          "### TOOL CATALOG (execute via /api/tools/execute):\n" +
          "- execute_command(command, cwd?) — run PowerShell commands on the host OS\n" +
          "- read_file(path), write_file(path, content), edit_file(path, target_text, replacement_text), list_directory(path?) — filesystem\n" +
          "- workspace_rag(query, top_k?) — find the most relevant snippets across ALL workspace files (with file paths)\n" +
          "- web_search(query, max_results?) — live DuckDuckGo search\n" +
          "- scrape_webpage(url) — fetch readable text from a URL\n" +
          "- python_eval(code) — run Python in a subprocess\n" +
          "- deep_search(topic) — multi-hop web research dossier\n" +
          "- deep_coder(task_name, files, test_code?) — full software engineering cycle with test verification\n" +
          "- launch_application(app_or_command) — open a Windows app\n" +
          "- system_info() — live OS / CPU / RAM / disk / Python facts\n" +
          "- manage_processes(action: list|kill, pattern?) — list or kill OS processes\n" +
          "- memory_save(key, value, category?) / memory_search(query) — persistent long-term memory\n" +
          "- mcp_* — tools from connected MCP servers\n\n" +
          "### EFFICIENCY: never re-run a tool for already-known output; if the goal is reached, stop and answer immediately; don't add decorative steps.\n" +
          "### PARALLEL: when several independent tool calls are needed, batch them in one turn so they execute simultaneously." +
          modeNote + "\n\n" +
          effortNote + "\n\n" +
          "### LANGUAGE: respond in the user's language. Be professional, direct, and precise."
      },
      { role: "user", content: prompt }
    ];

    statusLine.innerHTML = `⚡ ${escapeHtml(modelName)} is generating a response...`;
    const fullModelId = resolvePuterModel(modelInput.value.trim() || "deepseek/deepseek-v4-pro");
    // Reserve the estimated token budget against the 214k tokens/s guardrail.
    const estTokens = Math.ceil((prompt.length + modeNote.length + effortNote.length) / 4) + 4096;
    await acquireTokens(estTokens);

    // Auto-authenticate with Puter if not signed in yet
    if (typeof puter !== "undefined" && puter.auth && typeof puter.auth.isSignedIn === "function") {
      if (!puter.auth.isSignedIn()) {
        statusLine.innerHTML = `🔑 Ro'yxatdan o'tish oynasi ochilmoqda (Puter Auth)...`;
        try {
          await puter.auth.signIn();
        } catch (authErr) {
          console.warn("Puter auth popup skipped/dismissed:", authErr);
        }
      }
    }

    let response;
    try {
      response = await puter.ai.chat(messages, { model: fullModelId, stream: true });
    } catch (chatErr) {
      const errMsg = String(chatErr && chatErr.message ? chatErr.message : chatErr).toLowerCase();
      if (errMsg.includes("auth") || errMsg.includes("sign in") || errMsg.includes("login") || errMsg.includes("unauthorized") || errMsg.includes("quota")) {
        statusLine.innerHTML = `🔑 Puter avtorizatsiyasi talab qilinmoqda — oyna ochilmoqda...`;
        if (typeof puter !== "undefined" && puter.auth && typeof puter.auth.signIn === "function") {
          await puter.auth.signIn();
          response = await puter.ai.chat(messages, { model: fullModelId, stream: true });
        } else {
          throw chatErr;
        }
      } else {
        throw chatErr;
      }
    }

    let fullText = "";
    for await (const part of response) {
      if (part && part.text) {
        fullText += part.text;
        answerDiv.innerHTML = marked.parse(fullText);
        scrollToBottom();
      }
    }

    // Check for tool calls
    const toolMatches = [...fullText.matchAll(/<tool_call>\s*(.*?)\s*<\/tool_call>/gis)];
    if (toolMatches.length > 0) {
      statusLine.innerHTML = `🔧 Executing ${toolMatches.length} tool${toolMatches.length === 1 ? "" : "s"} in parallel...`;
      statusLine.style.display = "block";

      // Build all tool cards first, then run every tool in parallel
      const jobs = toolMatches.map((match) => {
        try {
          const parsed = JSON.parse(match[1]);
          const tName = parsed.name;
          const tArgs = parsed.arguments || parsed.parameters || {};

          const toolCard = document.createElement("div");
          toolCard.className = "tool-step-card";
          toolCard.innerHTML = `
            <div class="tool-header-line">
              <span class="tool-badge-name">🔧 ${escapeHtml(tName)}</span>
              <span style="color:var(--accent-amber)">Running...</span>
            </div>
            <div class="tool-args-preview">${escapeHtml(JSON.stringify(tArgs))}</div>
          `;
          card.insertBefore(toolCard, statusLine);

          return { tName, tArgs, toolCard };
        } catch (err) {
          console.warn("Tool parse error:", err);
          return null;
        }
      }).filter(Boolean);

      // Execute all tools simultaneously
      await Promise.all(jobs.map(async (job) => {
        try {
          const execRes = await apiFetch("/api/tools/execute", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tool_name: job.tName, arguments: job.tArgs })
          });
          const execData = await execRes.json();
          const resBox = document.createElement("div");
          resBox.className = "tool-result-box";
          resBox.textContent = execData.result || "Done";
          job.toolCard.querySelector(".tool-header-line").lastChild.textContent = "✓ Done";
          Object.assign(job.toolCard.querySelector(".tool-header-line").lastChild.style, { color: "var(--accent-emerald)" });
          job.toolCard.appendChild(resBox);
        } catch (err) {
          const resBox = document.createElement("div");
          resBox.className = "tool-result-box";
          resBox.textContent = "Error: " + err.message;
          job.toolCard.appendChild(resBox);
        }
      }));
    }
  } catch (err) {
    statusLine.innerHTML = `<span style="color:var(--accent-rose)">Puter.js error: ${err.message}</span>`;
  }
}

function handleAgentEvent(event, card, statusLine, state) {
  const dashStateBadge = document.getElementById("dash-agent-state");

  if (event.type === "status") {
    statusLine.innerHTML = `⚡ ${escapeHtml(event.data)}`;
    const text = (event.data || "").toLowerCase();
    if (text.includes("guard") || text.includes("sentinel") || text.includes("shield")) {
      updatePipelineStep("step-guard");
      if (dashStateBadge) dashStateBadge.textContent = "STATE: DUAL-SHIELD INSPECT";
    } else if (text.includes("plan") || text.includes("analyz") || text.includes("reason")) {
      updatePipelineStep("step-reason");
      if (dashStateBadge) dashStateBadge.textContent = "STATE: PLANNING / MCTS";
    } else if (text.includes("postcheck") || text.includes("verif") || text.includes("review")) {
      updatePipelineStep("step-verify");
      if (dashStateBadge) dashStateBadge.textContent = "STATE: DEEP VERIFICATION";
    }
  }
  else if (event.type === "thought") {
    updatePipelineStep("step-reason");
    if (dashStateBadge) dashStateBadge.textContent = "STATE: REASONING (CoT)";
    let acc = state.getThought();
    if (!acc) {
      acc = document.createElement("div");
      acc.className = "thought-accordion";
      acc.innerHTML = `
        <div class="thought-header">
          <span>🧠 Reasoning (CoT)</span>
          <span class="acc-toggle">▼</span>
        </div>
        <div class="thought-body"></div>
      `;
      const header = acc.querySelector(".thought-header");
      const body = acc.querySelector(".thought-body");
      header.addEventListener("click", () => {
        body.style.display = body.style.display === "none" ? "block" : "none";
      });
      card.insertBefore(acc, statusLine);
      state.setThought(acc);
    }
    const body = acc.querySelector(".thought-body");
    body.textContent += event.data + "\n";
  }
  else if (event.type === "tool_call") {
    const tName = event.data.name;
    updatePipelineStep("step-exec");
    if (dashStateBadge) dashStateBadge.textContent = `STATE: EXEC (${tName})`;
    const toolCard = document.createElement("div");
    toolCard.className = "tool-step-card";
    const tArgs = JSON.stringify(event.data.arguments || {});
    toolCard.id = `tool-${tName}-${Date.now()}`;
    toolCard.innerHTML = `
      <div class="tool-header-line">
        <span class="tool-badge-name">🔧 ${escapeHtml(tName)}</span>
        <span class="tool-status-tag" style="color:var(--accent-amber)">Running...</span>
      </div>
      <div class="tool-args-preview">${escapeHtml(tArgs)}</div>
    `;
    card.insertBefore(toolCard, statusLine);
  }
  else if (event.type === "metacognition") {
    const metaCard = document.createElement("div");
    metaCard.className = "tool-step-card";
    metaCard.style.borderLeft = "3px solid var(--accent-purple, #a855f7)";
    metaCard.style.background = "rgba(168, 85, 247, 0.08)";
    metaCard.innerHTML = `
      <div class="tool-header-line">
        <span class="tool-badge-name" style="color:var(--accent-purple, #a855f7)">🧠 Metacognitive Overseer [${escapeHtml(event.data.intervention)}]</span>
        <span class="tool-status-tag" style="color:var(--text-muted)">Entropy: ${event.data.entropy ?? 0} | Repetition: ${event.data.repetition ?? 0}</span>
      </div>
      <div class="tool-args-preview" style="color:var(--text-primary);font-style:italic;">${escapeHtml(event.data.recommendation)}</div>
    `;
    card.insertBefore(metaCard, statusLine);
  }
  else if (event.type === "tool_result") {
    const resCard = document.createElement("div");
    resCard.className = "tool-step-card";
    const resText = typeof event.data.result === "string" ? event.data.result : JSON.stringify(event.data.result);
    resCard.innerHTML = `
      <div class="tool-header-line">
        <span class="tool-badge-name" style="color:var(--accent-emerald)">✓ ${escapeHtml(event.data.name)} result</span>
      </div>
      <div class="tool-result-box">${escapeHtml(resText.slice(0, 1000))}${resText.length > 1000 ? "\n...(truncated)" : ""}</div>
    `;
    card.insertBefore(resCard, statusLine);
    if (event.data.name === "mcp_connect_preset") {
      fetchMcpTools();
    }
  }
  else if (event.type === "final_answer") {
    updatePipelineStep("step-final");
    if (dashStateBadge) dashStateBadge.textContent = "STATE: COMPLETED / VERIFIED";
    let ans = state.getAnswer();
    if (!ans) {
      ans = document.createElement("div");
      ans.className = "answer-content";
      card.insertBefore(ans, statusLine);
      state.setAnswer(ans);
    }
    ans.innerHTML = marked.parse(event.data);
    // Refresh telemetry stats after turn completion
    setTimeout(refreshDashboardData, 800);
  }
  else if (event.type === "error") {
    const errDiv = document.createElement("div");
    errDiv.style.color = "var(--accent-rose)";
    errDiv.style.padding = "10px";
    errDiv.style.background = "rgba(244, 63, 94, 0.1)";
    errDiv.style.borderRadius = "8px";
    errDiv.innerHTML = `<strong>Error:</strong> ${escapeHtml(event.data)}`;
    card.insertBefore(errDiv, statusLine);
    if (dashStateBadge) dashStateBadge.textContent = "STATE: ERROR ENCOUNTERED";
  }
}

function scrollToBottom() {
  messagesStream.scrollTop = messagesStream.scrollHeight;
}

function escapeHtml(text) {
  if (!text) return "";
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/* ==========================================================================
   SYSTEM DASHBOARD & TELEMETRY CONTROLLER
   ========================================================================== */

let dashPollingInterval = null;

function switchMainTab(tab) {
  const workspaceView = document.querySelector(".main-body");
  const dashboardView = document.getElementById("dashboard-view");
  const tabWsBtn = document.getElementById("tab-btn-workspace");
  const tabDashBtn = document.getElementById("tab-btn-dashboard");

  if (tab === "dashboard") {
    if (workspaceView) workspaceView.style.display = "none";
    if (dashboardView) dashboardView.style.display = "flex";
    tabWsBtn?.classList.remove("active");
    tabDashBtn?.classList.add("active");
    refreshDashboardData();
    if (!dashPollingInterval) {
      dashPollingInterval = setInterval(() => {
        if (dashboardView && dashboardView.style.display !== "none") {
          refreshDashboardData();
        }
      }, 5000);
    }
  } else {
    if (dashboardView) dashboardView.style.display = "none";
    if (workspaceView) workspaceView.style.display = "flex";
    tabDashBtn?.classList.remove("active");
    tabWsBtn?.classList.add("active");
    if (dashPollingInterval) {
      clearInterval(dashPollingInterval);
      dashPollingInterval = null;
    }
  }
}

function updatePipelineStep(stepId) {
  const steps = ["step-intent", "step-guard", "step-reason", "step-exec", "step-verify", "step-final"];
  steps.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove("active");
  });
  if (stepId) {
    const activeEl = document.getElementById(stepId);
    if (activeEl) activeEl.classList.add("active");
  }
}

async function refreshDashboardData() {
  // 1. Fetch Config & Brain Info
  try {
    const res = await apiFetch("/api/config");
    if (res.ok) {
      const cfg = await res.json();
      const modelEl = document.getElementById("kpi-model");
      const provEl = document.getElementById("kpi-provider");
      if (modelEl) modelEl.textContent = cfg.model || "Unknown";
      if (provEl) provEl.textContent = `Provider: ${(cfg.provider || "ollama").toUpperCase()} · Ready`;
    }
  } catch (e) { /* ignore */ }

  // 2. Fetch Tool Performance Stats
  try {
    const res = await apiFetch("/api/tools/stats");
    if (res.ok) {
      const data = await res.json();
      const stats = data.tools || {};
      const toolNames = Object.keys(stats);
      let totalCalls = 0;
      let totalFailures = 0;
      let rowsHtml = "";

      if (toolNames.length === 0) {
        rowsHtml = `<tr><td colspan="5" class="table-empty" style="text-align:center; padding:15px; color:var(--text-muted);">No tool calls recorded in this session yet.</td></tr>`;
      } else {
        toolNames.forEach(tName => {
          const s = stats[tName];
          const calls = s.calls || 0;
          const fails = s.failures || 0;
          const success = calls - fails;
          const avgSec = s.avg_duration ? `${s.avg_duration.toFixed(2)}s` : "-";
          totalCalls += calls;
          totalFailures += fails;
          rowsHtml += `
            <tr>
              <td><code>${escapeHtml(tName)}</code></td>
              <td><strong>${calls}</strong></td>
              <td style="color:var(--accent-emerald); font-weight:600;">${success}</td>
              <td style="color:${fails > 0 ? 'var(--accent-rose)' : 'var(--text-muted)'}; font-weight:600;">${fails}</td>
              <td>${avgSec}</td>
            </tr>
          `;
        });
      }

      const countEl = document.getElementById("kpi-tools-count");
      const subEl = document.getElementById("kpi-tools-sub");
      const bodyEl = document.getElementById("tools-perf-body");
      if (countEl) countEl.textContent = `${totalCalls} calls`;
      if (subEl) {
        const rate = totalCalls > 0 ? Math.round(((totalCalls - totalFailures) / totalCalls) * 100) : 100;
        subEl.textContent = `Success Rate: ${rate}% · ${totalFailures} failed`;
      }
      if (bodyEl) bodyEl.innerHTML = rowsHtml;
    }
  } catch (e) { /* ignore */ }

  // 3. Fetch Guard State & Cyber Defense
  try {
    const res = await apiFetch("/api/guard/state");
    if (res.ok) {
      const data = await res.json();
      const stateEl = document.getElementById("kpi-security-state");
      const stateSub = document.getElementById("kpi-security-sub");
      if (stateEl) {
        stateEl.textContent = data.dual_shield ? "Dual-Shield Sentinel ON" : "Sentinel Active";
        stateEl.style.color = "var(--accent-emerald)";
      }
      if (stateSub) {
        stateSub.textContent = `Repeat Guard: ${data.repeat_guard ? 'ON' : 'OFF'} · Malformed Guard: ${data.malformed_guard ? 'ON' : 'OFF'}`;
      }
    }
  } catch (e) { /* ignore */ }

  // 4. Fetch Memory Vault Items
  try {
    const res = await apiFetch("/api/memory/vault");
    if (res.ok) {
      const data = await res.json();
      const items = data.items || [];
      const memCount = document.getElementById("kpi-memory-count");
      const memSub = document.getElementById("kpi-memory-sub");
      if (memCount) memCount.textContent = `${items.length} units`;
      if (memSub) memSub.textContent = "SQLite memory store synchronized";
    }
  } catch (e) { /* ignore */ }

  // 5. Fetch Subagent Roles
  try {
    const res = await apiFetch("/api/staff/roles");
    if (res.ok) {
      const data = await res.json();
      const roles = data.roles || [];
      const grid = document.getElementById("swarm-cards-grid");
      const badgeCount = document.getElementById("swarm-roster-count");
      if (badgeCount && roles.length) badgeCount.textContent = `${roles.length} Specialists`;
      if (grid && roles.length > 0) {
        grid.innerHTML = roles.map(r => `
          <div class="swarm-role-card">
            <div class="role-badge">${escapeHtml(r.badge || r.category || 'Specialist')}</div>
            <h4>${escapeHtml(r.name || r.id)}</h4>
            <p>${escapeHtml(r.description || 'Autonomous agent role')}</p>
          </div>
        `).join("");
      }
    }
  } catch (e) { /* ignore */ }

  // 6. Fetch Recent Event Logs
  fetchRecentLogs();
}

async function fetchRecentLogs() {
  try {
    const res = await apiFetch("/api/logs/recent");
    if (res.ok) {
      const data = await res.json();
      const entries = data.logs || [];
      const terminal = document.getElementById("live-log-terminal");
      if (terminal && entries.length > 0) {
        terminal.innerHTML = entries.map(l => {
          let cls = "system";
          if (l.includes("ERROR") || l.includes("failed")) cls = "error";
          else if (l.includes("tool") || l.includes("execute")) cls = "tool";
          else if (l.includes("success") || l.includes("done")) cls = "success";
          return `<div class="log-entry ${cls}">[${new Date().toLocaleTimeString()}] ${escapeHtml(l)}</div>`;
        }).join("");
        terminal.scrollTop = terminal.scrollHeight;
      }
    }
  } catch (e) { /* ignore */ }
}

// Global expose
window.switchMainTab = switchMainTab;
window.refreshDashboardData = refreshDashboardData;
window.fetchRecentLogs = fetchRecentLogs;