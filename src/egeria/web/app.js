"use strict";
/* Egeria · interfaccia. Una pagina per chiedere (testo, immagini o dal vivo), lo storico e i ricordi.
   L'utente non vede mai il formato dell'API: le domande si costruiscono con campi e menu. */

// =================================================================== utilità

function h(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === false || value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "value") node.value = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const ICONS = {
  chat: '<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/>',
  history: '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
  brain: '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M12 5v13"/>',
  settings: '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
  moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
  plus: '<path d="M5 12h14"/><path d="M12 5v14"/>',
  check: '<polyline points="20 6 9 17 4 12"/>',
  alert: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
  x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  trash: '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>',
  search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
  image: '<rect width="18" height="18" x="3" y="3" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.09-3.09a2 2 0 0 0-2.82 0L6 21"/>',
  camera: '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>',
  monitor: '<rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" x2="16" y1="21" y2="21"/><line x1="12" x2="12" y1="17" y2="21"/>',
  film: '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18"/><path d="M3 7.5h4"/><path d="M3 12h18"/><path d="M3 16.5h4"/><path d="M17 3v18"/><path d="M17 7.5h4"/><path d="M17 16.5h4"/>',
  play: '<polygon points="6 3 20 12 6 21 6 3"/>',
  stop: '<rect width="14" height="14" x="5" y="5" rx="1"/>',
  edit: '<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/>',
  save: '<path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7"/><path d="M7 3v4a1 1 0 0 0 1 1h7"/>',
  text: '<path d="M17 6.1H3"/><path d="M21 12.1H3"/><path d="M15.1 18H3"/>',
  split: '<circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M13 6h3a2 2 0 0 1 2 2v7"/><path d="M11 18H8a2 2 0 0 1-2-2V9"/>',
};

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", "icon");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = ICONS[name]; // solo stringhe statiche
  return svg;
}

async function api(method, path, body) {
  const response = await fetch(path, {
    method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Errore del server (${response.status})`);
  return data;
}

function toast(message, kind = "info") {
  const node = h("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" },
    icon(kind === "error" ? "alert" : "check"), h("span", { text: message }));
  document.getElementById("toasts").append(node);
  setTimeout(() => node.remove(), kind === "error" ? 8000 : 3500);
}

async function busy(button, fn) {
  button.setAttribute("aria-busy", "true");
  button.disabled = true;
  try { return await fn(); } finally { button.removeAttribute("aria-busy"); button.disabled = false; }
}

const pct = (p) => `${Math.round(p * 100)}%`;
const fmt = (x) => Number(x).toLocaleString("it-IT", { maximumFractionDigits: Math.abs(x) >= 100 ? 0 : 1 });

function timeAgo(seconds) {
  const diff = Date.now() / 1000 - seconds;
  if (diff < 60) return "adesso";
  if (diff < 3600) return `${Math.floor(diff / 60)} minuti fa`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} ore fa`;
  return new Date(seconds * 1000).toLocaleString("it-IT", { dateStyle: "medium", timeStyle: "short" });
}

function downscale(source, maxSide = 1280) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
      const canvas = h("canvas", { width: Math.round(image.width * scale), height: Math.round(image.height * scale) });
      canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/jpeg", 0.9));
    };
    image.onerror = () => reject(new Error("immagine non leggibile"));
    if (typeof source === "string") image.src = source;
    else { const reader = new FileReader(); reader.onload = () => (image.src = reader.result); reader.onerror = reject; reader.readAsDataURL(source); }
  });
}

function slug(text, used) {
  const base = text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase()
    .replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 40) || "risposta";
  let id = base, n = 2;
  while (used.has(id)) id = `${base}_${n++}`;
  used.add(id);
  return id;
}

function storage(key, fallback) {
  try {
    // Fino a settembre 2026 il progetto si chiamava semLMM: si rileggono anche le chiavi vecchie.
    const value = localStorage.getItem(key) ?? localStorage.getItem(key.replace(/^egeria-/, "semlmm-"));
    return value ? JSON.parse(value) : fallback;
  } catch { return fallback; }
}
function store(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* archiviazione non disponibile */ }
}

// =================================================================== stato

const KINDS = {
  yesno: "Sì / No",
  choice: "Una tra più opzioni",
  scale: "Una scala (dal più basso al più alto)",
  number: "Un numero",
  word: "Una parola o un valore (nome, data, importo…)",
};

let uid = 0;
function newQuestion(partial = {}) {
  return { uid: ++uid, text: "", kind: "yesno", options: [], levels: ["Basso", "Medio", "Alto"], from: 0, to: 100, unit: "", alertOn: "", ...partial };
}

const S = {
  mode: "static",
  images: [],       // {url, path?, dataUrl?}
  questions: [newQuestion()],
  asked: null,      // {api, qids, byUid, labels}
  response: null,
  caseId: null,
  saved: false,
  corrections: {},
  stale: false,
};

const settings = { sensitivity: 0.5, useMemory: true, imageSide: 448, ...storage("egeria-settings", {}) };

// =================================================================== da domande a richiesta API

function niceStep(raw) {
  const power = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / power;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * power;
}

function numberEdges(from, to) {
  const step = niceStep((to - from) / 5);
  const edges = [];
  for (let x = from; x < to - step * 1e-6; x += step) edges.push(Math.round(x * 1e6) / 1e6);
  edges.push(to, null);
  return edges;
}

function intervalLabel(lo, hi, unit) {
  const u = unit ? ` ${unit}` : "";
  if (lo === null) return `meno di ${fmt(hi)}${u}`;
  if (hi === null) return `${fmt(lo)}${u} o più`;
  return `da ${fmt(lo)} a ${fmt(hi)}${u}`;
}

/** Etichette leggibili delle risposte, per chiave, a partire dalla domanda nel formato API. */
function labelsFor(apiQuestion) {
  switch (apiQuestion.type) {
    case "noul": return { true: "Sì", false: "No" };
    case "choice": case "rank":
      return Object.fromEntries(Object.entries(apiQuestion.criteria).map(([k, v]) => [k, v || k]));
    case "score": return Object.fromEntries(apiQuestion.criteria.map((v, i) => [String(i), v]));
    case "number": {
      const bins = Array.isArray(apiQuestion.criteria) ? apiQuestion.criteria : apiQuestion.criteria.bins;
      const unit = Array.isArray(apiQuestion.criteria) ? "" : apiQuestion.criteria.unit || "";
      return { __positional: bins.slice(0, -1).map((lo, i) => intervalLabel(lo, bins[i + 1], unit)) };
    }
    default: return {};
  }
}

function labelOf(labels, result, key) {
  if (labels.__positional) {
    const index = Object.keys(result.probabilities || {}).indexOf(key);
    return index >= 0 ? labels.__positional[index] : key;
  }
  return labels[key] !== undefined ? labels[key] : key;
}

/** Traduce le domande dell'interfaccia nel formato /v1/systemone. Solleva un errore con la domanda da correggere. */
function buildApi({ live = false } = {}) {
  const used = new Set();
  const api = {}, qids = [], byUid = {}, labels = {};
  for (const q of S.questions) {
    const fail = (message) => { const error = new Error(message); error.uid = q.uid; throw error; };
    const text = q.text.trim();
    if (!text) fail("Scrivi la domanda.");
    const qid = slug(text, used);
    let question;
    if (q.kind === "yesno") {
      question = { type: "noul", instructions: text };
      if (q.alertOn) question.alert_if = q.alertOn;
    } else if (q.kind === "choice") {
      const options = [...new Set(q.options.map((o) => o.trim()).filter(Boolean))];
      if (options.length < 2) fail("Aggiungi almeno due opzioni.");
      if (options.length > 26) fail("Al massimo 26 opzioni.");
      const keys = new Set();
      const criteria = Object.fromEntries(options.map((label) => [slug(label, keys), label]));
      question = { type: "choice", instructions: text, criteria };
      if (q.alertOn) { const key = Object.keys(criteria).find((k) => criteria[k] === q.alertOn); if (key) question.alert_if = key; }
    } else if (q.kind === "scale") {
      const levels = q.levels.map((l) => l.trim()).filter(Boolean);
      if (levels.length < 2) fail("Servono almeno due livelli.");
      if (levels.length > 10) fail("Al massimo 10 livelli.");
      question = { type: "score", instructions: text, criteria: levels };
      if (q.alertOn) { const index = levels.indexOf(q.alertOn); if (index >= 0) question.alert_if = String(index); }
    } else if (q.kind === "number") {
      const from = Number(q.from), to = Number(q.to);
      if (!Number.isFinite(from) || !Number.isFinite(to) || from >= to) fail("Indica un intervallo valido: «da» deve essere minore di «a».");
      question = { type: "number", instructions: text, criteria: { bins: numberEdges(from, to), unit: q.unit.trim() } };
    } else {
      question = { type: "open", instructions: text, top_k: 3 };
    }
    api[qid] = question;
    qids.push(qid);
    byUid[q.uid] = qid;
    labels[qid] = labelsFor(question);
  }
  if (!qids.length) throw new Error("Aggiungi almeno una domanda.");
  return { api, qids, byUid, labels };
}

/** Da domande nel formato API (es. dallo storico) a domande dell'interfaccia. */
function fromApi(apiQuestions) {
  return Object.values(apiQuestions).map((q) => {
    switch (q.type) {
      case "noul": return newQuestion({ text: q.instructions, kind: "yesno" });
      case "choice": case "rank": return newQuestion({ text: q.instructions, kind: "choice", options: Object.entries(q.criteria).map(([k, v]) => v || k) });
      case "score": return newQuestion({ text: q.instructions, kind: "scale", levels: [...q.criteria] });
      case "number": {
        const bins = (Array.isArray(q.criteria) ? q.criteria : q.criteria.bins).filter((x) => x !== null);
        return newQuestion({ text: q.instructions, kind: "number", from: bins[0], to: bins[bins.length - 1], unit: (q.criteria && q.criteria.unit) || "" });
      }
      default: return newQuestion({ text: q.instructions, kind: "word" });
    }
  });
}

function modelKey(result) {
  switch (result.type) {
    case "noul": return result.noul >= 0.5 ? "true" : "false";
    case "choice": return result.choice;
    case "rank": return result.ranking[0];
    case "score": case "number": { const entries = Object.entries(result.probabilities); return entries.reduce((a, b) => (b[1] > a[1] ? b : a))[0]; }
    case "open": return result.answer;
    default: return null;
  }
}

// =================================================================== schede delle domande

function markStale() {
  if (!S.response || S.stale || S.mode === "live") return;
  S.stale = true;
  document.querySelectorAll(".answer").forEach((node) => node.classList.add("stale"));
  document.getElementById("teach").hidden = true;
  document.getElementById("ask-status").textContent = "Hai cambiato qualcosa: premi «Chiedi» per aggiornare le risposte.";
}

function renderQuestions() {
  const list = document.getElementById("question-list");
  list.replaceChildren(...S.questions.map(questionCard));
}

function replaceCard(q, focusSelector) {
  const old = document.querySelector(`.question[data-uid="${q.uid}"]`);
  if (!old) return;
  const card = questionCard(q);
  old.replaceWith(card);
  if (focusSelector) { const target = card.querySelector(focusSelector); if (target) target.focus(); }
}

function chipsEditor(q, field, placeholder, ordered) {
  const input = h("input", { type: "text", class: "chip-input", placeholder, "aria-label": placeholder });
  const add = () => {
    const values = input.value.split(",").map((v) => v.trim()).filter(Boolean);
    input.value = ""; // prima di ridisegnare: il blur del campo rimosso non deve riaggiungere
    if (!values.length) return;
    q[field].push(...values.filter((v) => !q[field].includes(v)));
    markStale();
    replaceCard(q, ".chip-input");
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); add(); }
    else if (event.key === "Backspace" && !input.value && q[field].length) { q[field].pop(); markStale(); replaceCard(q, ".chip-input"); }
  });
  input.addEventListener("blur", () => { if (input.isConnected && input.value.trim()) add(); });
  return h("div", { class: "chips-editor" },
    q[field].map((value, index) => h("span", { class: "chip" },
      ordered ? h("span", { class: "chip-order", text: `${index + 1}.` }) : null,
      h("span", { text: value }),
      h("button", { type: "button", "aria-label": `Togli «${value}»`, onclick: () => { q[field].splice(index, 1); markStale(); replaceCard(q, ".chip-input"); } }, icon("x")))),
    input);
}

function questionCard(q) {
  const textInput = h("input", { type: "text", class: "question-text", value: q.text, placeholder: "Scrivi la domanda, es. «Il cliente è arrabbiato?»", "aria-label": "Domanda" });
  textInput.addEventListener("input", () => { q.text = textInput.value; markStale(); });
  const kind = h("select", { "aria-label": "Tipo di risposta" }, Object.entries(KINDS).map(([value, label]) =>
    h("option", { value, text: label, selected: q.kind === value })));
  kind.addEventListener("change", () => { q.kind = kind.value; q.alertOn = ""; markStale(); replaceCard(q); });
  const remove = h("button", { class: "icon-button small", type: "button", "aria-label": "Togli questa domanda", title: "Togli questa domanda",
    onclick: () => { S.questions = S.questions.filter((x) => x !== q); if (!S.questions.length) S.questions.push(newQuestion()); markStale(); renderQuestions(); } }, icon("trash"));

  const card = h("li", { class: "question", "data-uid": q.uid },
    h("div", { class: "question-top" }, h("span", { class: "question-number", "aria-hidden": "true" }), textInput, remove),
    h("div", { class: "question-row" }, h("span", { text: "Risposta:" }), kind));

  if (q.kind === "choice") {
    card.append(chipsEditor(q, "options", "Aggiungi un'opzione e premi Invio", false));
  } else if (q.kind === "scale") {
    card.append(h("p", { class: "question-hint", text: "I livelli vanno dal più basso al più alto." }), chipsEditor(q, "levels", "Aggiungi un livello e premi Invio", true));
  } else if (q.kind === "number") {
    const from = h("input", { type: "number", value: q.from, "aria-label": "Valore minimo" });
    const to = h("input", { type: "number", value: q.to, "aria-label": "Valore massimo" });
    const unit = h("input", { type: "text", class: "unit", value: q.unit, placeholder: "unità (es. euro)", "aria-label": "Unità di misura" });
    from.addEventListener("input", () => { q.from = from.value; markStale(); });
    to.addEventListener("input", () => { q.to = to.value; markStale(); });
    unit.addEventListener("input", () => { q.unit = unit.value; markStale(); });
    card.append(h("div", { class: "question-row" }, h("span", { text: "Di solito tra" }), from, h("span", { text: "e" }), to, unit));
  } else if (q.kind === "word") {
    card.append(h("p", { class: "question-hint", text: "Il modello legge o risponde con una parola o un valore: un nome, un giorno, una data, un importo, un codice. Per i documenti conviene la qualità «Massima» nelle impostazioni." }));
  }

  if (S.mode === "live" && q.kind !== "number" && q.kind !== "word") {
    const choices = q.kind === "yesno" ? [["true", "Sì"], ["false", "No"]] : (q.kind === "choice" ? q.options : q.levels).map((v) => [v, v]);
    const alertSelect = h("select", { "aria-label": "Avvisami quando la risposta è" },
      h("option", { value: "", text: "mai" }), choices.map(([value, label]) => h("option", { value, text: label, selected: q.alertOn === value })));
    alertSelect.addEventListener("change", () => { q.alertOn = alertSelect.value; });
    card.append(h("div", { class: "question-row" }, h("span", { text: "Avvisami quando la risposta è" }), alertSelect));
  }

  const error = h("p", { class: "question-error", role: "alert" });
  error.hidden = true;
  card.append(error);

  if (S.asked && S.response) {
    const qid = S.asked.byUid[q.uid];
    const result = qid && S.response.answers[qid];
    if (result) card.append(answerBlock(q, qid, result));
  }
  return card;
}

// =================================================================== risposte

function answerBlock(q, qid, result) {
  const labels = S.asked.labels[qid] || {};
  const top = modelKey(result);
  const label = (key) => labelOf(labels, result, key);
  let value, sure, bars = [], extra = null;
  if (result.type === "noul") {
    value = result.noul >= 0.5 ? "Sì" : "No";
    sure = Math.max(result.noul, 1 - result.noul);
    bars = [["true", result.noul], ["false", 1 - result.noul]];
  } else if (result.type === "open") {
    value = result.answer;
    sure = result.confidence;
    const others = result.candidates.slice(1).map((c) => c.text);
    if (others.length) extra = `Altre possibilità: ${others.join(", ")}`;
  } else if (result.type === "number") {
    value = `circa ${fmt(result.value)}${result.unit ? " " + result.unit : ""}`;
    sure = result.probabilities[top];
    extra = `Più probabile: ${label(top)} · quasi certamente tra ${fmt(result.interval[0])} e ${fmt(result.interval[1])}`;
    bars = Object.entries(result.probabilities);
  } else {
    value = label(top);
    sure = result.probabilities[top];
    bars = Object.entries(result.probabilities);
    if (result.type === "choice" || result.type === "rank") bars.sort((a, b) => b[1] - a[1]);
  }
  const classes = ["answer"];
  if (result.status === "uncertain") classes.push("uncertain");
  if (S.stale) classes.push("stale");
  if (S.mode === "live" && Live.alertValue(qid, result) >= Live.threshold()) classes.push("alerting");

  const block = h("div", { class: classes.join(" "), "aria-live": S.mode === "live" ? "off" : "polite" },
    h("div", { class: "answer-line" },
      h("span", { class: "answer-value", text: value }),
      h("span", { class: "answer-sure", text: `sicuro al ${pct(sure)}` }),
      result.status === "uncertain" ? h("span", { class: "badge warn" }, icon("alert"), "Non sono sicuro") : null),
    bars.length ? h("div", { class: "bars" }, bars.map(([key, p]) => h("div", { class: `bar-row${key === top ? " top" : ""}` },
      h("span", { class: "bar-label", text: label(key), title: label(key) }),
      h("span", { class: "bar-track", role: "img", "aria-label": `${label(key)}: ${pct(p)}` }, h("span", { class: "bar-fill", style: `width:${Math.max(0, Math.min(1, p)) * 100}%` })),
      h("span", { class: "bar-value", text: pct(p) })))) : null,
    extra ? h("p", { class: "muted small", text: extra }) : null);

  if (result.memory) {
    const same = result.memory.answer === top;
    block.append(h("div", { class: "memory-hint" }, icon("brain"),
      h("span", { text: `Nei casi simili che ricordo (${result.memory.support}) avevi deciso: ` }),
      h("b", { text: label(result.memory.answer) }),
      h("span", { class: `badge ${same ? "ok" : "disagree"}` }, icon(same ? "check" : "split"), same ? "uguale" : "diverso")));
  }

  if (S.mode === "static" && S.caseId) block.append(correctionControl(qid, result, label, top));
  return block;
}

function correctionControl(qid, result, label, top) {
  const container = h("div", { class: "correction" });
  const current = S.corrections[qid];
  const render = (open) => {
    container.replaceChildren();
    if (!open) {
      if (current !== undefined && current !== top) {
        container.append(h("span", { class: "corrected", text: `Corretto in: ${label(current)}` }),
          h("button", { class: "button ghost small", type: "button", onclick: () => { delete S.corrections[qid]; refreshAnswers(); } }, "Annulla correzione"));
      } else {
        container.append(h("button", { class: "button ghost small", type: "button", onclick: () => render(true) }, icon("edit"), "Correggi"));
      }
      return;
    }
    let control;
    if (result.type === "open") {
      control = h("input", { type: "text", value: current || top, "aria-label": "Risposta corretta" });
    } else {
      const keys = result.type === "noul" ? ["true", "false"] : Object.keys(result.probabilities);
      control = h("select", { "aria-label": "Risposta corretta" }, keys.map((key) => h("option", { value: key, text: label(key), selected: key === (current || top) })));
    }
    container.append(h("span", { class: "muted small", text: "La risposta giusta è:" }), control,
      h("button", { class: "button small", type: "button", onclick: () => { S.corrections[qid] = control.value; refreshAnswers(); } }, icon("check"), "Ok"),
      h("button", { class: "button ghost small", type: "button", onclick: () => render(false) }, "Annulla"));
    control.focus();
  };
  render(false);
  return container;
}

function refreshAnswers() {
  renderQuestions();
  renderTeach();
}

function renderTeach() {
  const panel = document.getElementById("teach");
  const button = document.getElementById("teach-button");
  panel.hidden = !(S.mode === "static" && S.response && S.caseId && !S.stale);
  panel.classList.toggle("saved", S.saved);
  panel.querySelector(".teach-title").textContent = S.saved ? "Salvato nei ricordi" : "Le risposte vanno bene?";
  const corrections = Object.entries(S.corrections).filter(([qid, key]) => S.response && S.response.answers[qid] && key !== modelKey(S.response.answers[qid])).length;
  button.replaceChildren(icon(S.saved ? "check" : "save"),
    S.saved ? "Aggiorna il ricordo" : corrections ? `Salva nei ricordi (${corrections} ${corrections === 1 ? "correzione" : "correzioni"})` : "Sì, salva nei ricordi");
}

// =================================================================== chiedere

async function ask() {
  if (S.mode === "live") { Live.running ? Live.stop() : Live.start(); return; }
  const text = document.getElementById("input-text").value.trim();
  if (!text && !S.images.length) { toast("Scrivi un testo o aggiungi un'immagine da guardare.", "error"); document.getElementById("input-text").focus(); return; }
  let built;
  try { built = buildApi(); } catch (error) { showQuestionError(error); return; }
  clearQuestionErrors();
  const button = document.getElementById("ask-button");
  const status = document.getElementById("ask-status");
  await busy(button, async () => {
    status.textContent = "Sto guardando…";
    const started = performance.now();
    try {
      const images = [];
      for (const image of S.images) {
        if (!image.path) { const upload = await api("POST", "/api/media", { data: image.dataUrl }); image.path = upload.path; }
        images.push({ type: "image", path: image.path });
      }
      const state = images.length ? [...(text ? [{ type: "text", text }] : []), ...images] : text;
      const request = { state, questions: built.api, min_confidence: settings.sensitivity, image_max_side: settings.imageSide };
      request.memory = settings.useMemory ? { recall: 3 } : { recall: 0 };
      const created = await api("POST", "/api/cases", { request });
      Object.assign(S, { asked: built, response: created.response, caseId: created.id, saved: false, corrections: {}, stale: false });
      refreshAnswers();
      const uncertain = created.flags.uncertain.length + created.flags.disagreements.length;
      status.textContent = `Fatto in ${((performance.now() - started) / 1000).toFixed(1)} s` +
        (uncertain ? " · alcune risposte vanno controllate" : "");
      refreshInfo();
    } catch (error) {
      status.textContent = "";
      toast(error.message, "error");
    }
  });
}

function showQuestionError(error) {
  clearQuestionErrors();
  if (error.uid) {
    const card = document.querySelector(`.question[data-uid="${error.uid}"]`);
    card.classList.add("invalid");
    const node = card.querySelector(".question-error");
    node.textContent = error.message;
    node.hidden = false;
    card.scrollIntoView({ behavior: "smooth", block: "center" });
  } else {
    toast(error.message, "error");
  }
}

function clearQuestionErrors() {
  document.querySelectorAll(".question.invalid").forEach((card) => card.classList.remove("invalid"));
  document.querySelectorAll(".question-error").forEach((node) => { node.hidden = true; });
}

async function teach() {
  const decisions = {};
  for (const qid of S.asked.qids) {
    const result = S.response.answers[qid];
    if (result) decisions[qid] = S.corrections[qid] !== undefined ? S.corrections[qid] : modelKey(result);
  }
  await busy(document.getElementById("teach-button"), async () => {
    try {
      await api("POST", `/api/cases/${encodeURIComponent(S.caseId)}/review`, { decisions, note: document.getElementById("teach-note").value.trim() });
      S.saved = true;
      renderTeach();
      toast("Salvato nei ricordi: lo userò per i casi simili.", "success");
      refreshInfo();
    } catch (error) { toast(error.message, "error"); }
  });
}

// =================================================================== cosa guardare

function renderImages() {
  document.getElementById("image-list").replaceChildren(...S.images.map((image, index) => h("li", {},
    h("img", { src: image.url, alt: `Immagine ${index + 1}` }),
    h("button", { class: "icon-button", type: "button", "aria-label": `Togli l'immagine ${index + 1}`, onclick: () => { S.images.splice(index, 1); markStale(); renderImages(); renderQuestions(); } }, icon("x")))));
}

async function addImageFiles(files) {
  for (const file of files) {
    try {
      const dataUrl = await downscale(file, 1280);
      S.images.push({ url: dataUrl, dataUrl });
    } catch (error) { toast(`${file.name || "Immagine"}: ${error.message}`, "error"); }
  }
  markStale();
  renderImages();
  renderQuestions();
}

function setMode(mode) {
  if (Live.running && mode !== "live") Live.stop();
  S.mode = mode;
  for (const button of document.querySelectorAll("#mode-switch button")) button.setAttribute("aria-checked", String(button.dataset.mode === mode));
  document.getElementById("static-input").hidden = mode !== "static";
  document.getElementById("live-input").hidden = mode !== "live";
  S.response = null; S.asked = null; S.caseId = null; S.stale = false;
  renderQuestions();
  renderTeach();
  renderAskButton();
  document.getElementById("ask-status").textContent = "";
}

function renderAskButton() {
  const button = document.getElementById("ask-button");
  if (S.mode === "live") {
    button.replaceChildren(icon(Live.running ? "stop" : "play"), Live.running ? "Ferma" : "Avvia");
    button.disabled = !Live.ready;
  } else {
    button.replaceChildren(icon("chat"), "Chiedi");
    button.disabled = false;
  }
}

function resetAll(questions = [newQuestion()]) {
  if (Live.running) Live.stop();
  document.getElementById("input-text").value = "";
  document.getElementById("teach-note").value = "";
  Object.assign(S, { images: [], questions, asked: null, response: null, caseId: null, saved: false, corrections: {}, stale: false });
  renderImages();
  renderQuestions();
  renderTeach();
  document.getElementById("ask-status").textContent = "";
}

// =================================================================== esempi e domande salvate

const TICKET = "Buongiorno, sono tre giorni che i pagamenti ai nostri fornitori falliscono con l'errore 'IBAN non valido', ma l'IBAN è corretto e fino alla settimana scorsa funzionava. Abbiamo stipendi da pagare venerdì. Se non risolvete entro domani passiamo a un altro fornitore.";

const EXAMPLES = [
  { label: "Messaggio di un cliente", mode: "static", text: TICKET, questions: [
    { text: "Il cliente è urgente?", kind: "yesno" },
    { text: "Quale reparto deve occuparsene?", kind: "choice", options: ["Pagamenti", "Assistenza tecnica", "Commerciale"] },
    { text: "Quanto è arrabbiato il cliente?", kind: "scale", levels: ["Calmo", "Infastidito", "Arrabbiato", "Furioso"] },
    { text: "Entro quale giorno vanno pagati gli stipendi?", kind: "word" },
  ] },
  { label: "Foto di un incidente", mode: "static", image: "examples/immagini/incidente_auto.jpg", questions: [
    { text: "Il veicolo è danneggiato?", kind: "yesno" },
    { text: "Quanto è grave il danno?", kind: "scale", levels: ["Nessun danno", "Lieve", "Moderato", "Grave"] },
    { text: "Di che colore è l'auto?", kind: "choice", options: ["Bianca", "Nera", "Rossa", "Blu"] },
  ] },
  { label: "Foto di uno scontrino", mode: "static", image: "examples/immagini/scontrino.jpg", questions: [
    { text: "Qual è il totale dello scontrino?", kind: "number", from: 0, to: 50, unit: "euro" },
    { text: "È stato pagato con la carta?", kind: "yesno" },
    { text: "Che tipo di negozio è?", kind: "choice", options: ["Supermercato", "Ristorante", "Farmacia", "Abbigliamento"] },
  ] },
  { label: "Dal vivo: controllo incendi", mode: "live", questions: [
    { text: "C'è un incendio o del fumo?", kind: "yesno", alertOn: "true" },
    { text: "Ci sono persone nell'immagine?", kind: "yesno" },
  ] },
];

function hasWork() {
  return document.getElementById("input-text").value.trim() || S.images.length || S.questions.some((q) => q.text.trim());
}

function loadExample(example) {
  if (hasWork() && !window.confirm("Sostituire quello che hai scritto con l'esempio?")) return;
  setMode(example.mode);
  resetAll(example.questions.map((q) => newQuestion({ ...q, options: [...(q.options || [])], levels: [...(q.levels || ["Basso", "Medio", "Alto"])] })));
  if (example.text) document.getElementById("input-text").value = example.text;
  if (example.image) { S.images = [{ url: `/files?path=${encodeURIComponent(example.image)}`, path: example.image }]; renderImages(); renderQuestions(); }
  toast(example.mode === "live" ? "Esempio caricato: scegli la telecamera, lo schermo o un video e premi «Avvia»." : "Esempio caricato: premi «Chiedi».");
}

function savedSets() { return storage("egeria-domande", {}); }

function renderSavedSets() {
  const select = document.getElementById("saved-sets");
  const sets = savedSets();
  select.replaceChildren(h("option", { value: "", text: Object.keys(sets).length ? "Domande salvate…" : "Nessuna domanda salvata" }),
    ...Object.keys(sets).sort().map((name) => h("option", { value: name, text: name })));
}

function plainQuestions() {
  return S.questions.filter((q) => q.text.trim()).map(({ uid: _, ...rest }) => rest);
}

// =================================================================== dal vivo

const Live = {
  stream: null, ready: false, running: false, frames: [], latencies: [], cooldown: {},

  init() {
    const labels = { webcam: ["camera", "Telecamera"], screen: ["monitor", "Schermo"], file: ["film", "Video da file"] };
    for (const button of document.querySelectorAll(".live-sources [data-source]")) {
      const [iconName, label] = labels[button.dataset.source];
      button.append(icon(iconName), label);
      button.addEventListener("click", () => this.choose(button.dataset.source));
    }
    this.video = document.getElementById("live-video");
    document.getElementById("live-file").addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (!file) return;
      this.stopStream();
      this.video.srcObject = null;
      this.video.src = URL.createObjectURL(file);
      this.video.loop = true;
      this.setReady();
    });
    const threshold = document.getElementById("live-threshold");
    threshold.addEventListener("input", () => { document.getElementById("live-threshold-value").textContent = pct(threshold.value); });
    document.getElementById("clear-alerts").addEventListener("click", () => this.renderEmptyAlerts());
    this.renderEmptyAlerts();
  },

  threshold() { return Number(document.getElementById("live-threshold").value); },

  renderEmptyAlerts() {
    document.getElementById("alert-list").replaceChildren(h("li", { class: "muted small", text: "Nessun avviso. Scegli in ogni domanda «Avvisami quando la risposta è…»." }));
  },

  async choose(source) {
    for (const button of document.querySelectorAll(".live-sources [data-source]")) button.setAttribute("aria-checked", String(button.dataset.source === source));
    if (source === "file") { document.getElementById("live-file").click(); return; }
    try {
      this.stopStream();
      this.stream = source === "webcam"
        ? await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 } }, audio: false })
        : await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
      this.video.removeAttribute("src");
      this.video.srcObject = this.stream;
      this.stream.getVideoTracks()[0].addEventListener("ended", () => { this.stop(); this.ready = false; renderAskButton(); });
      this.setReady();
    } catch (error) {
      toast(`Non riesco ad accedere: ${error.message}`, "error");
    }
  },

  setReady() {
    this.video.play().catch(() => {});
    document.getElementById("live-placeholder").hidden = true;
    this.ready = true;
    renderAskButton();
  },

  stopStream() {
    if (this.stream) this.stream.getTracks().forEach((track) => track.stop());
    this.stream = null;
  },

  grab(maxSide) {
    const width = this.video.videoWidth, height = this.video.videoHeight;
    if (!width || !height) return null;
    const scale = Math.min(1, maxSide / Math.max(width, height));
    const canvas = h("canvas", { width: Math.round(width * scale), height: Math.round(height * scale) });
    canvas.getContext("2d").drawImage(this.video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  },

  async start() {
    try { buildApi({ live: true }); } catch (error) { showQuestionError(error); return; }
    clearQuestionErrors();
    this.running = true;
    this.frames = []; this.latencies = [];
    renderAskButton();
    document.getElementById("ask-status").textContent = "Sto guardando dal vivo…";
    while (this.running) {
      const started = performance.now();
      let built;
      try { built = buildApi({ live: true }); } catch { built = S.asked; } // domanda in modifica: tiene le ultime valide
      const maxSide = Math.min(settings.imageSide, 448);
      const frame = this.grab(maxSide);
      if (frame && built) {
        try {
          const response = await api("POST", "/api/monitor/frame", { frame, questions: built.api, max_side: maxSide });
          S.asked = built;
          S.response = response;
          this.frames.push(performance.now());
          this.latencies.push(response.server_ms);
          this.updateAnswers();
          this.checkAlerts(response, frame, built);
        } catch (error) {
          toast(`Mi sono fermato: ${error.message}`, "error");
          this.stop();
          break;
        }
      }
      const fps = Number(document.getElementById("live-fps").value);
      const wait = fps ? 1000 / fps - (performance.now() - started) : 0;
      await new Promise((resolve) => setTimeout(resolve, Math.max(wait, frame ? 0 : 250)));
    }
  },

  stop() {
    this.running = false;
    renderAskButton();
    document.getElementById("ask-status").textContent = "Fermato.";
  },

  updateAnswers() {
    // Aggiorna solo i riquadri delle risposte, senza toccare i campi che l'utente sta modificando.
    for (const q of S.questions) {
      const card = document.querySelector(`.question[data-uid="${q.uid}"]`);
      if (!card) continue;
      const qid = S.asked.byUid[q.uid];
      const result = qid && S.response.answers[qid];
      const old = card.querySelector(".answer");
      if (!result) { if (old) old.remove(); continue; }
      const block = answerBlock(q, qid, result);
      if (old) old.replaceWith(block); else card.append(block);
    }
    this.frames = this.frames.filter((t) => performance.now() - t < 5000);
    const sorted = [...this.latencies.slice(-30)].sort((a, b) => a - b);
    document.getElementById("live-stats").replaceChildren(
      h("span", { class: "badge", text: `${(this.frames.length / 5).toFixed(1)} immagini/s` }),
      h("span", { class: "badge", text: `${Math.round(sorted[Math.floor(sorted.length / 2)] || 0)} ms` }));
  },

  alertValue(qid, result) {
    const question = S.asked && S.asked.api[qid];
    if (!question || question.alert_if === undefined) return -1;
    if (result.type === "noul") return question.alert_if === "true" ? result.noul : 1 - result.noul;
    const value = (result.probabilities || {})[question.alert_if];
    return value === undefined ? -1 : value;
  },

  checkAlerts(response, frame, built) {
    const now = Date.now();
    for (const [qid, result] of Object.entries(response.answers)) {
      const value = this.alertValue(qid, result);
      if (value < this.threshold() || (this.cooldown[qid] && now - this.cooldown[qid] < 5000)) continue;
      this.cooldown[qid] = now;
      const question = built.api[qid];
      const answer = labelOf(built.labels[qid], result, question.alert_if);
      const list = document.getElementById("alert-list");
      if (list.querySelector(".muted")) list.replaceChildren();
      const item = h("li", { class: "alert-item" },
        h("img", { src: frame, alt: "Immagine al momento dell'avviso" }),
        h("div", {},
          h("div", {}, h("span", { class: "badge danger" }, icon("alert"), "Avviso"), " ", h("span", { class: "muted small", text: new Date().toLocaleTimeString("it-IT") })),
          h("p", { text: `${question.instructions} → ${answer} (${pct(value)})` }),
          h("div", { class: "alert-actions" },
            h("button", { class: "button small", type: "button", onclick: (event) => this.remember(event.currentTarget, frame, response, built, question.instructions) }, icon("save"), "Salva nei ricordi"),
            h("button", { class: "button ghost small", type: "button", onclick: () => item.remove() }, "Ignora"))));
      list.prepend(item);
    }
  },

  async remember(button, frame, response, built, text) {
    await busy(button, async () => {
      try {
        const upload = await api("POST", "/api/media", { data: frame });
        const decisions = Object.fromEntries(Object.entries(response.answers).map(([qid, r]) => [qid, modelKey(r)]));
        await api("POST", "/api/memory", {
          state: [{ type: "text", text: "Immagine dal vivo:" }, { type: "image", path: upload.path }],
          decisions, questions: built.api, note: `Avviso: ${text}`, source: "dal vivo",
        });
        button.replaceChildren(icon("check"), "Salvato");
        button.disabled = true;
        refreshInfo();
      } catch (error) { toast(error.message, "error"); }
    });
  },
};

// =================================================================== storico

const History = {
  view: "da_rivedere",

  async load() {
    for (const button of document.querySelectorAll("#history-filter button")) button.setAttribute("aria-checked", String(button.dataset.view === this.view));
    try {
      const data = await api("GET", `/api/cases?view=${this.view}`);
      for (const [key, value] of Object.entries(data.counts)) {
        const node = document.querySelector(`#history-filter [data-count="${key}"]`);
        if (node) node.textContent = value;
      }
      const list = document.getElementById("history-list");
      if (!data.cases.length) {
        list.replaceChildren(h("li", { class: "empty", text: this.view === "da_rivedere" ? "Niente da controllare." : "Non hai ancora fatto domande. Vai su «Chiedi»." }));
        return;
      }
      list.replaceChildren(...data.cases.map((item) => h("li", {}, h("button", { class: "history-item", type: "button", onclick: () => this.open(item.id) },
        item.thumb ? h("img", { class: "history-thumb", src: item.thumb, alt: "" }) : h("span", { class: "history-thumb" }, icon("text")),
        h("span", {}, h("span", { class: "history-text", text: item.text || "Immagine senza testo" }),
          h("span", { class: "history-meta", text: [timeAgo(item.created), `${item.questions} ${item.questions === 1 ? "domanda" : "domande"}`,
            item.images ? `${item.images} ${item.images === 1 ? "immagine" : "immagini"}` : null].filter(Boolean).join(" · ") })),
        h("span", { class: "history-badges" }, ...this.badges(item))))));
    } catch (error) { toast(error.message, "error"); }
  },

  badges(item) {
    if (item.review !== "in_attesa") return [h("span", { class: "badge ok" }, icon("check"), "Nei ricordi")];
    const badges = [];
    if (item.flags.uncertain.length) badges.push(h("span", { class: "badge warn" }, icon("alert"), `${item.flags.uncertain.length} incerte`));
    if (item.flags.disagreements.length) badges.push(h("span", { class: "badge disagree" }, icon("split"), `${item.flags.disagreements.length} diverse dai ricordi`));
    return badges;
  },

  async open(id) {
    try {
      const item = await api("GET", `/api/cases/${encodeURIComponent(id)}`);
      showView("chiedi");
      setMode("static");
      const state = item.request.state;
      const questions = fromApi(item.request.questions);
      resetAll(questions);
      if (Array.isArray(state)) {
        document.getElementById("input-text").value = state.filter((p) => p.type === "text").map((p) => p.text).join("\n");
        S.images = state.filter((p) => p.type === "image").map((p, i) => ({ url: item.images[i], path: p.path }));
      } else {
        document.getElementById("input-text").value = typeof state === "string" ? state : JSON.stringify(state, null, 2);
      }
      const qids = Object.keys(item.request.questions);
      const byUid = Object.fromEntries(questions.map((q, i) => [q.uid, qids[i]]));
      const labels = Object.fromEntries(qids.map((qid) => [qid, labelsFor(item.request.questions[qid])]));
      Object.assign(S, { asked: { api: item.request.questions, qids, byUid, labels }, response: item.response, caseId: item.id, saved: item.review !== "in_attesa" });
      S.corrections = {};
      if (item.final) for (const [qid, key] of Object.entries(item.final)) if (item.response.answers[qid] && key !== modelKey(item.response.answers[qid])) S.corrections[qid] = key;
      document.getElementById("teach-note").value = item.note || "";
      renderImages();
      refreshAnswers();
      document.getElementById("ask-status").textContent = `Analisi del ${new Date(item.created * 1000).toLocaleString("it-IT")}`;
    } catch (error) { toast(error.message, "error"); }
  },
};

// =================================================================== ricordi

const Memories = {
  async load() {
    try {
      const data = await api("GET", "/api/memory?limit=300");
      document.getElementById("memory-total").textContent = `${data.total} ${data.total === 1 ? "ricordo" : "ricordi"}`;
      document.getElementById("memory-show-all").hidden = true;
      this.render(data.items, "Nessun ricordo ancora. Fai una domanda e premi «Salva nei ricordi».");
    } catch (error) { toast(error.message, "error"); }
  },

  render(items, emptyText) {
    const grid = document.getElementById("memory-grid");
    if (!items.length) { grid.replaceChildren(h("p", { class: "empty", text: emptyText })); return; }
    grid.replaceChildren(...items.map((item) => h("article", { class: "memory-card" },
      item.images.length ? h("img", { src: item.images[0], alt: "", loading: "lazy" }) : null,
      h("div", { class: "memory-body" },
        item.similarity !== undefined ? h("span", { class: "similar", text: `somiglianza ${pct(Math.max(0, item.similarity))}` }) : null,
        item.plain ? h("p", { class: "memory-text", text: item.plain }) : null,
        h("ul", { class: "qa" }, item.readable.map((qa) => h("li", {}, h("span", { text: `${qa.question} ` }), h("b", { text: qa.answer })))),
        item.note ? h("p", { class: "muted small", text: item.note }) : null,
        h("div", { class: "memory-foot" },
          h("span", { text: item.meta && item.meta.created ? timeAgo(item.meta.created) : "" }),
          h("button", { class: "button ghost small danger", type: "button", "aria-label": "Dimentica questo ricordo", onclick: () => this.remove(item.id) }, icon("trash"), "Dimentica"))))));
  },

  async search(state, button) {
    const run = async () => {
      try {
        const data = await api("POST", "/api/memory/search", { state, k: 12 });
        document.getElementById("memory-show-all").hidden = false;
        document.getElementById("memory-total").textContent = "I più simili";
        this.render(data.items, "Nessun ricordo ancora.");
      } catch (error) { toast(error.message, "error"); }
    };
    return button ? busy(button, run) : run();
  },

  async remove(id) {
    if (!window.confirm("Dimenticare questo ricordo? Non si potrà recuperare.")) return;
    try { await api("DELETE", `/api/memory/${encodeURIComponent(id)}`); toast("Ricordo dimenticato", "success"); this.load(); refreshInfo(); }
    catch (error) { toast(error.message, "error"); }
  },
};

// =================================================================== navigazione, impostazioni, avvio

const VIEWS = { chiedi: ["chat", "Chiedi"], storico: ["history", "Storico"], ricordi: ["brain", "Ricordi"] };

function showView(name) {
  for (const tab of document.querySelectorAll(".tab")) {
    if (tab.dataset.view === name) tab.setAttribute("aria-current", "page"); else tab.removeAttribute("aria-current");
  }
  for (const view of document.querySelectorAll(".view")) view.hidden = view.id !== `view-${name}`;
  history.replaceState(null, "", `#${name}`);
  if (name === "storico") History.load();
  if (name === "ricordi") Memories.load();
}

async function refreshInfo() {
  const status = document.getElementById("server-status");
  try {
    const info = await api("GET", "/api/info");
    const modelUp = info.model_status === "ok";
    status.classList.toggle("offline", !modelUp);
    status.textContent = modelUp ? "Pronto" : "Modello non raggiungibile";
    status.title = modelUp ? `${info.model} · ${info.gpu || info.device} · ${info.model_url}`
      : `${info.model_error}. Storico e ricordi restano consultabili.`;
    document.getElementById("settings-info").textContent = modelUp
      ? `Modello: ${info.model} · ${info.gpu || info.device} · ${info.memories} ricordi`
      : `Il modello non risponde su ${info.model_url}: avvialo con «egeria model-server». ${info.memories} ricordi`;
    setPill("storico", info.cases.da_rivedere, false);
    setPill("ricordi", info.memories, true);
  } catch {
    status.classList.add("offline");
    status.textContent = "Server web non raggiungibile";
  }
}

function setPill(view, count, neutral) {
  const tab = document.querySelector(`.tab[data-view="${view}"]`);
  let pill = tab.querySelector(".pill");
  if (!count) { if (pill) pill.remove(); return; }
  if (!pill) { pill = h("span", { class: `pill${neutral ? " neutral" : ""}` }); tab.append(pill); }
  pill.textContent = count;
  pill.title = neutral ? `${count} ricordi` : `${count} da controllare`;
}

function setupSettings() {
  const dialog = document.getElementById("settings-dialog");
  document.getElementById("settings-button").append(icon("settings"));
  document.getElementById("settings-button").addEventListener("click", () => {
    for (const radio of dialog.querySelectorAll('input[name="sensitivity"]')) radio.checked = Number(radio.value) === settings.sensitivity;
    document.getElementById("use-memory").checked = settings.useMemory;
    document.getElementById("image-quality").value = String(settings.imageSide);
    dialog.showModal();
  });
  dialog.addEventListener("close", () => {
    const checked = dialog.querySelector('input[name="sensitivity"]:checked');
    if (checked) settings.sensitivity = Number(checked.value);
    settings.useMemory = document.getElementById("use-memory").checked;
    settings.imageSide = Number(document.getElementById("image-quality").value);
    store("egeria-settings", settings);
  });
}

function setupTheme() {
  const button = document.getElementById("theme-button");
  const render = () => button.replaceChildren(icon(document.documentElement.dataset.theme === "dark" ? "sun" : "moon"));
  button.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("egeria-theme", next); } catch { /* niente */ }
    render();
  });
  render();
}

document.addEventListener("DOMContentLoaded", () => {
  setupTheme();
  setupSettings();
  for (const tab of document.querySelectorAll(".tab")) {
    const [iconName, label] = VIEWS[tab.dataset.view];
    tab.append(icon(iconName), h("span", { class: "tab-text", text: label }));
    tab.addEventListener("click", () => showView(tab.dataset.view));
  }

  // Cosa guardare
  for (const button of document.querySelectorAll("#mode-switch button")) button.addEventListener("click", () => setMode(button.dataset.mode));
  document.getElementById("input-text").addEventListener("input", markStale);
  const drop = document.getElementById("image-drop");
  const imageInput = document.getElementById("image-input");
  drop.querySelector(".dropzone-text").append(icon("image"), " Trascina qui un'immagine, incollala (Ctrl+V) oppure ",
    h("button", { class: "link-button", type: "button", onclick: () => imageInput.click() }, "scegli un file"));
  imageInput.addEventListener("change", () => { addImageFiles([...imageInput.files]); imageInput.value = ""; });
  drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("dragover"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
  drop.addEventListener("drop", (event) => { event.preventDefault(); drop.classList.remove("dragover"); addImageFiles([...event.dataTransfer.files].filter((f) => f.type.startsWith("image/"))); });
  document.addEventListener("paste", (event) => {
    if (document.getElementById("view-chiedi").hidden || S.mode !== "static") return;
    const files = [...(event.clipboardData ? event.clipboardData.files : [])].filter((f) => f.type.startsWith("image/"));
    if (files.length) { event.preventDefault(); addImageFiles(files); }
  });
  Live.init();

  // Esempi, domande salvate, ricomincia
  const examples = document.getElementById("examples");
  EXAMPLES.forEach((example, index) => examples.append(h("option", { value: String(index), text: example.label })));
  examples.addEventListener("change", () => { if (examples.value !== "") loadExample(EXAMPLES[Number(examples.value)]); examples.value = ""; });
  document.getElementById("reset-button").addEventListener("click", () => { if (!hasWork() || window.confirm("Cancellare testo, immagini e domande?")) resetAll(); });
  renderSavedSets();
  document.getElementById("saved-sets").addEventListener("change", (event) => {
    const name = event.target.value;
    event.target.value = "";
    if (!name) return;
    if (S.questions.some((q) => q.text.trim()) && !window.confirm(`Sostituire le domande attuali con «${name}»?`)) return;
    S.questions = savedSets()[name].map((q) => newQuestion({ ...q, options: [...q.options], levels: [...q.levels] }));
    markStale();
    renderQuestions();
  });
  document.getElementById("save-set").addEventListener("click", () => {
    if (!plainQuestions().length) { toast("Scrivi prima almeno una domanda.", "error"); return; }
    document.getElementById("set-name").value = "";
    document.getElementById("name-dialog").showModal();
  });
  document.getElementById("name-dialog").addEventListener("close", () => {
    const dialog = document.getElementById("name-dialog");
    const name = document.getElementById("set-name").value.trim();
    if (dialog.returnValue !== "save" || !name) return;
    const sets = savedSets();
    sets[name] = plainQuestions();
    store("egeria-domande", sets);
    renderSavedSets();
    toast(`Domande salvate come «${name}»`, "success");
  });

  // Domande e azioni
  const add = document.getElementById("add-question");
  add.append(icon("plus"), "Aggiungi una domanda");
  add.addEventListener("click", () => {
    const q = newQuestion();
    S.questions.push(q);
    markStale();
    renderQuestions();
    document.querySelector(`.question[data-uid="${q.uid}"] .question-text`).focus();
  });
  document.getElementById("ask-button").addEventListener("click", ask);
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !document.getElementById("view-chiedi").hidden) { event.preventDefault(); ask(); }
  });
  document.getElementById("teach-button").addEventListener("click", teach);

  // Storico e ricordi
  for (const button of document.querySelectorAll("#history-filter button")) button.addEventListener("click", () => { History.view = button.dataset.view; History.load(); });
  document.getElementById("memory-search-button").append(icon("search"), "Cerca");
  document.getElementById("memory-search-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const query = document.getElementById("memory-query").value.trim();
    if (query) Memories.search(query, document.getElementById("memory-search-button")); else Memories.load();
  });
  const imageSearch = document.getElementById("memory-image-button");
  imageSearch.append(icon("image"), "Cerca con un'immagine");
  imageSearch.addEventListener("click", () => document.getElementById("memory-image-input").click());
  document.getElementById("memory-image-input").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    event.target.value = "";
    if (!file) return;
    try {
      const upload = await api("POST", "/api/media", { data: await downscale(file, 1280) });
      await Memories.search([{ type: "text", text: "Immagine ricevuta:" }, { type: "image", path: upload.path }], imageSearch);
    } catch (error) { toast(error.message, "error"); }
  });
  document.getElementById("memory-show-all").addEventListener("click", () => { document.getElementById("memory-query").value = ""; Memories.load(); });

  renderQuestions();
  renderAskButton();
  renderTeach();
  const initial = location.hash.slice(1);
  showView(VIEWS[initial] ? initial : "chiedi");
  refreshInfo();
  setInterval(refreshInfo, 30000);
});
