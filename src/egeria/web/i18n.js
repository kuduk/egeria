"use strict";
/* Egeria · lingue dell'interfaccia (italiano e inglese).
   - testi statici di index.html: attributi data-i18n (textContent), data-i18n-placeholder,
     data-i18n-aria-label, data-i18n-title, applicati da applyStaticTexts();
   - testi dinamici di app.js: t("chiave", {nome: valore}), con {nome} sostituito e, se il valore
     della chiave è un oggetto {one, other}, la forma scelta in base a vars.count.
   Lingua iniziale: la preferenza salvata (localStorage "egeria-lang"), altrimenti quella del browser
   (it* → italiano, tutto il resto → inglese). Gli errori che arrivano dal server restano in italiano:
   in inglese si traducono solo quelli noti (serverError). */

const LANGS = ["it", "en"];

const I18N = {
  it: {
    // intestazione e navigazione
    "skip": "Vai al contenuto",
    "nav.label": "Sezioni",
    "nav.ask": "Chiedi",
    "nav.history": "Storico",
    "nav.memories": "Ricordi",
    "settings.open": "Impostazioni",
    "theme.toggle": "Tema chiaro o scuro",
    "lang.switch": "Passa all'inglese (English)",
    "lang.short": "EN",

    // Chiedi · cosa guardare
    "look.title": "Cosa guardare",
    "look.mode": "Modalità",
    "look.static": "Testo e immagini",
    "look.live": "Dal vivo",
    "look.textLabel": "Testo da analizzare",
    "look.textPlaceholder": "Scrivi o incolla qui un testo: un messaggio, un'email, una descrizione…",
    "look.drop": " Trascina qui un'immagine, incollala (Ctrl+V) oppure ",
    "look.chooseFile": "scegli un file",
    "look.images": "Immagini aggiunte",
    "look.image": "Immagine {n}",
    "look.removeImage": "Togli l'immagine {n}",
    "look.examples": "Prova un esempio",
    "look.choose": "Scegli…",
    "look.reset": "Ricomincia da zero",

    // dal vivo
    "live.preview": "Anteprima dal vivo",
    "live.placeholder": "Scegli da dove prendere le immagini",
    "live.source": "Sorgente",
    "live.webcam": "Telecamera",
    "live.screen": "Schermo",
    "live.file": "Video da file",
    "live.fps": "Quante immagini al secondo",
    "live.fps1": "1 al secondo",
    "live.fps2": "2 al secondo",
    "live.fps5": "5 al secondo",
    "live.fpsMax": "Il più possibile",
    "live.threshold": "Avvisami quando sono sicuro almeno al",
    "live.alerts": "Avvisi",
    "live.clear": "Svuota",
    "live.noAlerts": "Nessun avviso. Scegli in ogni domanda «Avvisami quando la risposta è…».",
    "live.alert": "Avviso",
    "live.alertImage": "Immagine al momento dell'avviso",
    "live.ignore": "Ignora",
    "live.saved": "Salvato",
    "live.alertNote": "Avviso: {question}",
    "live.start": "Avvia",
    "live.stop": "Ferma",
    "live.watching": "Sto guardando dal vivo…",
    "live.stopped": "Fermato.",
    "live.fpsStat": "{fps} immagini/s",
    "live.badVideo": "Scegli un file video (MP4, WebM, Ogg, MOV o MKV).",
    "live.noAccess": "Non riesco ad accedere: {error}",
    "live.halted": "Mi sono fermato: {error}",

    // Chiedi · cosa chiedere
    "ask.title": "Cosa chiedere",
    "ask.savedLabel": "Domande salvate",
    "ask.savedPlaceholder": "Domande salvate…",
    "ask.noSaved": "Nessuna domanda salvata",
    "ask.saveSet": "Salva queste domande",
    "ask.add": "Aggiungi una domanda",
    "ask.button": "Chiedi",
    "ask.looking": "Sto guardando…",
    "ask.done": "Fatto in {seconds} s",
    "ask.doneCheck": " · alcune risposte vanno controllate",
    "ask.stale": "Hai cambiato qualcosa: premi «Chiedi» per aggiornare le risposte.",
    "ask.nothing": "Scrivi un testo o aggiungi un'immagine da guardare.",

    // schede delle domande
    "kind.yesno": "Sì / No",
    "kind.choice": "Una tra più opzioni",
    "kind.scale": "Una scala (dal più basso al più alto)",
    "kind.short": "Una risposta breve (un nome, un numero, una data…)",
    "kind.estimate": "Una stima (quanti, quanto…)",
    "scale.low": "Basso",
    "scale.medium": "Medio",
    "scale.high": "Alto",
    "q.placeholder": "Scrivi la domanda, es. «Il cliente è arrabbiato?»",
    "q.label": "Domanda",
    "q.kind": "Tipo di risposta",
    "q.remove": "Togli questa domanda",
    "q.answer": "Risposta:",
    "q.addOption": "Aggiungi un'opzione e premi Invio",
    "q.levelsHint": "I livelli vanno dal più basso al più alto.",
    "q.addLevel": "Aggiungi un livello e premi Invio",
    "q.min": "Valore minimo",
    "q.max": "Valore massimo",
    "q.unitPlaceholder": "unità (es. euro)",
    "q.unit": "Unità di misura",
    "q.between": "Di solito tra",
    "q.and": "e",
    "q.estimateHint": "Per quantità non scritte: persone, età, distanze. Se il numero è scritto, come il totale di uno scontrino, usa «Una risposta breve».",
    "q.shortHint": "Il modello legge o risponde con una parola o un valore: un nome, un giorno, una data, un importo, un codice. Per i documenti conviene la qualità «Massima» nelle impostazioni.",
    "q.batch": "Calibra sullo storico",
    "q.batchHint": "Corregge la risposta con quelle date alla stessa domanda nei casi precedenti (da 50 casi). Non usarla per eventi rari, come un incendio: creerebbe falsi allarmi.",
    "q.alertWhen": "Avvisami quando la risposta è",
    "q.never": "mai",
    "q.removeChip": "Togli «{value}»",
    "q.errText": "Scrivi la domanda.",
    "q.errTwoOptions": "Aggiungi almeno due opzioni.",
    "q.errMaxOptions": "Al massimo 26 opzioni.",
    "q.errTwoLevels": "Servono almeno due livelli.",
    "q.errMaxLevels": "Al massimo 10 livelli.",
    "q.errRange": "Indica un intervallo valido: «da» deve essere minore di «a».",
    "q.errNone": "Aggiungi almeno una domanda.",

    // risposte
    "answer.yes": "Sì",
    "answer.no": "No",
    "answer.sure": "sicuro al {pct}",
    "answer.notSure": "Non sono sicuro",
    "answer.others": "Altre possibilità: {list}",
    "answer.about": "circa {value}",
    "answer.numberExtra": "Più probabile: {label} · quasi certamente tra {lo} e {hi}",
    "answer.numberExtraMean": "Media {value} · quasi certamente tra {lo} e {hi}",
    "answer.lessThan": "meno di {hi}{unit}",
    "answer.orMore": "{lo}{unit} o più",
    "answer.range": "da {lo} a {hi}{unit}",
    "answer.memory": "Nei casi simili che ricordo ({count}) avevi deciso: ",
    "answer.batchApplied": "Calibrata sullo storico ({count} casi precedenti).",
    "answer.batchWaiting": { one: "Non ancora calibrata sullo storico: {count} caso su {min}.", other: "Non ancora calibrata sullo storico: {count} casi su {min}." },
    "answer.same": "uguale",
    "answer.different": "diverso",
    "correct.button": "Correggi",
    "correct.done": "Corretto in: {label}",
    "correct.undo": "Annulla correzione",
    "correct.prompt": "La risposta giusta è:",
    "correct.label": "Risposta corretta",
    "correct.ok": "Ok",
    "correct.cancel": "Annulla",

    // salvataggio nei ricordi
    "teach.question": "Le risposte vanno bene?",
    "teach.saved": "Salvato nei ricordi",
    "teach.help": "Correggi quelle sbagliate con «Correggi», poi salva: il caso verrà ricordato e usato per i casi simili.",
    "teach.noteLabel": "Nota",
    "teach.notePlaceholder": "Nota facoltativa: cosa hai verificato, com'è andata…",
    "teach.update": "Aggiorna il ricordo",
    "teach.saveCorrections": { one: "Salva nei ricordi ({count} correzione)", other: "Salva nei ricordi ({count} correzioni)" },
    "teach.save": "Sì, salva nei ricordi",
    "teach.toast": "Salvato nei ricordi: lo userò per i casi simili.",
    "teach.remember": "Salva nei ricordi",

    // esempi e domande salvate
    "examples.replace": "Sostituire quello che hai scritto con l'esempio?",
    "examples.loadedLive": "Esempio caricato: scegli la telecamera, lo schermo o un video e premi «Avvia».",
    "examples.loaded": "Esempio caricato: premi «Chiedi».",
    "reset.confirm": "Cancellare testo, immagini e domande?",
    "sets.replace": "Sostituire le domande attuali con «{name}»?",
    "sets.empty": "Scrivi prima almeno una domanda.",
    "sets.saved": "Domande salvate come «{name}»",
    "sets.title": "Salva le domande",
    "sets.name": "Nome",
    "sets.namePlaceholder": "es. Messaggi dei clienti",
    "sets.cancel": "Annulla",
    "sets.save": "Salva",

    // storico
    "history.title": "Storico delle domande",
    "history.filter": "Filtro",
    "history.review": "Da controllare",
    "history.all": "Tutte",
    "history.help": "Le analisi «da controllare» hanno risposte incerte o diverse da quelle dei casi simili già ricordati. Aprile, correggi se serve e salvale nei ricordi.",
    "history.nothing": "Niente da controllare.",
    "history.none": "Non hai ancora fatto domande. Vai su «Chiedi».",
    "history.noText": "Immagine senza testo",
    "history.questions": { one: "{count} domanda", other: "{count} domande" },
    "history.images": { one: "{count} immagine", other: "{count} immagini" },
    "history.inMemory": "Nei ricordi",
    "history.uncertain": { one: "{count} incerta", other: "{count} incerte" },
    "history.disagree": { one: "{count} diversa dai ricordi", other: "{count} diverse dai ricordi" },
    "history.opened": "Analisi del {date}",

    // ricordi
    "memory.title": "Ricordi",
    "memory.help": "Sono i casi che hai salvato. Quando fai una domanda su qualcosa di simile, il modello ti mostra cosa avevi deciso.",
    "memory.searchLabel": "Cerca tra i ricordi",
    "memory.searchPlaceholder": "Descrivi qualcosa: trovo i ricordi più simili",
    "memory.search": "Cerca",
    "memory.searchImage": "Cerca con un'immagine",
    "memory.showAll": "Mostra tutti",
    "memory.count": { one: "{count} ricordo", other: "{count} ricordi" },
    "memory.none": "Nessun ricordo ancora. Fai una domanda e premi «Salva nei ricordi».",
    "memory.noneShort": "Nessun ricordo ancora.",
    "memory.similar": "I più simili",
    "memory.similarity": "somiglianza {pct}",
    "memory.forget": "Dimentica",
    "memory.forgetLabel": "Dimentica questo ricordo",
    "memory.forgetConfirm": "Dimenticare questo ricordo? Non si potrà recuperare.",
    "memory.forgotten": "Ricordo dimenticato",
    "memory.liveImage": "Immagine dal vivo:",

    // stato del server
    "info.ready": "Pronto",
    "info.modelDown": "Modello non raggiungibile",
    "info.modelDownTitle": "{error}. Storico e ricordi restano consultabili.",
    "info.model": "Modello: {model} · {device} · {memories}",
    "info.suspended": { one: " · {count} sospeso (immagine non più leggibile, vedi memories-sospese.jsonl)", other: " · {count} sospesi (immagini non più leggibili, vedi memories-sospese.jsonl)" },
    "info.modelDownSettings": "Il modello non risponde su {url}: avvialo con «egeria model-server». {memories}",
    "info.webDown": "Server web non raggiungibile",
    "info.toReview": "{count} da controllare",

    // impostazioni
    "settings.title": "Impostazioni",
    "settings.language": "Lingua · Language",
    "settings.sensitivity": "Quando segnalare una risposta come incerta",
    "settings.often": "Spesso: più prudente, più casi da controllare",
    "settings.normal": "Normale",
    "settings.rarely": "Raramente: meno casi da controllare",
    "settings.useMemory": "Usa i ricordi per mostrare cosa avevi deciso nei casi simili",
    "settings.quality": "Qualità delle immagini",
    "settings.fast": "Veloce",
    "settings.normalQuality": "Normale",
    "settings.detailed": "Dettagliata",
    "settings.max": "Massima: per leggere testi e documenti (più lenta)",
    "settings.done": "Fatto",

    // tempo ed errori
    "time.now": "adesso",
    "time.minutes": { one: "{count} minuto fa", other: "{count} minuti fa" },
    "time.hours": { one: "{count} ora fa", other: "{count} ore fa" },
    "error.server": "Errore del server ({status})",
    "error.image": "immagine non leggibile",
    "error.imageName": "Immagine",
  },

  en: {
    "skip": "Skip to content",
    "nav.label": "Sections",
    "nav.ask": "Ask",
    "nav.history": "History",
    "nav.memories": "Memories",
    "settings.open": "Settings",
    "theme.toggle": "Light or dark theme",
    "lang.switch": "Switch to Italian (Italiano)",
    "lang.short": "IT",

    "look.title": "What to look at",
    "look.mode": "Mode",
    "look.static": "Text and images",
    "look.live": "Live",
    "look.textLabel": "Text to analyze",
    "look.textPlaceholder": "Write or paste some text here: a message, an email, a description…",
    "look.drop": " Drag an image here, paste it (Ctrl+V) or ",
    "look.chooseFile": "choose a file",
    "look.images": "Added images",
    "look.image": "Image {n}",
    "look.removeImage": "Remove image {n}",
    "look.examples": "Try an example",
    "look.choose": "Choose…",
    "look.reset": "Start over",

    "live.preview": "Live preview",
    "live.placeholder": "Choose where to take the images from",
    "live.source": "Source",
    "live.webcam": "Camera",
    "live.screen": "Screen",
    "live.file": "Video from file",
    "live.fps": "Frames per second",
    "live.fps1": "1 per second",
    "live.fps2": "2 per second",
    "live.fps5": "5 per second",
    "live.fpsMax": "As many as possible",
    "live.threshold": "Alert me when my confidence is at least",
    "live.alerts": "Alerts",
    "live.clear": "Clear",
    "live.noAlerts": "No alerts. In each question, choose “Alert me when the answer is…”.",
    "live.alert": "Alert",
    "live.alertImage": "Image at the time of the alert",
    "live.ignore": "Ignore",
    "live.saved": "Saved",
    "live.alertNote": "Alert: {question}",
    "live.start": "Start",
    "live.stop": "Stop",
    "live.watching": "Watching live…",
    "live.stopped": "Stopped.",
    "live.fpsStat": "{fps} frames/s",
    "live.badVideo": "Choose a video file (MP4, WebM, Ogg, MOV or MKV).",
    "live.noAccess": "Cannot access it: {error}",
    "live.halted": "I stopped: {error}",

    "ask.title": "What to ask",
    "ask.savedLabel": "Saved questions",
    "ask.savedPlaceholder": "Saved questions…",
    "ask.noSaved": "No saved questions",
    "ask.saveSet": "Save these questions",
    "ask.add": "Add a question",
    "ask.button": "Ask",
    "ask.looking": "Looking…",
    "ask.done": "Done in {seconds} s",
    "ask.doneCheck": " · some answers need checking",
    "ask.stale": "You changed something: press “Ask” to update the answers.",
    "ask.nothing": "Write some text or add an image to look at.",

    "kind.yesno": "Yes / No",
    "kind.choice": "One of several options",
    "kind.scale": "A scale (from lowest to highest)",
    "kind.short": "A short answer (a name, a number, a date…)",
    "kind.estimate": "An estimate (how many, how much…)",
    "scale.low": "Low",
    "scale.medium": "Medium",
    "scale.high": "High",
    "q.placeholder": "Write the question, e.g. “Is the customer angry?”",
    "q.label": "Question",
    "q.kind": "Answer type",
    "q.remove": "Remove this question",
    "q.answer": "Answer:",
    "q.addOption": "Add an option and press Enter",
    "q.levelsHint": "Levels go from lowest to highest.",
    "q.addLevel": "Add a level and press Enter",
    "q.min": "Minimum value",
    "q.max": "Maximum value",
    "q.unitPlaceholder": "unit (e.g. euro)",
    "q.unit": "Unit of measure",
    "q.between": "Usually between",
    "q.and": "and",
    "q.estimateHint": "For quantities that are not written down: people, age, distances. If the number is written, such as a receipt total, use “A short answer”.",
    "q.shortHint": "The model reads or answers with a word or a value: a name, a day, a date, an amount, a code. For documents, choose “Maximum” image quality in the settings.",
    "q.batch": "Calibrate on the history",
    "q.batchHint": "Corrects the answer using those given to the same question in earlier cases (from 50 cases). Do not use it for rare events, such as a fire: it would create false alarms.",
    "q.alertWhen": "Alert me when the answer is",
    "q.never": "never",
    "q.removeChip": "Remove “{value}”",
    "q.errText": "Write the question.",
    "q.errTwoOptions": "Add at least two options.",
    "q.errMaxOptions": "At most 26 options.",
    "q.errTwoLevels": "At least two levels are needed.",
    "q.errMaxLevels": "At most 10 levels.",
    "q.errRange": "Enter a valid range: the first value must be lower than the second.",
    "q.errNone": "Add at least one question.",

    "answer.yes": "Yes",
    "answer.no": "No",
    "answer.sure": "{pct} sure",
    "answer.notSure": "Not sure",
    "answer.others": "Other possibilities: {list}",
    "answer.about": "about {value}",
    "answer.numberExtra": "Most likely: {label} · almost certainly between {lo} and {hi}",
    "answer.numberExtraMean": "Mean {value} · almost certainly between {lo} and {hi}",
    "answer.lessThan": "less than {hi}{unit}",
    "answer.orMore": "{lo}{unit} or more",
    "answer.range": "from {lo} to {hi}{unit}",
    "answer.memory": "In the similar cases I remember ({count}) you had decided: ",
    "answer.batchApplied": "Calibrated on the history ({count} earlier cases).",
    "answer.batchWaiting": "Not calibrated on the history yet: {count} of {min} cases.",
    "answer.same": "same",
    "answer.different": "different",
    "correct.button": "Correct",
    "correct.done": "Corrected to: {label}",
    "correct.undo": "Undo correction",
    "correct.prompt": "The right answer is:",
    "correct.label": "Correct answer",
    "correct.ok": "Ok",
    "correct.cancel": "Cancel",

    "teach.question": "Are the answers right?",
    "teach.saved": "Saved to memories",
    "teach.help": "Fix the wrong ones with “Correct”, then save: the case will be remembered and used for similar cases.",
    "teach.noteLabel": "Note",
    "teach.notePlaceholder": "Optional note: what you checked, how it went…",
    "teach.update": "Update the memory",
    "teach.saveCorrections": { one: "Save to memories ({count} correction)", other: "Save to memories ({count} corrections)" },
    "teach.save": "Yes, save to memories",
    "teach.toast": "Saved to memories: I will use it for similar cases.",
    "teach.remember": "Save to memories",

    "examples.replace": "Replace what you have written with the example?",
    "examples.loadedLive": "Example loaded: choose the camera, the screen or a video and press “Start”.",
    "examples.loaded": "Example loaded: press “Ask”.",
    "reset.confirm": "Clear text, images and questions?",
    "sets.replace": "Replace the current questions with “{name}”?",
    "sets.empty": "Write at least one question first.",
    "sets.saved": "Questions saved as “{name}”",
    "sets.title": "Save the questions",
    "sets.name": "Name",
    "sets.namePlaceholder": "e.g. Customer messages",
    "sets.cancel": "Cancel",
    "sets.save": "Save",

    "history.title": "Question history",
    "history.filter": "Filter",
    "history.review": "To review",
    "history.all": "All",
    "history.help": "Analyses “to review” have answers that are uncertain or that differ from those of similar cases already remembered. Open them, correct them if needed and save them to memories.",
    "history.nothing": "Nothing to review.",
    "history.none": "You have not asked anything yet. Go to “Ask”.",
    "history.noText": "Image without text",
    "history.questions": { one: "{count} question", other: "{count} questions" },
    "history.images": { one: "{count} image", other: "{count} images" },
    "history.inMemory": "In memories",
    "history.uncertain": { one: "{count} uncertain", other: "{count} uncertain" },
    "history.disagree": { one: "{count} differs from memories", other: "{count} differ from memories" },
    "history.opened": "Analysis of {date}",

    "memory.title": "Memories",
    "memory.help": "These are the cases you have saved. When you ask about something similar, the model shows you what you had decided.",
    "memory.searchLabel": "Search memories",
    "memory.searchPlaceholder": "Describe something: I will find the most similar memories",
    "memory.search": "Search",
    "memory.searchImage": "Search with an image",
    "memory.showAll": "Show all",
    "memory.count": { one: "{count} memory", other: "{count} memories" },
    "memory.none": "No memories yet. Ask a question and press “Save to memories”.",
    "memory.noneShort": "No memories yet.",
    "memory.similar": "Most similar",
    "memory.similarity": "similarity {pct}",
    "memory.forget": "Forget",
    "memory.forgetLabel": "Forget this memory",
    "memory.forgetConfirm": "Forget this memory? It cannot be recovered.",
    "memory.forgotten": "Memory forgotten",
    "memory.liveImage": "Live image:",

    "info.ready": "Ready",
    "info.modelDown": "Model unreachable",
    "info.modelDownTitle": "{error}. History and memories are still available.",
    "info.model": "Model: {model} · {device} · {memories}",
    "info.suspended": { one: " · {count} suspended (image no longer readable, see memories-sospese.jsonl)", other: " · {count} suspended (images no longer readable, see memories-sospese.jsonl)" },
    "info.modelDownSettings": "The model does not respond at {url}: start it with “egeria model-server”. {memories}",
    "info.webDown": "Web server unreachable",
    "info.toReview": "{count} to review",

    "settings.title": "Settings",
    "settings.language": "Language · Lingua",
    "settings.sensitivity": "When to flag an answer as uncertain",
    "settings.often": "Often: more cautious, more cases to review",
    "settings.normal": "Normal",
    "settings.rarely": "Rarely: fewer cases to review",
    "settings.useMemory": "Use memories to show what you had decided in similar cases",
    "settings.quality": "Image quality",
    "settings.fast": "Fast",
    "settings.normalQuality": "Normal",
    "settings.detailed": "Detailed",
    "settings.max": "Maximum: for reading text and documents (slower)",
    "settings.done": "Done",

    "time.now": "just now",
    "time.minutes": { one: "{count} minute ago", other: "{count} minutes ago" },
    "time.hours": { one: "{count} hour ago", other: "{count} hours ago" },
    "error.server": "Server error ({status})",
    "error.image": "unreadable image",
    "error.imageName": "Image",

    // errori noti del server (che li manda in italiano), tradotti solo qui
    "server.unreachable": "model server unreachable at {url} (start it with: egeria model-server)",
    "server.token": "the model server rejects the token (--model-token)",
    "server.modelError": "model server error ({status}): {text}",
    "server.caseNotFound": "case not found",
    "server.memoryNotFound": "memory not found",
    "server.badImage": "invalid image",
    "server.noDecisions": "confirmed decisions are required",
    "server.memoryExists": "a memory {id} already exists",
  },
};

/** Errori del server riconoscibili con certezza: [espressione, chiave, nomi dei gruppi]. Gli altri restano come arrivano. */
const SERVER_ERRORS = [
  [/^server del modello non raggiungibile su (\S+) \(avvialo con: egeria model-server\)$/, "server.unreachable", ["url"]],
  [/^il server del modello rifiuta il token \(--model-token\)$/, "server.token", []],
  [/^errore del server del modello \((\d+)\): ([\s\S]*)$/, "server.modelError", ["status", "text"]],
  [/^caso non trovato$/, "server.caseNotFound", []],
  [/^ricordo non trovato$/, "server.memoryNotFound", []],
  [/^immagine non valida$/, "server.badImage", []],
  [/^servono le decisioni confermate$/, "server.noDecisions", []],
  [/^esiste già un ricordo (.+)$/, "server.memoryExists", ["id"]],
];

function detectLanguage() {
  try {
    const saved = localStorage.getItem("egeria-lang");
    if (LANGS.includes(saved)) return saved;
  } catch { /* archiviazione non disponibile */ }
  const preferred = (navigator.languages && navigator.languages[0]) || navigator.language || "";
  return preferred.toLowerCase().startsWith("it") ? "it" : "en";
}

let LANG = detectLanguage();
document.documentElement.lang = LANG;

/** Locale per numeri e date: quella del browser se è della stessa lingua (es. en-GB), altrimenti una di default. */
function locale(lang = LANG) {
  const own = (navigator.languages || [navigator.language || ""]).find((l) => l && l.toLowerCase().startsWith(lang));
  return own || (lang === "it" ? "it-IT" : "en-US");
}

function tIn(lang, key, vars = {}) {
  const dict = I18N[lang] || I18N.it;
  let entry = Object.prototype.hasOwnProperty.call(dict, key) ? dict[key] : I18N.it[key];
  if (entry === undefined) { console.warn(`i18n: chiave mancante «${key}»`); return key; }
  if (typeof entry === "object") {
    let form = "other";
    try { form = new Intl.PluralRules(lang).select(Number(vars.count)); } catch { form = Number(vars.count) === 1 ? "one" : "other"; }
    entry = entry[form] !== undefined ? entry[form] : entry.other;
  }
  return entry.replace(/\{(\w+)\}/g, (match, name) => (vars[name] !== undefined && vars[name] !== null ? String(vars[name]) : match));
}

/** Testo nella lingua corrente, con {nome} sostituito da vars.nome. */
function t(key, vars) { return tIn(LANG, key, vars); }

/** Messaggio d'errore del server: in inglese traduce solo quelli noti, gli altri restano in italiano. */
function serverError(message) {
  const text = String(message);
  if (LANG === "it") return text;
  for (const [pattern, key, names] of SERVER_ERRORS) {
    const match = pattern.exec(text);
    if (match) return t(key, Object.fromEntries(names.map((name, i) => [name, match[i + 1]])));
  }
  return text;
}

/** Applica i testi statici (attributi data-i18n*) dentro root. */
function applyStaticTexts(root = document) {
  for (const node of root.querySelectorAll("[data-i18n]")) node.textContent = t(node.dataset.i18n);
  for (const node of root.querySelectorAll("[data-i18n-placeholder]")) node.setAttribute("placeholder", t(node.dataset.i18nPlaceholder));
  for (const node of root.querySelectorAll("[data-i18n-aria-label]")) node.setAttribute("aria-label", t(node.dataset.i18nAriaLabel));
  for (const node of root.querySelectorAll("[data-i18n-title]")) node.setAttribute("title", t(node.dataset.i18nTitle));
}
