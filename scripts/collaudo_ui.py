"""Collaudo dell'interfaccia in un browser headless (Playwright): percorso completo con screenshot, in italiano e in inglese.

    uv pip install --python .venv/bin/python playwright && .venv/bin/python -m playwright install chromium
    .venv/bin/egeria model-server                                                          # il modello (porta 8100)
    .venv/bin/egeria serve --memory runs/ui-test/memoria --data runs/ui-test --port 8001   # istanza web di prova
    .venv/bin/python scripts/collaudo_ui.py http://127.0.0.1:8001 runs/ui-test/screenshot

Passaggio in italiano (browser con lingua it-IT): controlla che la pagina si apra dal vivo, passa a testo e
immagini, scrive due domande a mano, chiede, corregge e salva nei ricordi, prova l'esempio con la foto, la modalità dal vivo, lo storico, i ricordi e il tema.
Passaggio in inglese: parte in italiano, passa all'inglese con il pulsante in alto senza perdere il lavoro,
chiede, torna all'italiano e di nuovo all'inglese (le risposte restano), corregge e salva, prova le impostazioni
(lingua), un esempio, la modalità dal vivo, lo storico e i ricordi. In ogni pagina cerca parole dell'interfaccia
rimaste nell'altra lingua (esclusi i testi scritti dall'utente) e controlla che niente esca dalla pagina a
1440 e a 390 px di larghezza. Controlla anche la lingua scelta dal browser (en-US → inglese).
Stampa gli errori della console del browser e i problemi trovati (uscita 1 se ce ne sono).
Usa un'istanza separata: crea casi e ricordi di prova. Sul modello in CPU le risposte sono lente: attese lunghe.
"""
import os
import re
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/ui-test/screenshot"
os.makedirs(OUT, exist_ok=True)
ANSWER_TIMEOUT = 300_000  # ms: il modello in CPU impiega secondi per domanda
errors = []    # console del browser
problems = []  # controlli falliti


def check(condition, message):
    if not condition:
        problems.append(message)
    return condition


# Parole dell'interfaccia che non devono comparire nell'altra lingua (confronto per parola intera, maiuscole comprese).
ITALIAN_WORDS = [
    "Chiedi", "Storico", "Ricordi", "ricordi", "Cosa guardare", "Cosa chiedere", "Aggiungi", "Non sono sicuro", "Salva",
    "Impostazioni", "Sì", "sicuro", "Risposta", "Domanda", "Domande", "domanda", "domande", "Correggi", "Dimentica", "Cerca",
    "Prova un esempio", "Ricomincia", "Scegli", "Avviso", "Avvisi", "Avvisami", "Pronto", "Modello", "Testo e immagini",
    "Dal vivo", "Immagine", "Immagini", "immagine", "immagini", "Basso", "Medio", "Alto", "Tutte", "controllare", "adesso",
    "minuti fa", "ore fa", "Fatto", "Annulla", "Nessun", "Nessuna", "Tema", "Svuota", "Telecamera", "Schermo",
    "Mostra tutti", "somiglianza", "uguale", "diverso", "Nota", "Sezioni", "Vai al contenuto", "Sto guardando", "Fermato",
    "circa", "Più probabile", "meno di", "o più", "Lingua", "Qualità", "Veloce", "Normale", "Spesso", "Raramente",
]
ENGLISH_WORDS = [
    "Ask", "History", "Memories", "What to look at", "What to ask", "Add a question", "Not sure", "Save to memories",
    "Settings", "Yes", "sure", "Answer", "Question", "Correct", "Forget", "Search", "Try an example", "Start over",
    "Choose", "Alert", "Alerts", "Ready", "Text and images", "Live", "Image", "Low", "Medium", "High", "All", "To review",
    "just now", "ago", "Done", "Cancel", "No alerts", "Theme", "Clear", "Camera", "Screen", "Show all", "similarity",
    "same", "different", "Sections", "Skip to content", "Looking", "Stopped", "about", "Most likely", "less than",
    "or more", "Language", "Quality",
]
# Testi scritti dall'utente (o salvati da lui): non si controllano. Il selettore della lingua è bilingue apposta.
USER_CONTENT = ", ".join([
    ".history-text", ".memory-text", ".memory-note", ".qa", "#language-field",
    '#saved-sets option:not([value=""])',
])

VISIBLE_TEXTS = """(skip) => {
  const out = [];
  const visible = (el) => el && (el.checkVisibility ? el.checkVisibility({ visibilityProperty: true }) : !!el.offsetParent);
  const skipped = (el) => !!el.closest(skip);
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode, el = node.parentElement;
    const text = node.textContent.trim();
    if (!text || !el || el.closest("script, style, select") || skipped(el) || !visible(el)) continue;
    out.push(text);
  }
  for (const el of document.querySelectorAll("[aria-label], [placeholder], [title], img[alt]")) {
    if (skipped(el) || !visible(el)) continue;
    for (const name of ["aria-label", "placeholder", "title", "alt"]) { const value = el.getAttribute(name); if (value) out.push(value); }
  }
  for (const select of document.querySelectorAll("select")) {
    if (skipped(select) || !visible(select)) continue;
    for (const option of select.options) if (!skipped(option)) out.push(option.textContent.trim());
  }
  return out;
}"""

OVERFLOW = """() => {
  const width = document.documentElement.clientWidth;
  const out = [];
  for (const el of document.querySelectorAll("body *")) {
    if (!el.checkVisibility || !el.checkVisibility() || el.closest(".visually-hidden, .skip-link, .toasts")) continue;
    const box = el.getBoundingClientRect();
    if (box.width && box.right > width + 1) out.push(`${el.tagName.toLowerCase()}${el.id ? "#" + el.id : ""}.${el.className} → ${Math.round(box.right)}px`);
  }
  return { scroll: document.documentElement.scrollWidth, width, out: out.slice(0, 8) };
}"""


def leftovers(page, words, where):
    """Parole dell'altra lingua nei testi visibili (e segnaposto {nome} non sostituiti)."""
    found = []
    for text in page.evaluate(VISIBLE_TEXTS, USER_CONTENT):
        for word in words:
            if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text):
                found.append(f"{where}: «{word}» in «{text[:90]}»")
        if re.search(r"\{\w+\}", text):
            found.append(f"{where}: segnaposto non sostituito in «{text[:90]}»")
    problems.extend(found)
    return found


def layout(page, name, widths=(1440, 390)):
    """Screenshot a più larghezze; segnala ciò che esce dalla pagina in orizzontale."""
    for width in widths:
        page.set_viewport_size({"width": width, "height": 1000 if width > 600 else 844})
        page.evaluate("window.scrollTo(0, 0)")  # la barra in alto è sticky: nello screenshot resta in cima
        page.wait_for_timeout(300)
        result = page.evaluate(OVERFLOW)
        check(result["scroll"] <= result["width"] and not result["out"],
              f"{name} a {width}px: esce dalla pagina (scrollWidth {result['scroll']}): {result['out']}")
        page.screenshot(path=f"{OUT}/{name}_{width}.png", full_page=True)
    page.set_viewport_size({"width": 1440, "height": 1000})


def new_page(browser, locale):
    context = browser.new_context(locale=locale, viewport={"width": 1440, "height": 1000})
    page = context.new_page()
    page.set_default_timeout(60_000)
    page.on("console", lambda m: errors.append(f"console {m.type}: {m.text}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    return context, page


def text_of(page, selector):
    return page.locator(selector).first.inner_text().strip()


with sync_playwright() as p:
    browser = p.chromium.launch()

    # ------------------------------------------------------------------ italiano (browser in italiano)
    context, page = new_page(browser, "it-IT")
    page.goto(URL + "/")
    page.wait_for_selector(".question")
    check(page.evaluate("document.documentElement.lang") == "it", "it-IT: la pagina non parte in italiano")
    # la pagina si apre dal vivo: casella «Dal vivo» selezionata, anteprima visibile, testo nascosto
    check(page.get_attribute("#mode-switch [data-mode='live']", "aria-checked") == "true"
          and page.is_visible("#live-input") and not page.is_visible("#input-text"), "it: la pagina non si apre dal vivo")
    check(page.locator(".question-check").count() == 0, "it: casella della calibrazione sullo storico visibile dal vivo")
    page.screenshot(path=f"{OUT}/01_vuota.png", full_page=True)
    # scrivo una domanda a mano, senza esempi, in modalità testo e immagini
    page.click("#mode-switch [data-mode='static']")
    page.fill("#input-text", "Salve, il mio ordine 4471 non è ancora arrivato e il tracking è fermo da una settimana. Vorrei il rimborso.")
    page.fill(".question-text", "Il cliente chiede un rimborso?")
    page.click("#add-question")
    page.locator(".question-text").nth(1).fill("Qual è il problema?")
    page.locator(".question select").nth(1).select_option("choice")
    chip = page.locator(".question").nth(1).locator(".chip-input")
    for option in ["Consegna in ritardo", "Prodotto difettoso", "Fatturazione"]:
        chip.fill(option); chip.press("Enter")
        chip = page.locator(".question").nth(1).locator(".chip-input")
    # calibrazione sullo storico sulla seconda domanda: con uno storico vuoto la risposta dice quanti casi mancano
    page.locator(".question").nth(1).locator(".question-check input").check()
    page.screenshot(path=f"{OUT}/02_domande_scritte.png", full_page=True)
    page.click("#ask-button")
    page.wait_for_selector(".answer", timeout=ANSWER_TIMEOUT)
    page.wait_for_timeout(500)
    page.screenshot(path=f"{OUT}/03_risposte.png", full_page=True)
    check(text_of(page, ".answer-sure").startswith("sicuro al"), "it: manca «sicuro al» nella risposta")
    check(page.locator(".question").nth(0).locator(".batch-note").count() == 0,
          "it: nota della calibrazione sullo storico su una domanda senza la casella")
    check("Non ancora calibrata sullo storico" in text_of(page, ".question:nth-child(2) .batch-note"),
          "it: manca la nota «Non ancora calibrata sullo storico» sulla domanda con la casella")
    # correggo la prima risposta e salvo nei ricordi
    page.locator(".answer").first.get_by_role("button", name="Correggi").click()
    page.locator(".correction select").first.select_option(index=1)
    page.locator(".correction").first.get_by_role("button", name="Ok").click()
    page.click("#teach-button")
    page.wait_for_selector(".teach.saved", timeout=60000)
    page.screenshot(path=f"{OUT}/04_salvato.png", full_page=True)
    leftovers(page, ENGLISH_WORDS, "it · Chiedi")
    layout(page, "it_chiedi")
    # esempio con foto
    page.once("dialog", lambda d: d.accept())
    page.select_option("#examples", label="Foto di un incidente")
    page.click("#ask-button")
    page.wait_for_function("document.querySelectorAll('.answer').length >= 3", timeout=ANSWER_TIMEOUT)
    page.wait_for_timeout(500)
    page.screenshot(path=f"{OUT}/05_foto.png", full_page=True)
    # dal vivo
    page.click("#mode-switch [data-mode='live']")
    page.screenshot(path=f"{OUT}/06_dal_vivo.png", full_page=True)
    leftovers(page, ENGLISH_WORDS, "it · Dal vivo")
    # storico e ricordi
    page.click(".tab[data-view='storico']")
    page.click("#history-filter [data-view='tutti']")
    page.wait_for_timeout(800)
    page.screenshot(path=f"{OUT}/07_storico.png", full_page=True)
    leftovers(page, ENGLISH_WORDS, "it · Storico")
    page.click(".tab[data-view='ricordi']")
    page.wait_for_timeout(800)
    page.screenshot(path=f"{OUT}/08_ricordi.png", full_page=True)
    leftovers(page, ENGLISH_WORDS, "it · Ricordi")
    # tema chiaro
    page.click("#theme-button")
    page.click(".tab[data-view='chiedi']")
    page.wait_for_timeout(300)
    page.screenshot(path=f"{OUT}/09_chiaro.png", full_page=True)
    context.close()

    # ------------------------------------------------------------------ lingua dal browser
    context, page = new_page(browser, "en-US")
    page.goto(URL + "/")
    page.wait_for_selector(".question")
    check(page.evaluate("document.documentElement.lang") == "en", "en-US: la pagina non parte in inglese")
    check(text_of(page, ".tab[data-view='chiedi'] .tab-text") == "Ask", "en-US: la voce «Ask» non è in inglese")
    context.close()

    # ------------------------------------------------------------------ inglese (parte in italiano e cambia lingua)
    context, page = new_page(browser, "it-IT")
    dialogs = []
    page.on("dialog", lambda d: (dialogs.append(d.message), d.accept()))
    page.goto(URL + "/")
    page.wait_for_selector(".question")
    state = "Hello, my order 4471 has not arrived yet and the tracking has been stuck for a week. I would like a refund."
    page.click("#mode-switch [data-mode='static']")
    page.fill("#input-text", state)
    page.fill(".question-text", "Is the customer asking for a refund?")
    button = page.locator("#lang-button")
    check(button.inner_text().strip() == "EN" and button.get_attribute("aria-label") == "Passa all'inglese (English)",
          "it: il pulsante della lingua non mostra «EN» con la sua etichetta")
    button.click()
    page.wait_for_timeout(200)
    check(page.evaluate("document.documentElement.lang") == "en", "en: <html lang> non è «en»")
    check(button.inner_text().strip() == "IT" and button.get_attribute("aria-label") == "Switch to Italian (Italiano)",
          "en: il pulsante della lingua non mostra «IT» con la sua etichetta")
    check(page.evaluate("document.activeElement.id") == "lang-button", "en: il focus si è perso cambiando lingua")
    expected = {
        ".tab[data-view='chiedi'] .tab-text": "Ask", ".tab[data-view='storico'] .tab-text": "History",
        ".tab[data-view='ricordi'] .tab-text": "Memories", "#look-title": "What to look at", "#ask-title": "What to ask",
        "#ask-button": "Ask", "#add-question": "Add a question", "#reset-button": "Start over",
        "#mode-switch [data-mode='static']": "Text and images", "#mode-switch [data-mode='live']": "Live",
    }
    for selector, text in expected.items():
        check(text_of(page, selector) == text, f"en: {selector} è «{text_of(page, selector)}» invece di «{text}»")
    check(page.input_value("#input-text") == state, "en: il testo scritto si è perso cambiando lingua")
    check(page.input_value(".question-text") == "Is the customer asking for a refund?", "en: la domanda si è persa cambiando lingua")
    check(page.get_attribute("#input-text", "placeholder").startswith("Write or paste"), "en: segnaposto del testo non tradotto")
    page.screenshot(path=f"{OUT}/en_01_cambiata.png", full_page=True)
    page.click("#add-question")
    page.locator(".question-text").nth(1).fill("What is the problem?")
    page.locator(".question select").nth(1).select_option("choice")
    chip = page.locator(".question").nth(1).locator(".chip-input")
    check(chip.get_attribute("placeholder") == "Add an option and press Enter", "en: segnaposto delle opzioni non tradotto")
    for option in ["Late delivery", "Defective product", "Billing"]:
        chip.fill(option); chip.press("Enter")
        chip = page.locator(".question").nth(1).locator(".chip-input")
    page.click("#ask-button")
    page.wait_for_selector(".answer", timeout=ANSWER_TIMEOUT)
    page.wait_for_timeout(500)
    check(re.fullmatch(r"\d+% sure", text_of(page, ".answer-sure")) is not None, f"en: «{text_of(page, '.answer-sure')}» invece di «N% sure»")
    check(text_of(page, ".answer-value") in ("Yes", "No"), f"en: risposta sì/no «{text_of(page, '.answer-value')}»")
    check(text_of(page, "#ask-status").startswith("Done in"), f"en: stato «{text_of(page, '#ask-status')}»")
    page.screenshot(path=f"{OUT}/en_02_risposte.png", full_page=True)
    # avanti e indietro tra le lingue: le risposte restano, tradotte
    answers = page.locator(".answer").count()
    button.click()
    page.wait_for_timeout(200)
    check(page.locator(".answer").count() == answers, "it: le risposte si sono perse cambiando lingua")
    check(text_of(page, ".answer-sure").startswith("sicuro al") and text_of(page, ".answer-value") in ("Sì", "No"),
          "it: le risposte non sono tornate in italiano")
    check(page.input_value("#input-text") == state, "it: il testo si è perso cambiando lingua")
    button.click()
    page.wait_for_timeout(200)
    check(page.locator(".answer").count() == answers and text_of(page, ".answer-value") in ("Yes", "No"),
          "en: le risposte non sono tornate in inglese")
    # correggo e salvo nei ricordi
    page.locator(".answer").first.get_by_role("button", name="Correct", exact=True).click()
    page.locator(".correction select").first.select_option(index=1)
    page.locator(".correction").first.get_by_role("button", name="Ok", exact=True).click()
    check(text_of(page, "#teach-button") == "Save to memories (1 correction)", f"en: pulsante «{text_of(page, '#teach-button')}»")
    page.click("#teach-button")
    page.wait_for_selector(".teach.saved", timeout=60000)
    check(text_of(page, ".teach-title") == "Saved to memories", "en: «Saved to memories» mancante")
    page.screenshot(path=f"{OUT}/en_03_salvato.png", full_page=True)
    leftovers(page, ITALIAN_WORDS, "en · Ask")
    layout(page, "en_ask")
    # impostazioni: il selettore della lingua cambia subito tutto, il focus resta sul selettore
    page.click("#settings-button")
    page.wait_for_selector("#settings-dialog[open]")
    check(text_of(page, "#settings-title") == "Settings", "en: titolo delle impostazioni non tradotto")
    leftovers(page, ITALIAN_WORDS, "en · Settings")
    page.screenshot(path=f"{OUT}/en_04_impostazioni.png")
    page.select_option("#language", "it")
    check(text_of(page, "#settings-title") == "Impostazioni" and text_of(page, ".tab[data-view='chiedi'] .tab-text") == "Chiedi",
          "it: il selettore della lingua non cambia l'interfaccia")
    page.select_option("#language", "en")
    check(page.evaluate("document.activeElement.id") == "language", "en: il focus ha lasciato il selettore della lingua")
    page.get_by_role("button", name="Done", exact=True).click()
    # dialogo per salvare le domande
    page.click("#save-set")
    page.wait_for_selector("#name-dialog[open]")
    leftovers(page, ITALIAN_WORDS, "en · Save the questions")
    page.get_by_role("button", name="Cancel", exact=True).click()
    # esempio in inglese (le immagini sono le stesse)
    page.select_option("#examples", label="Photo of an accident")
    page.wait_for_timeout(300)
    check(dialogs and dialogs[-1] == "Replace what you have written with the example?", f"en: conferma «{dialogs[-1] if dialogs else ''}»")
    check(page.input_value(".question-text") == "Is the vehicle damaged?", "en: esempio non tradotto")
    check(page.locator(".toast").last.inner_text().strip() == "Example loaded: press “Ask”.", "en: notifica dell'esempio non tradotta")
    leftovers(page, ITALIAN_WORDS, "en · Example")
    page.screenshot(path=f"{OUT}/en_05_esempio.png", full_page=True)
    # dal vivo
    page.click("#mode-switch [data-mode='live']")
    check(text_of(page, "[data-source='webcam']") == "Camera" and text_of(page, "#ask-button") == "Start", "en: dal vivo non tradotto")
    leftovers(page, ITALIAN_WORDS, "en · Live")
    layout(page, "en_live")
    # storico e ricordi
    page.click(".tab[data-view='storico']")
    page.click("#history-filter [data-view='tutti']")
    page.wait_for_timeout(800)
    leftovers(page, ITALIAN_WORDS, "en · History")
    layout(page, "en_history")
    page.click(".tab[data-view='ricordi']")
    page.wait_for_selector(".memory-card")
    answers = page.locator(".qa b").all_inner_texts()
    check("Sì" not in answers and ("Yes" in answers or "No" in answers), f"en: risposte sì/no dei ricordi {answers[:6]}")
    leftovers(page, ITALIAN_WORDS, "en · Memories")
    layout(page, "en_memories")
    # la scelta resta dopo aver ricaricato la pagina
    page.reload()
    page.wait_for_selector(".memory-card")
    check(page.evaluate("document.documentElement.lang") == "en", "en: la lingua scelta non resta dopo aver ricaricato")
    context.close()
    browser.close()

print("\n".join(errors) or "nessun errore in console")
print("\n".join(problems) or "nessun problema: testi tradotti, niente parole dell'altra lingua, niente fuori pagina")
sys.exit(1 if problems or errors else 0)
