"""Collaudo dell'interfaccia in un browser headless (Playwright): percorso completo con screenshot.

    uv pip install --python .venv/bin/python playwright && .venv/bin/python -m playwright install chromium
    .venv/bin/egeria model-server                                                          # il modello (porta 8100)
    .venv/bin/egeria serve --memory runs/ui-test/memoria --data runs/ui-test --port 8001   # istanza web di prova
    .venv/bin/python scripts/collaudo_ui.py http://127.0.0.1:8001 runs/ui-test/screenshot

Scrive due domande a mano, chiede, corregge e salva nei ricordi, prova l'esempio con la foto,
la modalità dal vivo, lo storico, i ricordi e il tema. Stampa gli errori della console del browser.
Usa un'istanza separata: crea casi e ricordi di prova.
"""
import os
import sys
from playwright.sync_api import sync_playwright
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/ui-test/screenshot"
os.makedirs(OUT, exist_ok=True)
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.on("console", lambda m: errors.append(f"console {m.type}: {m.text}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.goto(URL + "/")
    page.wait_for_selector(".question")
    page.screenshot(path=f"{OUT}/01_vuota.png", full_page=True)
    # scrivo una domanda a mano, senza esempi
    page.fill("#input-text", "Salve, il mio ordine 4471 non è ancora arrivato e il tracking è fermo da una settimana. Vorrei il rimborso.")
    page.fill(".question-text", "Il cliente chiede un rimborso?")
    page.click("#add-question")
    page.locator(".question-text").nth(1).fill("Qual è il problema?")
    page.locator(".question select").nth(1).select_option("choice")
    chip = page.locator(".question").nth(1).locator(".chip-input")
    for option in ["Consegna in ritardo", "Prodotto difettoso", "Fatturazione"]:
        chip.fill(option); chip.press("Enter")
        chip = page.locator(".question").nth(1).locator(".chip-input")
    page.screenshot(path=f"{OUT}/02_domande_scritte.png", full_page=True)
    page.click("#ask-button")
    page.wait_for_selector(".answer", timeout=120000)
    page.wait_for_timeout(500)
    page.screenshot(path=f"{OUT}/03_risposte.png", full_page=True)
    # correggo la prima risposta e salvo nei ricordi
    page.locator(".answer").first.get_by_role("button", name="Correggi").click()
    page.locator(".correction select").first.select_option(index=1)
    page.locator(".correction").first.get_by_role("button", name="Ok").click()
    page.click("#teach-button")
    page.wait_for_selector(".teach.saved", timeout=60000)
    page.screenshot(path=f"{OUT}/04_salvato.png", full_page=True)
    # esempio con foto
    page.once("dialog", lambda d: d.accept())
    page.select_option("#examples", label="Foto di un incidente")
    page.click("#ask-button")
    page.wait_for_function("document.querySelectorAll('.answer').length >= 3", timeout=120000)
    page.wait_for_timeout(500)
    page.screenshot(path=f"{OUT}/05_foto.png", full_page=True)
    # dal vivo
    page.click("#mode-switch [data-mode='live']")
    page.screenshot(path=f"{OUT}/06_dal_vivo.png", full_page=True)
    # storico e ricordi
    page.click(".tab[data-view='storico']")
    page.click("#history-filter [data-view='tutti']")
    page.wait_for_timeout(800)
    page.screenshot(path=f"{OUT}/07_storico.png", full_page=True)
    page.click(".tab[data-view='ricordi']")
    page.wait_for_timeout(800)
    page.screenshot(path=f"{OUT}/08_ricordi.png", full_page=True)
    # tema chiaro
    page.click("#theme-button")
    page.click(".tab[data-view='chiedi']")
    page.wait_for_timeout(300)
    page.screenshot(path=f"{OUT}/09_chiaro.png", full_page=True)
    browser.close()
print("\n".join(errors) or "nessun errore in console")
