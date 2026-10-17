"""Browser QA of the running app (Chromium via Playwright): every page at desktop and narrow widths, console errors,
keyboard focus, and the Playground's Output panel. Screenshots go to OUT."""
import json, sys, time
from playwright.sync_api import sync_playwright

BASE, OUT = sys.argv[1], sys.argv[2]
PAGES = [("home", ""), ("learn", "Learn"), ("playground", "Playground"), ("batch", "Batch_Run"),
         ("qem_performance", "QEM_Performance"), ("noise_discrepancy", "Noise_Discrepancy"),
         ("cost_vs_precision", "Cost_vs_Precision"), ("model_mismatch", "Model_Mismatch"),
         ("theory", "Theory"), ("ibm", "IBM_Quantum")]
res = []
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=sys.argv[3] if len(sys.argv) > 3 else None)
    for vw, vh, tag in ((1440, 900, "desktop"), (390, 844, "narrow")):
        ctx = b.new_context(viewport=dict(width=vw, height=vh))
        page = ctx.new_page()
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        for name, path in PAGES:
            errors.clear()
            t0 = time.time()
            page.goto(BASE + "/" + path, wait_until="networkidle", timeout=120000)
            page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=120000)
            # wait until Streamlit stops running
            for _ in range(240):
                if page.locator("[data-testid='stStatusWidget']").count() == 0:
                    break
                time.sleep(0.5)
            time.sleep(1.0)
            dt = time.time() - t0
            exc = page.locator("[data-testid='stException']").count()
            sw = page.evaluate("document.documentElement.scrollWidth")
            page.screenshot(path="%s/%s_%s.png" % (OUT, tag, name), full_page=False)
            # keyboard: tab through the first controls and record what gets focus
            page.keyboard.press("Tab")
            focus = []
            for _ in range(6):
                page.keyboard.press("Tab")
                focus.append(page.evaluate("(document.activeElement && (document.activeElement.getAttribute('aria-label') || document.activeElement.innerText || document.activeElement.tagName) || '').slice(0,40)"))
            res.append(dict(viewport=tag, page=name, load_s=round(dt, 2), exceptions=exc, horizontal_overflow=sw > vw + 2,
                            scroll_width=sw, console_errors=[e[:200] for e in errors][:5], focus=focus))
            print(json.dumps(res[-1]), flush=True)
        ctx.close()
    # Playground: Output panel visible on desktop
    ctx = b.new_context(viewport=dict(width=1440, height=1000))
    page = ctx.new_page()
    page.goto(BASE + "/Playground", wait_until="networkidle", timeout=120000)
    time.sleep(4)
    out = page.get_by_text("Ideal prediction").count()
    page.screenshot(path="%s/desktop_playground_full.png" % OUT, full_page=True)
    res.append(dict(check="output panel present", ok=out > 0))
    ctx.close()
    b.close()
json.dump(res, open(OUT + "/browser_qa.json", "w"), indent=1)
