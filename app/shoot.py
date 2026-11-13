"""Playwright screenshot runner (v2). Real chromium, fake media for voice.
Run AFTER the server is up on http://127.0.0.1:8765 and the index is built.
"""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8765"
ROOT = Path("/home/user/Doubao/chats/38450574805643778")
SHOTS = ROOT / "screenshots"
for sub in ("portal", "aggregator", "cms", "rag", "translation", "voice"):
    (SHOTS / sub).mkdir(parents=True, exist_ok=True)


async def shoot(page, name, full=True):
    path = SHOTS / name
    await page.screenshot(path=str(path), full_page=full)
    print("saved", path)


async def wait_stream(page, selector, timeout=90):
    for _ in range(timeout):
        await page.wait_for_timeout(1000)
        txt = await page.text_content(selector)
        if txt and not txt.startswith("（"):
            await page.wait_for_timeout(1500)
            txt2 = await page.text_content(selector)
            if txt == txt2:
                return


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            executable_path="/usr/local/bin/chromium",
            args=["--use-fake-ui-for-media-stream",
                  "--use-fake-device-for-media-stream",
                  "--no-sandbox", "--disable-dev-shm-usage"],
        )
        ctx = await browser.new_context(viewport={"width": 1440, "height": 1000})
        page = await ctx.new_page()

        # ---------- portal ----------
        await page.goto(BASE + "/", wait_until="networkidle")
        await page.wait_for_timeout(1200)
        await shoot(page, "portal/portal-v2.png")

        # ---------- aggregator ----------
        await page.goto(BASE + "/aggregator", wait_until="networkidle")
        await page.wait_for_timeout(1500)
        await shoot(page, "aggregator/aggregator-list.png")
        await page.select_option("#f-source", "AHA Centre")
        await page.wait_for_timeout(800)
        await shoot(page, "aggregator/aggregator-filter-source.png")

        # ---------- CMS ----------
        await page.goto(BASE + "/cms", wait_until="networkidle")
        await page.wait_for_timeout(1200)
        await shoot(page, "cms/cms-news-list.png")
        await page.click('button[data-tab="aggregator"]')
        await page.wait_for_timeout(1000)
        await shoot(page, "cms/cms-aggregator-tab.png")
        await page.click('button[data-tab="term"]')
        await page.wait_for_timeout(1000)
        await shoot(page, "cms/cms-terms-tab.png")
        await page.click('button[data-act="review"]')
        await page.wait_for_timeout(800)
        await shoot(page, "cms/cms-review-toggle.png")

        # ---------- RAG (real BigModel + DeepSeek) ----------
        await page.goto(BASE + "/rag", wait_until="networkidle")
        await page.wait_for_timeout(800)
        await page.fill("#q", "OSOCC 是什么？它的主要职责有哪些？")
        await page.click("#ask")
        await wait_stream(page, "#answer")
        await page.wait_for_timeout(800)
        await shoot(page, "rag/rag-answer-bigmodel.png")

        # refuse case: out-of-domain question
        await page.fill("#q", "如何做巧克力蛋糕？")
        await page.click("#ask")
        await wait_stream(page, "#answer")
        await page.wait_for_timeout(600)
        await shoot(page, "rag/rag-refuse-offtopic.png")

        # ---------- translation ----------
        await page.goto(BASE + "/translate", wait_until="networkidle")
        await page.wait_for_timeout(800)
        await page.select_option("#tgt", "th")
        await page.fill("#text",
            "The On-Site Operations Coordination Centre (OSOCC) coordinates international "
            "Urban Search and Rescue (USAR) teams after a major earthquake.")
        await page.click("#go")
        await wait_stream(page, "#out")
        await page.wait_for_timeout(800)
        await shoot(page, "translation/translate-th-term-highlight.png")

        # ---------- voice ----------
        await page.goto(BASE + "/voice", wait_until="networkidle")
        await page.wait_for_timeout(800)
        await page.evaluate(
            "window.__voiceInject('The On-Site Operations Coordination Centre coordinates international USAR teams.')"
        )
        await wait_stream(page, "#out")
        await page.wait_for_timeout(800)
        await shoot(page, "voice/voice-translate.png")

        await browser.close()


asyncio.run(main())
print("ALL SHOTS DONE")
