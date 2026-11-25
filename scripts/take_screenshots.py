"""Save screenshots of every app page to reports/screenshots/ (for the README and slides).

Optional tool, NOT part of requirements.txt. Needs Playwright in any Python environment and an
installed Google Chrome:
    pip install playwright            # e.g. in a separate venv
    streamlit run app.py              # in another terminal (default port 8501)
    python scripts/take_screenshots.py [--url http://localhost:8501]

Each page is opened with ?example=at_risk so a real example student is loaded.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "reports" / "screenshots"
PAGES = [("01_home", ""), ("02_prediction", "prediction"), ("03_clustering", "clustering"),
         ("04_planner", "planner"), ("05_evaluation", "evaluation"), ("06_about", "about")]
MOBILE = [("mobile_02_prediction", "prediction"), ("mobile_04_planner", "planner")]


def wait_for_app(page) -> None:
    """Wait until the page title is rendered and the app has been idle (no 'Running' widget) for 2 s."""
    page.wait_for_selector("h1", timeout=60_000)
    idle_since = None
    deadline = time.time() + 90
    while time.time() < deadline:
        running = page.query_selector('[data-testid="stStatusWidget"]') is not None
        if running:
            idle_since = None
        elif idle_since is None:
            idle_since = time.time()
        elif time.time() - idle_since > 2.0:
            break
        time.sleep(0.25)
    time.sleep(2.0)  # Plotly charts finish drawing


def fit_viewport(page, width: int) -> None:
    """Streamlit scrolls inside its main container, so grow the viewport to the content height."""
    height = page.evaluate("() => { const m = document.querySelector('[data-testid=\"stMain\"]');"
                           " return Math.max(m ? m.scrollHeight : 0, document.body.scrollHeight); }")
    page.set_viewport_size({"width": width, "height": int(height) + 40})
    time.sleep(1.5)


def shoot(page, base: str, name: str, path: str, example: str, after=None) -> None:
    """Open one page with an example student and save a full-height screenshot."""
    page.goto(f"{base}/{path}?example={example}", wait_until="domcontentloaded")
    wait_for_app(page)
    if after:
        after(page)
    width = page.viewport_size["width"]
    fit_viewport(page, width)
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
    page.set_viewport_size({"width": width, "height": 900 if width > 600 else 844})
    print(f"  saved reports/screenshots/{name}.png")


def run_simulation(page) -> None:
    """Click the Study Planner's simulate button and wait for the replay."""
    page.get_by_role("button", name="▶ Simulate a study session").click()
    time.sleep(1.0)
    wait_for_app(page)


def main() -> int:
    """Screenshot every page (desktop) and two pages at phone width."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--example", default="at_risk")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        desk = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        for name, path in PAGES:
            shoot(desk, args.url, name, path, args.example, run_simulation if path == "planner" else None)
        phone = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True)
        for name, path in MOBILE:
            shoot(phone, args.url, name, path, args.example)
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
