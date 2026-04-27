#!/usr/bin/env python3
"""Screenshot a single table (or section) from a generated ``report.html``.

Renders the HTML in headless Chromium (Playwright) so the output PNG matches
the browser rendering exactly: CSS styling, colored rank cells, aggregate
columns, tooltips layout, etc.

Prerequisites (run once):
    pip install playwright
    playwright install chromium

Usage:
    python -m audio_bench.table_to_png plots/report.html overview-tbl \\
        --output plots/overview.png
    python -m audio_bench.table_to_png plots/report.html cat-Overview \\
        --output plots/overview_section.png --full_section
"""

import argparse
import os
import sys
from pathlib import Path


def screenshot_element(html_path, element_id, output_path, *,
                       full_section=False, scale=2, width=1800, expand=False):
    from playwright.sync_api import sync_playwright

    html_path = Path(html_path).resolve()
    if not html_path.is_file():
        sys.exit(f"HTML file not found: {html_path}")

    url = html_path.as_uri()
    selector = f"#{element_id}" if full_section else f"table#{element_id}, #{element_id}"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": width, "height": 1000},
            device_scale_factor=scale,
        )
        page = context.new_page()
        page.goto(url, wait_until="load")

        locator = page.locator(selector).first
        if locator.count() == 0:
            browser.close()
            sys.exit(f"Element #{element_id} not found in {html_path}")

        if expand:
            locator.evaluate(
                "el => el.querySelectorAll('.toggle-btn').forEach("
                "b => { if (b.textContent.trim() === '+') b.click(); })"
            )

        # Strip overflow clipping on ancestors (e.g. .figure-wrapper has
        # overflow-x: auto) so wide expanded tables are not cropped.
        locator.evaluate(
            "el => { for (let n = el.parentElement; n; n = n.parentElement) {"
            "  n.style.overflow = 'visible';"
            "  n.style.maxWidth = 'none';"
            "  n.style.width = 'auto';"
            "} }"
        )

        locator.scroll_into_view_if_needed()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        locator.screenshot(path=output_path)
        browser.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Screenshot a table or section from a generated report.html.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--html_path", default="plots/report.html", help="Path to report.html (e.g. plots/report.html)")
    parser.add_argument("--element_id", default="overview-tbl",
                        help="DOM id of the table or section (e.g. overview-tbl)")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: <element_id>.png next to the HTML)")
    parser.add_argument("--scale", type=int, default=2,
                        help="Device scale factor for sharpness")
    parser.add_argument("--width", type=int, default=1800,
                        help="Viewport width (raise for very wide overview tables)")
    parser.add_argument("--full_section", action="store_true",
                        help="Screenshot the enclosing section/div instead of just the table")
    parser.add_argument("--expand", action="store_true",
                        help="Click all [+] toggles inside the target to expand sub-columns before screenshotting")
    args = parser.parse_args()

    output = args.output or str(Path(args.html_path).parent / f"{args.element_id}.png")
    screenshot_element(args.html_path, args.element_id, output,
                       full_section=args.full_section,
                       scale=args.scale, width=args.width, expand=args.expand)


if __name__ == "__main__":
    main()
