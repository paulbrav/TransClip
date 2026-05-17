#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

CHATGPT_URL = "https://chatgpt.com/"
DEFAULT_PROFILE = Path("~/.cache/granite-speach-chatgpt-ui").expanduser()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask ChatGPT through the visible browser UI using Playwright.",
    )
    parser.add_argument("prompt", nargs="*", help="Prompt text. Reads stdin when omitted.")
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help=f"Persistent browser profile directory. Default: {DEFAULT_PROFILE}",
    )
    parser.add_argument(
        "--model",
        default="Extended Pro",
        help='Visible model label to select before sending. Use "" to skip selection.',
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--browser-channel",
        default=os.environ.get("CHATGPT_UI_BROWSER_CHANNEL", "chrome"),
        help='Playwright browser channel, such as "chrome" or "chromium".',
    )
    parser.add_argument(
        "--reuse-current-chat",
        action="store_true",
        help="Do not try to open a new chat before sending the prompt.",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Leave the browser open after printing the answer.",
    )
    args = parser.parse_args(argv)

    prompt = " ".join(args.prompt).strip() if args.prompt else sys.stdin.read().strip()
    if not prompt:
        print("prompt is required on argv or stdin", file=sys.stderr)
        return 2

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright is not installed. Run with: uv run --extra chatgpt-ui scripts/chatgpt_ui_ask.py ...",
            file=sys.stderr,
        )
        return 2

    args.profile.expanduser().mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        launch_kwargs = {
            "user_data_dir": str(args.profile.expanduser()),
            "headless": args.headless,
            "viewport": {"width": 1440, "height": 900},
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        if args.browser_channel:
            launch_kwargs["channel"] = args.browser_channel
        try:
            context = playwright.chromium.launch_persistent_context(**launch_kwargs)
        except PlaywrightTimeoutError:
            raise
        except Exception as exc:
            if args.browser_channel == "chrome":
                print(
                    "Could not launch Chrome channel; retrying bundled Chromium. "
                    "Install Chrome support with: uv run --extra chatgpt-ui playwright install chrome",
                    file=sys.stderr,
                )
                launch_kwargs.pop("channel", None)
                context = playwright.chromium.launch_persistent_context(**launch_kwargs)
            else:
                print(f"could not launch browser: {exc}", file=sys.stderr)
                return 1

        page = context.pages[0] if context.pages else context.new_page()
        try:
            page.goto(CHATGPT_URL, wait_until="domcontentloaded", timeout=60_000)
            wait_for_composer(page, args.timeout)
            if not args.reuse_current_chat:
                open_new_chat(page)
                wait_for_composer(page, args.timeout)
            if args.model:
                select_model(page, args.model)
            before_count = assistant_message_count(page)
            submit_prompt(page, prompt)
            answer = wait_for_answer(page, before_count, args.timeout)
            print(answer)
            return 0
        except PlaywrightTimeoutError as exc:
            print(f"timed out while driving ChatGPT UI: {exc}", file=sys.stderr)
            return 1
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        finally:
            if args.keep_open:
                print("\nBrowser left open. Press Ctrl+C here when finished.", file=sys.stderr)
                try:
                    while True:
                        time.sleep(3600)
                except KeyboardInterrupt:
                    pass
            context.close()


def wait_for_composer(page, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    selectors = [
        '[contenteditable="true"]',
        'textarea[placeholder*="Ask"]',
        "textarea",
    ]
    while time.monotonic() < deadline:
        if login_visible(page):
            raise RuntimeError(
                "ChatGPT is not logged in in this browser profile. "
                "Run again with --keep-open, log in manually, then rerun."
            )
        for selector in selectors:
            loc = page.locator(selector).last
            if loc.count() and loc.is_visible():
                return
        page.wait_for_timeout(500)
    raise RuntimeError("could not find ChatGPT composer")


def login_visible(page) -> bool:
    for text in ("Log in", "Sign up"):
        loc = page.get_by_text(text, exact=True)
        if loc.count() and loc.first.is_visible():
            return True
    return False


def open_new_chat(page) -> None:
    candidates = [
        page.get_by_label("New chat"),
        page.get_by_role("link", name="New chat"),
        page.get_by_role("button", name="New chat"),
        page.locator('a[href="/"], a[href="/?temporary-chat=true"]').first,
    ]
    for candidate in candidates:
        try:
            if candidate.count() and candidate.first.is_visible():
                candidate.first.click(timeout=2_000)
                page.wait_for_timeout(750)
                return
        except Exception:
            continue


def select_model(page, model: str) -> None:
    if page.get_by_text(model, exact=True).count():
        visible_matches = [
            page.get_by_text(model, exact=True).nth(i).is_visible()
            for i in range(page.get_by_text(model, exact=True).count())
        ]
        if any(visible_matches):
            return

    buttons = page.locator("button")
    for i in range(buttons.count()):
        button = buttons.nth(i)
        try:
            if not button.is_visible():
                continue
            label = button.inner_text(timeout=500).strip()
            if any(token in label for token in ("GPT", "Pro", "Extended", "Auto")):
                button.click(timeout=2_000)
                break
        except Exception:
            continue
    else:
        return

    option = page.get_by_text(model, exact=True)
    try:
        option.first.wait_for(state="visible", timeout=5_000)
        option.first.click(timeout=5_000)
        page.wait_for_timeout(500)
    except Exception:
        print(f'warning: could not select visible model "{model}"', file=sys.stderr)


def submit_prompt(page, prompt: str) -> None:
    composer = find_composer(page)
    composer.click()
    try:
        composer.fill(prompt)
    except Exception:
        page.keyboard.insert_text(prompt)

    for selector in (
        '[data-testid="send-button"]',
        'button[aria-label="Send prompt"]',
        'button[aria-label="Send message"]',
    ):
        button = page.locator(selector).last
        try:
            if button.count() and button.is_visible() and button.is_enabled():
                button.click(timeout=2_000)
                return
        except Exception:
            continue
    page.keyboard.press("Enter")


def find_composer(page):
    selectors = [
        '[contenteditable="true"]',
        'textarea[placeholder*="Ask"]',
        "textarea",
    ]
    for selector in selectors:
        loc = page.locator(selector).last
        if loc.count() and loc.is_visible():
            return loc
    raise RuntimeError("could not find ChatGPT composer")


def assistant_message_count(page) -> int:
    return page.locator('[data-message-author-role="assistant"]').count()


def wait_for_answer(page, before_count: int, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    last_text = ""
    stable_since: float | None = None
    while time.monotonic() < deadline:
        text = last_assistant_text(page, before_count)
        if text and text != last_text:
            last_text = text
            stable_since = time.monotonic()
        elif text and stable_since and time.monotonic() - stable_since >= 3.0:
            if not response_is_streaming(page):
                return text
        page.wait_for_timeout(500)
    if last_text:
        return last_text
    raise RuntimeError("no assistant response appeared before timeout")


def last_assistant_text(page, before_count: int) -> str:
    return page.evaluate(
        """
        ({ beforeCount }) => {
          const roleNodes = Array.from(document.querySelectorAll('[data-message-author-role="assistant"]'));
          const candidates = roleNodes.length > beforeCount ? roleNodes.slice(beforeCount) : roleNodes;
          const node = candidates[candidates.length - 1];
          if (node && node.innerText.trim()) return node.innerText.trim();

          const articles = Array.from(document.querySelectorAll('article'));
          for (let i = articles.length - 1; i >= 0; i--) {
            const text = articles[i].innerText.trim();
            if (text && !text.includes('You said:')) return text;
          }
          return '';
        }
        """,
        {"beforeCount": before_count},
    )


def response_is_streaming(page) -> bool:
    return page.evaluate(
        """
        () => {
          const labels = ['Stop generating', 'Stop streaming', 'Cancel'];
          for (const label of labels) {
            const el = document.querySelector(`[aria-label="${label}"]`);
            if (el) return true;
          }
          const buttons = Array.from(document.querySelectorAll('button'));
          return buttons.some((button) => labels.includes(button.innerText.trim()));
        }
        """,
    )


if __name__ == "__main__":
    raise SystemExit(main())
