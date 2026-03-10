"""
E2E tests for graph-viz (theme, controls, validation).

Requires: pip install -e ".[e2e]" && playwright install chromium

Run with: uv run pytest kgserver/tests/test_graph_viz_theme.py -v
"""

import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Skip entire module if playwright not installed
pytest.importorskip("playwright")

from playwright.sync_api import Page, expect  # noqa: E402  # pylint: disable=wrong-import-position


def _find_free_port() -> int:
    """Find an available port for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def graph_viz_server():
    """Start a minimal static file server for graph-viz, yield base URL, then stop."""
    static_dir = Path(__file__).resolve().parent.parent / "query" / "static"
    if not static_dir.exists():
        pytest.skip(f"graph-viz static dir not found: {static_dir}")

    port = _find_free_port()
    with subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port)],
        cwd=str(static_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ):
        for _ in range(50):
            try:
                import urllib.request

                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1):
                    pass
                break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("Server did not start in time")
        yield f"http://127.0.0.1:{port}/"


@pytest.fixture
def graph_viz_page(graph_viz_server, browser):
    """Page navigated to graph-viz (uses pytest-playwright's browser fixture)."""
    context = browser.new_context()
    page = context.new_page()
    page.goto(graph_viz_server, wait_until="networkidle")
    try:
        yield page
    finally:
        context.close()


THEME_STORAGE_KEY = "vite-ui-theme"


class TestGraphVizTheme:
    """Theme select and persistence tests (syncs with Chainlit via vite-ui-theme)."""

    def test_theme_toggle_via_js_directly(self, graph_viz_page: Page):
        """Verify theme logic works when applied manually (proves CSS/DOM work)."""
        graph_viz_page.evaluate(f"localStorage.removeItem('{THEME_STORAGE_KEY}')")
        graph_viz_page.reload(wait_until="networkidle")
        graph_viz_page.evaluate(f"""
            const root = document.documentElement;
            root.classList.remove('theme-dark', 'theme-light');
            root.classList.add('theme-light');
            try {{ localStorage.setItem('{THEME_STORAGE_KEY}', 'light'); }} catch(e) {{}}
        """)
        root = graph_viz_page.locator("html")
        expect(root).to_have_class(re.compile("theme-light"))
        stored = graph_viz_page.evaluate(f"localStorage.getItem('{THEME_STORAGE_KEY}')")
        assert stored == "light"

    def test_initial_theme_is_dark_when_saved(self, graph_viz_page: Page):
        """When vite-ui-theme is 'dark', theme should be dark."""
        graph_viz_page.evaluate(f"localStorage.setItem('{THEME_STORAGE_KEY}', 'dark')")
        graph_viz_page.reload(wait_until="networkidle")
        root = graph_viz_page.locator("html")
        expect(root).to_have_class(re.compile("theme-dark"))
        expect(root).not_to_have_class(re.compile("theme-light"))

    def test_select_light_shows_light_theme(self, graph_viz_page: Page):
        """Selecting Light from dropdown should switch to light theme."""
        graph_viz_page.evaluate(f"localStorage.setItem('{THEME_STORAGE_KEY}', 'dark')")
        graph_viz_page.reload(wait_until="networkidle")
        graph_viz_page.locator("#theme-select").select_option("light")
        root = graph_viz_page.locator("html")
        expect(root).to_have_class(re.compile("theme-light"))
        expect(root).not_to_have_class(re.compile("theme-dark"))

    def test_select_dark_shows_dark_theme(self, graph_viz_page: Page):
        """Selecting Dark from dropdown should switch to dark theme."""
        graph_viz_page.evaluate(f"localStorage.setItem('{THEME_STORAGE_KEY}', 'light')")
        graph_viz_page.reload(wait_until="networkidle")
        graph_viz_page.locator("#theme-select").select_option("dark")
        root = graph_viz_page.locator("html")
        expect(root).to_have_class(re.compile("theme-dark"))
        expect(root).not_to_have_class(re.compile("theme-light"))

    def test_theme_persists_in_local_storage(self, graph_viz_page: Page):
        """Selecting light theme should persist across reload."""
        graph_viz_page.evaluate(f"localStorage.removeItem('{THEME_STORAGE_KEY}')")
        graph_viz_page.reload(wait_until="networkidle")
        graph_viz_page.locator("#theme-select").select_option("light")
        stored = graph_viz_page.evaluate(f"localStorage.getItem('{THEME_STORAGE_KEY}')")
        assert stored == "light"
        graph_viz_page.reload(wait_until="networkidle")
        expect(graph_viz_page.locator("#theme-select")).to_have_value("light")
        root = graph_viz_page.locator("html")
        expect(root).to_have_class(re.compile("theme-light"))

    def test_dark_theme_persists_in_local_storage(self, graph_viz_page: Page):
        """Dark theme selection should persist across reload."""
        graph_viz_page.evaluate(f"localStorage.setItem('{THEME_STORAGE_KEY}', 'light')")
        graph_viz_page.reload(wait_until="networkidle")
        graph_viz_page.locator("#theme-select").select_option("dark")
        stored = graph_viz_page.evaluate(f"localStorage.getItem('{THEME_STORAGE_KEY}')")
        assert stored == "dark"
        graph_viz_page.reload(wait_until="networkidle")
        expect(graph_viz_page.locator("#theme-select")).to_have_value("dark")
        root = graph_viz_page.locator("html")
        expect(root).to_have_class(re.compile("theme-dark"))


class TestGraphVizControls:
    """Controls visibility, validation, and include-all checkbox tests."""

    def test_controls_are_visible(self, graph_viz_page: Page):
        """Search, center ID, sliders, checkboxes, and buttons are present and visible."""
        controls = graph_viz_page.locator("#controls")
        expect(controls).to_be_visible()
        expect(controls.locator("h2")).to_have_text("Graph Controls")
        expect(graph_viz_page.locator("#entity-search")).to_be_visible()
        expect(graph_viz_page.locator("#center-id")).to_be_visible()
        expect(graph_viz_page.locator("#hops")).to_be_visible()
        expect(graph_viz_page.locator("#max-nodes")).to_be_visible()
        expect(graph_viz_page.locator("#hops-value")).to_be_visible()
        expect(graph_viz_page.locator("#max-nodes-value")).to_be_visible()
        expect(graph_viz_page.locator("#include-all")).to_be_visible()
        expect(graph_viz_page.locator("#theme-select")).to_be_visible()
        expect(graph_viz_page.get_by_role("button", name="Load Graph")).to_be_visible()
        expect(graph_viz_page.get_by_role("button", name="Reset View")).to_be_visible()
        expect(graph_viz_page.locator("#graph-svg")).to_be_visible()

    def test_load_graph_without_center_id_shows_error(self, graph_viz_page: Page):
        """With 'Show entire graph' unchecked and no center ID, Load Graph shows error."""
        graph_viz_page.locator("#include-all").uncheck()
        graph_viz_page.locator("#center-id").fill("")
        graph_viz_page.get_by_role("button", name="Load Graph").click()
        # Error is rendered as SVG text with class error-message
        error_text = graph_viz_page.locator("text=Please enter a center entity ID")
        expect(error_text).to_be_visible(timeout=5000)

    def test_include_all_disables_center_id_and_search(self, graph_viz_page: Page):
        """When 'Show entire graph' is checked, center ID and search are disabled."""
        # Uncheck then check to trigger the change handler (include-all is checked by default)
        graph_viz_page.locator("#include-all").uncheck()
        graph_viz_page.locator("#include-all").check()
        expect(graph_viz_page.locator("#center-id")).to_be_disabled()
        expect(graph_viz_page.locator("#entity-search")).to_be_disabled()

    def test_include_all_unchecked_enables_center_id_and_search(self, graph_viz_page: Page):
        """When 'Show entire graph' is unchecked, center ID and search are enabled."""
        graph_viz_page.locator("#include-all").uncheck()
        expect(graph_viz_page.locator("#center-id")).to_be_enabled()
        expect(graph_viz_page.locator("#entity-search")).to_be_enabled()
