"""Centralised Playwright client for CollegeTransfer.net scraping.

This module exposes a lightweight context manager that owns the Playwright
lifecycle so the rest of the codebase can reuse a single browser/page session
when harvesting course information. Keeping the session logic here prevents
subtle drift between one-off scripts and the main system.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from playwright.async_api import async_playwright, Browser, Page, Playwright


@dataclass
class CollegeTransferClient:
    """Async context manager responsible for Playwright session management."""

    headless: bool = True
    default_timeout_ms: int = 30_000

    _playwright: Optional[Playwright] = None
    _browser: Optional[Browser] = None
    _page: Optional[Page] = None

    async def __aenter__(self) -> "CollegeTransferClient":
        self._playwright = await async_playwright().start()
        browser_kwargs = {"headless": self.headless}
        self._browser = await self._playwright.chromium.launch(**browser_kwargs)
        self._page = await self._browser.new_page()
        self._page.set_default_timeout(self.default_timeout_ms)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Playwright page not initialised; use 'async with CollegeTransferClient()'.")
        return self._page

    async def close(self) -> None:
        """Gracefully close browser resources."""
        if self._page is not None:
            await self._page.close()
            self._page = None
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def ensure_ready(self) -> Page:
        """Return an initialised page instance, awaiting the context if needed."""
        if self._page is None:
            # When users instantiate manually without a context manager we lazily create resources.
            await self.__aenter__()
        return self.page


async def get_default_client(headless: bool = True) -> CollegeTransferClient:
    """Convenience helper for one-off usage without manual context setup."""
    client = CollegeTransferClient(headless=headless)
    await client.__aenter__()
    return client


async def close_client(client: CollegeTransferClient) -> None:
    """Close a previously created client."""
    await client.close()


__all__ = ["CollegeTransferClient", "get_default_client", "close_client"]
