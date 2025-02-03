from typing import Optional, Dict
from browserbase import Browserbase
from browserbase.types.session_create_params import BrowserSettings, BrowserSettingsFingerprint, BrowserSettingsFingerprintScreen, BrowserSettingsViewport
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig
import logging
import os
from asteroid_sdk.api.generated.asteroid_api_client.api.run.update_run_metadata import sync_detailed
from asteroid_sdk.api.generated.asteroid_api_client.models.update_run_metadata_body import UpdateRunMetadataBody
from asteroid_sdk.api.generated.asteroid_api_client.client import Client
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
logger = logging.getLogger(__name__)

async def init_browser(browserbase_project_id: Optional[str] = None, folder_name: Optional[str] = None, headless: bool = False,
                 height: int = 800, width: int = 1280) -> tuple[Browser, Optional[str], Optional[str]]:
    """Initialize browser with Browserbase if credentials are provided, otherwise use local browser.
    Returns a tuple of (Browser, debug_url, session_id)"""
    cdp_url = None
    debug_url = None
    session_id = None
    if browserbase_project_id:
        logger.info("Connecting to Browserbase session...")
        bb = Browserbase(api_key=os.getenv("BROWSERBASE_API_KEY"))
        browser_settings = BrowserSettings(
            block_ads=True,
            fingerprint=BrowserSettingsFingerprint(
                browsers=["chrome"],
                devices=["desktop"],
                locales=["en-US"],
                operating_systems=["macos"],
                screen=BrowserSettingsFingerprintScreen(
                    max_height=height,
                    max_width=width,
                    min_height=height,
                    min_width=width
                )
            ),
            solve_captchas=True,
            viewport=BrowserSettingsViewport(
                height=height,
                width=width
            )
        )
        
        session = bb.sessions.create(project_id=browserbase_project_id, browser_settings=browser_settings,
                                     region="us-east-1", proxies=False)
        session_id = session.id
        debug_urls = bb.sessions.debug(session.id)

        debug_url = debug_urls.debugger_fullscreen_url
        logger.info(f"Browserbase Debug Connection URL: {debug_url}")
        logger.info(f"Browserbase CDP URL: {session.connect_url}")

        cdp_url = session.connect_url
        headless = False  
    else:
        logger.info("Using local browser (no Browserbase).")

    browser = Browser(
        config=BrowserConfig(
            headless=headless,
            cdp_url=cdp_url,
            new_context_config=BrowserContextConfig(
                apply_click_styling=True,
                apply_form_related=True,
                browser_window_size={"width": 1024, "height": 768},
                save_recording_path=folder_name
            ),
        )
    )
    return browser, debug_url, session_id


async def do_update_run_metadata(run_id: str, run_metadata: Dict[str, str]) -> None:
    """
    Updates the run metadata with the provided run_metadata
    """
    print(f"Updating run metadata for run_id: {run_id}")
    
    asteroid_client = Client(
        base_url="http://localhost:8080/api/v1",
        headers={"X-Asteroid-Api-Key": os.getenv("ASTEROID_API_KEY")}
    )

    metadata = UpdateRunMetadataBody.from_dict(run_metadata)

    try:
        sync_detailed(
            run_id, 
            client=asteroid_client,
            body=metadata
        )
    except Exception as e:
        logger.error(f"Error updating run metadata: {e}")

    logger.info(f"Run metadata has been updated with: {run_metadata}")