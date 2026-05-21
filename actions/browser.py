import webbrowser
import urllib.parse
import time
import logging
import json
import os
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from collections import deque
import pickle

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logging.warning("⚠️ Selenium not installed. Using webbrowser fallback.")


# =========================
# LOGGING SETUP
# =========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# ENUMS & DATA CLASSES
# =========================

class BrowserType(Enum):
    """Available browsers"""
    CHROME = "chrome"
    FIREFOX = "firefox"
    EDGE = "edge"
    SAFARI = "safari"


class SearchEngine(Enum):
    """Available search engines"""
    GOOGLE = "google"
    YOUTUBE = "youtube"
    BING = "bing"
    WIKIPEDIA = "wikipedia"
    GITHUB = "github"
    STACKOVERFLOW = "stackoverflow"


@dataclass
class BrowsingSession:
    """Browser session information"""
    session_id: str
    browser_type: BrowserType
    start_time: datetime
    end_time: Optional[datetime] = None
    pages_visited: int = 0
    total_duration: float = 0.0
    tabs_opened: int = 0


@dataclass
class PageVisit:
    """Information about a page visit"""
    url: str
    title: str
    timestamp: datetime
    duration: float
    status_code: Optional[int] = None
    screenshot_path: Optional[str] = None


@dataclass
class BrowserConfig:
    """Browser configuration"""
    browser_type: BrowserType = BrowserType.CHROME
    headless: bool = False
    incognito: bool = False
    user_data_dir: Optional[str] = None
    use_existing_profile: bool = False  # Use existing Chrome profile
    profile_name: str = "Default"  # Profile name (Default, Profile 1, etc.)
    email: Optional[str] = None  # Email for profile identification
    disable_images: bool = False
    disable_notifications: bool = True
    user_agent: Optional[str] = None
    proxy: Optional[str] = None
    window_size: Tuple[int, int] = (1920, 1080)


# =========================
# ADVANCED BROWSER MANAGER
# =========================

class AdvancedBrowserManager:
    """
    Advanced browser management system with:
    - Multiple browser support
    - Tab management
    - Cookie handling
    - History tracking
    - Screenshots
    - Performance monitoring
    - Automation capabilities
    """

    def __init__(self, config: BrowserConfig = None):
        """Initialize browser manager"""
        self.config = config or BrowserConfig()
        self.driver = None
        self.session_history: deque = deque(maxlen=50)
        self.page_visits: List[PageVisit] = []
        self.tabs: List[str] = []
        self.cookies: Dict[str, Any] = {}
        self.callbacks: List[Callable] = []
        self.current_session: Optional[BrowsingSession] = None
        self.start_time = datetime.now()

        logger.info(f"✅ Advanced Browser Manager initialized")

    # =========================
    # BROWSER INITIALIZATION
    # =========================

    def _get_chrome_user_data_dir(self) -> Optional[str]:
        """
        Get Chrome user data directory based on OS
        """
        import platform
        import os

        username = os.getenv('USERNAME') or os.getenv('USER')

        if platform.system() == "Windows":
            return f"C:\\Users\\{username}\\AppData\\Local\\Google\\Chrome\\User Data"
        elif platform.system() == "Darwin":  # macOS
            return f"/Users/{username}/Library/Application Support/Google/Chrome"
        elif platform.system() == "Linux":
            return f"/home/{username}/.config/google-chrome"

        return None

    def _initialize_selenium(self) -> Optional[webdriver.Chrome]:
        """Initialize Selenium WebDriver with profile support"""
        try:
            if not SELENIUM_AVAILABLE:
                logger.warning("⚠️ Selenium not available, using webbrowser")
                return None

            options = ChromeOptions()

            # Use existing Chrome profile if requested
            if self.config.use_existing_profile:
                chrome_user_data = self._get_chrome_user_data_dir()

                if chrome_user_data and os.path.exists(chrome_user_data):
                    options.add_argument(f"user-data-dir={chrome_user_data}")
                    options.add_argument(f"profile-directory={self.config.profile_name}")

                    logger.info(f"✅ Using Chrome profile: {self.config.profile_name}")

                    if self.config.email:
                        logger.info(f"✅ Profile email: {self.config.email}")
                else:
                    logger.warning(f"⚠️ Chrome user data not found, using default")

            elif self.config.user_data_dir:
                options.add_argument(f"user-data-dir={self.config.user_data_dir}")

            # Add options based on config
            if self.config.headless:
                options.add_argument("--headless")

            if self.config.incognito:
                options.add_argument("--incognito")

            if self.config.disable_images:
                prefs = {"profile.managed_default_content_settings.images": 2}
                options.add_experimental_option("prefs", prefs)

            if self.config.disable_notifications:
                options.add_argument("--disable-notifications")

            if self.config.user_agent:
                options.add_argument(f"user-agent={self.config.user_agent}")

            if self.config.proxy:
                options.add_argument(f"--proxy-server={self.config.proxy}")

            # Set window size
            options.add_argument(f"--window-size={self.config.window_size[0]},{self.config.window_size[1]}")

            # Additional options
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)

            driver = webdriver.Chrome(options=options)
            logger.info("✅ Selenium WebDriver initialized")
            return driver

        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Selenium: {e}")
            return None

    def start_session(self) -> Optional[str]:
        """Start a new browser session"""
        try:
            session_id = f"session_{int(time.time())}"

            self.driver = self._initialize_selenium()

            self.current_session = BrowsingSession(
                session_id=session_id,
                browser_type=self.config.browser_type,
                start_time=datetime.now()
            )

            logger.info(f"✅ Session started: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"❌ Failed to start session: {e}")
            return None

    def end_session(self) -> bool:
        """End current browser session"""
        try:
            if self.driver:
                self.driver.quit()

            if self.current_session:
                self.current_session.end_time = datetime.now()
                self.current_session.total_duration = (
                    self.current_session.end_time - self.current_session.start_time
                ).total_seconds()
                self.session_history.append(self.current_session)

            logger.info("✅ Session ended")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to end session: {e}")
            return False

    # =========================
    # WEBSITE OPERATIONS
    # =========================

    def _format_url(self, url: str) -> str:
        """Format and validate URL"""
        url = url.strip()

        if not url:
            return ""

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        return url

    def open_website(self, url: str) -> bool:
        """Open website in browser"""
        try:
            formatted_url = self._format_url(url)

            if not formatted_url:
                logger.error("❌ Invalid URL")
                return False

            logger.info(f"🌐 Opening: {formatted_url}")

            if self.driver:
                self.driver.get(formatted_url)
                time.sleep(1)
            else:
                webbrowser.open(formatted_url)

            # Track visit
            visit = PageVisit(
                url=formatted_url,
                title=self.get_page_title() if self.driver else "Unknown",
                timestamp=datetime.now(),
                duration=0
            )

            self.page_visits.append(visit)
            if self.current_session:
                self.current_session.pages_visited += 1

            logger.info(f"✅ Website opened: {formatted_url}")

            # Trigger callbacks
            self._trigger_callbacks({"action": "open_website", "url": formatted_url})

            return True

        except Exception as e:
            logger.error(f"❌ Error opening website: {e}")
            return False

    # =========================
    # COMMON WEBSITES
    # =========================

    def open_youtube(self) -> bool:
        """Open YouTube"""
        return self.open_website("https://youtube.com")

    def open_google(self) -> bool:
        """Open Google"""
        return self.open_website("https://google.com")

    def open_github(self) -> bool:
        """Open GitHub"""
        return self.open_website("https://github.com")

    def open_whatsapp_web(self) -> bool:
        """Open WhatsApp Web"""
        return self.open_website("https://web.whatsapp.com")

    def open_instagram(self) -> bool:
        """Open Instagram"""
        return self.open_website("https://instagram.com")

    def open_twitter(self) -> bool:
        """Open Twitter/X"""
        return self.open_website("https://twitter.com")

    def open_linkedin(self) -> bool:
        """Open LinkedIn"""
        return self.open_website("https://linkedin.com")

    def open_facebook(self) -> bool:
        """Open Facebook"""
        return self.open_website("https://facebook.com")

    def open_gmail(self) -> bool:
        """Open Gmail"""
        return self.open_website("https://mail.google.com")

    def open_stackoverflow(self) -> bool:
        """Open Stack Overflow"""
        return self.open_website("https://stackoverflow.com")

    # =========================
    # SEARCH OPERATIONS
    # =========================

    def search(self, query: str, engine: SearchEngine = SearchEngine.GOOGLE) -> bool:
        """Search using specified search engine"""
        try:
            if not query:
                logger.error("❌ No search query provided")
                return False

            encoded_query = urllib.parse.quote(query)

            search_urls = {
                SearchEngine.GOOGLE: f"https://www.google.com/search?q={encoded_query}",
                SearchEngine.YOUTUBE: f"https://www.youtube.com/results?search_query={encoded_query}",
                SearchEngine.BING: f"https://www.bing.com/search?q={encoded_query}",
                SearchEngine.WIKIPEDIA: f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}",
                SearchEngine.GITHUB: f"https://github.com/search?q={encoded_query}",
                SearchEngine.STACKOVERFLOW: f"https://stackoverflow.com/search?q={encoded_query}"
            }

            url = search_urls.get(engine, search_urls[SearchEngine.GOOGLE])

            logger.info(f"🔍 Searching on {engine.value}: {query}")

            return self.open_website(url)

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return False

    def google_search(self, query: str) -> bool:
        """Google search"""
        return self.search(query, SearchEngine.GOOGLE)

    def youtube_search(self, query: str) -> bool:
        """YouTube search"""
        return self.search(query, SearchEngine.YOUTUBE)

    def wikipedia_search(self, query: str) -> bool:
        """Wikipedia search"""
        return self.search(query, SearchEngine.WIKIPEDIA)

    def github_search(self, query: str) -> bool:
        """GitHub search"""
        return self.search(query, SearchEngine.GITHUB)

    def stackoverflow_search(self, query: str) -> bool:
        """Stack Overflow search"""
        return self.search(query, SearchEngine.STACKOVERFLOW)

    # =========================
    # TAB MANAGEMENT
    # =========================

    def open_new_tab(self, url: Optional[str] = None) -> bool:
        """Open new tab"""
        try:
            if not self.driver:
                logger.error("❌ Browser session not active")
                return False

            self.driver.execute_script("window.open('');")
            self.driver.switch_to.window(self.driver.window_handles[-1])

            if url:
                self.open_website(url)

            self.tabs.append(url or "about:blank")
            if self.current_session:
                self.current_session.tabs_opened += 1

            logger.info("✅ New tab opened")
            return True

        except Exception as e:
            logger.error(f"❌ Error opening tab: {e}")
            return False

    def close_tab(self) -> bool:
        """Close current tab"""
        try:
            if not self.driver or len(self.driver.window_handles) <= 1:
                logger.warning("⚠️ No tab to close")
                return False

            if self.tabs:
                self.tabs.pop()

            self.driver.close()
            self.driver.switch_to.window(self.driver.window_handles[-1])

            logger.info("✅ Tab closed")
            return True

        except Exception as e:
            logger.error(f"❌ Error closing tab: {e}")
            return False

    def switch_to_tab(self, tab_index: int) -> bool:
        """Switch to specific tab"""
        try:
            if not self.driver or tab_index >= len(self.driver.window_handles):
                logger.error("❌ Invalid tab index")
                return False

            self.driver.switch_to.window(self.driver.window_handles[tab_index])
            logger.info(f"✅ Switched to tab {tab_index}")
            return True

        except Exception as e:
            logger.error(f"❌ Error switching tab: {e}")
            return False

    def get_open_tabs(self) -> List[str]:
        """Get list of open tabs"""
        if not self.driver:
            return []

        return [
            self.driver.get_window_handle()
            for _ in self.driver.window_handles
        ]

    # =========================
    # MULTIPLE TABS
    # =========================

    def open_multiple_websites(self, urls: List[str], delay: float = 0.5) -> bool:
        """Open multiple websites in separate tabs"""
        try:
            for i, url in enumerate(urls):
                if i == 0:
                    self.open_website(url)
                else:
                    self.open_new_tab(url)

                time.sleep(delay)

            logger.info(f"✅ Opened {len(urls)} websites")
            return True

        except Exception as e:
            logger.error(f"❌ Error opening multiple websites: {e}")
            return False

    def open_multiple_searches(
        self,
        queries: List[str],
        engine: SearchEngine = SearchEngine.GOOGLE,
        delay: float = 0.5
    ) -> bool:
        """Open multiple searches in tabs"""
        try:
            for i, query in enumerate(queries):
                if i == 0:
                    self.search(query, engine)
                else:
                    self.open_new_tab()
                    self.search(query, engine)

                time.sleep(delay)

            logger.info(f"✅ Opened {len(queries)} searches")
            return True

        except Exception as e:
            logger.error(f"❌ Error opening multiple searches: {e}")
            return False

    # =========================
    # PAGE INTERACTION
    # =========================

    def get_page_title(self) -> Optional[str]:
        """Get current page title"""
        try:
            if not self.driver:
                return None

            return self.driver.title

        except Exception as e:
            logger.warning(f"⚠️ Could not get page title: {e}")
            return None

    def get_page_url(self) -> Optional[str]:
        """Get current page URL"""
        try:
            if not self.driver:
                return None

            return self.driver.current_url

        except Exception as e:
            logger.warning(f"⚠️ Could not get page URL: {e}")
            return None

    def go_back(self) -> bool:
        """Go back in history"""
        try:
            if not self.driver:
                return False

            self.driver.back()
            logger.info("✅ Went back in history")
            return True

        except Exception as e:
            logger.error(f"❌ Error going back: {e}")
            return False

    def go_forward(self) -> bool:
        """Go forward in history"""
        try:
            if not self.driver:
                return False

            self.driver.forward()
            logger.info("✅ Went forward in history")
            return True

        except Exception as e:
            logger.error(f"❌ Error going forward: {e}")
            return False

    def refresh(self) -> bool:
        """Refresh current page"""
        try:
            if not self.driver:
                return False

            self.driver.refresh()
            logger.info("✅ Page refreshed")
            return True

        except Exception as e:
            logger.error(f"❌ Error refreshing page: {e}")
            return False

    # =========================
    # COOKIES & STORAGE
    # =========================

    def save_cookies(self, filename: str) -> bool:
        """Save cookies to file"""
        try:
            if not self.driver:
                return False

            cookies = self.driver.get_cookies()

            with open(filename, 'w') as f:
                json.dump(cookies, f)

            logger.info(f"✅ Cookies saved to: {filename}")
            return True

        except Exception as e:
            logger.error(f"❌ Error saving cookies: {e}")
            return False

    def load_cookies(self, filename: str) -> bool:
        """Load cookies from file"""
        try:
            if not self.driver or not os.path.exists(filename):
                return False

            with open(filename, 'r') as f:
                cookies = json.load(f)

            for cookie in cookies:
                try:
                    self.driver.add_cookie(cookie)
                except:
                    pass

            logger.info(f"✅ Cookies loaded from: {filename}")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading cookies: {e}")
            return False

    def clear_cookies(self) -> bool:
        """Clear all cookies"""
        try:
            if not self.driver:
                return False

            self.driver.delete_all_cookies()
            logger.info("✅ Cookies cleared")
            return True

        except Exception as e:
            logger.error(f"❌ Error clearing cookies: {e}")
            return False

    # =========================
    # SCREENSHOTS & EXPORTS
    # =========================

    def take_screenshot(self, filename: Optional[str] = None) -> Optional[str]:
        """Take screenshot of current page"""
        try:
            if not self.driver:
                logger.error("❌ No active browser session")
                return None

            if not filename:
                filename = f"screenshot_{int(time.time())}.png"

            self.driver.save_screenshot(filename)

            logger.info(f"✅ Screenshot saved: {filename}")
            return filename

        except Exception as e:
            logger.error(f"❌ Error taking screenshot: {e}")
            return None

    def get_page_source(self) -> Optional[str]:
        """Get current page HTML source"""
        try:
            if not self.driver:
                return None

            return self.driver.page_source

        except Exception as e:
            logger.warning(f"⚠️ Could not get page source: {e}")
            return None

    # =========================
    # CALLBACKS
    # =========================

    def add_callback(self, callback: Callable) -> None:
        """Add callback for browser events"""
        self.callbacks.append(callback)
        logger.info(f"✅ Callback added: {callback.__name__}")

    def _trigger_callbacks(self, data: Dict[str, Any]) -> None:
        """Trigger all registered callbacks"""
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.warning(f"⚠️ Callback error: {e}")

    # =========================
    # HISTORY & STATISTICS
    # =========================

    def get_visit_history(self, limit: int = 10) -> List[Dict]:
        """Get page visit history"""
        history = []
        for visit in self.page_visits[-limit:]:
            history.append({
                "url": visit.url,
                "title": visit.title,
                "timestamp": visit.timestamp.strftime("%H:%M:%S"),
                "duration": f"{visit.duration:.2f}s"
            })
        return history

    def get_stats(self) -> Dict[str, Any]:
        """Get browser statistics"""
        return {
            "total_pages_visited": len(self.page_visits),
            "total_tabs_opened": len(self.tabs),
            "sessions_count": len(self.session_history),
            "uptime": str(datetime.now() - self.start_time)
        }

    def print_stats(self) -> None:
        """Print statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("📊 BROWSER STATISTICS")
        print("="*50)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title():.<35} {value}")
        print("="*50 + "\n")

    def print_history(self, limit: int = 10) -> None:
        """Print visit history"""
        history = self.get_visit_history(limit)
        print("\n" + "="*50)
        print(f"📜 BROWSING HISTORY (Last {limit})")
        print("="*50)
        for i, item in enumerate(history, 1):
            print(f"\n{i}. {item['title']}")
            print(f"   URL: {item['url']}")
            print(f"   Time: {item['timestamp']}")
        print("="*50 + "\n")

    def export_history(self, filename: str) -> None:
        """Export history to JSON"""
        history = self.get_visit_history(len(self.page_visits))
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ History exported to: {filename}")


# =========================
# QUICK FUNCTIONS
# =========================

def quick_open(url: str) -> bool:
    """Quick function to open URL"""
    manager = AdvancedBrowserManager()
    return manager.open_website(url)


def quick_search(query: str, engine: SearchEngine = SearchEngine.GOOGLE) -> bool:
    """Quick function to search"""
    manager = AdvancedBrowserManager()
    return manager.search(query, engine)


# =========================
# TESTING & DEMONSTRATION
# =========================

def main():
    """Comprehensive test of the browser manager"""

    print("\n" + "="*50)
    print("🌐 ADVANCED BROWSER MANAGER")
    print("="*50 + "\n")

    # Initialize browser with Selenium if available
    config = BrowserConfig(
        browser_type=BrowserType.CHROME,
        headless=False,
        window_size=(1920, 1080)
    )

    manager = AdvancedBrowserManager(config)

    # Add callback
    def on_browser_event(data):
        logger.info(f"Browser event: {data}")

    manager.add_callback(on_browser_event)

    # Test 1: Start session
    print("📝 Test 1: Starting Browser Session")
    if SELENIUM_AVAILABLE:
        manager.start_session()
        time.sleep(2)

        # Test 2: Open websites
        print("\n📝 Test 2: Opening Websites")
        manager.open_youtube()
        time.sleep(2)

        manager.open_google()
        time.sleep(2)

        # Test 3: Multiple searches
        print("\n📝 Test 3: Multiple Searches")
        queries = ["python programming", "web development", "AI tutorial"]
        manager.open_multiple_searches(queries, SearchEngine.YOUTUBE)
        time.sleep(2)

        # Test 4: Get page info
        print("\n📝 Test 4: Page Information")
        print(f"Title: {manager.get_page_title()}")
        print(f"URL: {manager.get_page_url()}")

        # Test 5: Take screenshot
        print("\n📝 Test 5: Take Screenshot")
        screenshot = manager.take_screenshot()
        print(f"Screenshot: {screenshot}")

        # Test 6: Refresh
        print("\n📝 Test 6: Refresh Page")
        manager.refresh()
        time.sleep(1)

        # Test 7: Navigate
        print("\n📝 Test 7: Navigation")
        manager.go_back()
        time.sleep(1)

        # End session
        manager.end_session()
    else:
        # Fallback to webbrowser
        print("⚠️ Selenium not available, using webbrowser fallback")
        manager.open_website("https://youtube.com")
        manager.google_search("python")

    # Show statistics
    manager.print_stats()
    manager.print_history(limit=5)

    # Export history
    manager.export_history("browser_history.json")

    print("✅ All tests complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Program interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")


# =========================
# SIMPLE FUNCTION WRAPPERS
# =========================

_browser = AdvancedBrowserManager()

def open_youtube():
    _browser.open_youtube()

def open_google():
    _browser.open_google()