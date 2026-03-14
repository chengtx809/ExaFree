"""
Exa 自动化登录与 API Key 提取。

流程：
1. 通过邮箱验证码登录 auth.exa.ai
2. 完成 onboarding（若存在）
3. 在 billing 页面兑换优惠码
4. 在 API Keys 页面创建 key 并提取
"""

from __future__ import annotations

import random
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover - runtime dependency check
    sync_playwright = None
    PlaywrightTimeoutError = Exception


UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    flags=re.IGNORECASE,
)


class ExaAutomation:
    """Exa 自动化流程封装（同步，适合 run_in_executor 调用）。"""

    def __init__(
        self,
        proxy: str = "",
        timeout_ms: int = 90_000,
        log_callback=None,
    ) -> None:
        self.proxy = (proxy or "").strip()
        self.timeout_ms = timeout_ms
        self.log_callback = log_callback

    def register_and_setup(
        self,
        email: str,
        mail_client,
        coupon_code: str = "",
        redeem_coupon: bool = False,
    ) -> Dict[str, Any]:
        """
        执行 Exa 登录 + 初始化流程，返回可落库配置。
        """
        if sync_playwright is None:
            return {
                "success": False,
                "error": "playwright 未安装，请先安装 playwright 并执行 playwright install chromium",
            }

        start_time = datetime.now()
        self._log("info", f"🌐 打开 Exa 登录页: {email}")

        try:
            with sync_playwright() as p:
                launch_kwargs: Dict[str, Any] = {
                    "headless": True,
                }
                if self.proxy:
                    launch_kwargs["proxy"] = {"server": self.proxy}
                launch_kwargs["args"] = ["--no-sandbox", "--disable-dev-shm-usage"]

                browser = p.chromium.launch(**launch_kwargs)
                context = browser.new_context()
                page = context.new_page()

                try:
                    self._login_with_otp(page, email, mail_client, start_time)
                    onboarding_key = self._complete_onboarding(page)

                    balance = None
                    coupon_status = "not_attempted"
                    if redeem_coupon:
                        balance, coupon_status = self._redeem_coupon(page, coupon_code)

                    created_api_key = self._create_api_key(page)
                    account_config = self._build_account_config(
                        email=email,
                        api_key=created_api_key,
                        coupon_status=coupon_status,
                        balance=balance,
                    )

                    return {
                        "success": True,
                        "config": account_config,
                        "created_api_key": created_api_key,
                        "onboarding_api_key": onboarding_key,
                        "coupon_status": coupon_status,
                        "balance": balance,
                    }
                finally:
                    try:
                        context.close()
                    except Exception:
                        pass
                    try:
                        browser.close()
                    except Exception:
                        pass
        except Exception as exc:
            self._log("error", f"❌ Exa 自动化失败: {exc}")
            return {"success": False, "error": str(exc)}

    def refresh_api_key(
        self,
        email: str,
        mail_client,
    ) -> Dict[str, Any]:
        """刷新账号 key（登录 + 重新创建 key，不重复兑换优惠码）。"""
        return self.register_and_setup(
            email=email,
            mail_client=mail_client,
            coupon_code="",
            redeem_coupon=False,
        )

    def _login_with_otp(self, page, email: str, mail_client, start_time: datetime) -> None:
        auth_url = "https://auth.exa.ai/?callbackUrl=https%3A%2F%2Fdashboard.exa.ai%2F"
        page.goto(auth_url, wait_until="domcontentloaded", timeout=self.timeout_ms)

        page.wait_for_selector('input[placeholder="Email"]', timeout=60_000)
        page.fill('input[placeholder="Email"]', email)
        page.locator('form:has(input[placeholder="Email"]) button[type="submit"]').first.click()

        page.wait_for_selector('text="Verify your email"', timeout=60_000)
        self._log("info", "📬 等待验证码邮件...")
        code = mail_client.poll_for_code(timeout=240, interval=3, since_time=start_time)
        if not code:
            raise RuntimeError("未收到 Exa OTP 验证码")
        self._log("info", f"✅ 收到 OTP: {code}")

        page.fill('input[placeholder="Enter verification code"]', code)
        page.locator('button:has-text("VERIFY CODE")').first.click()

        invalid_tip = page.locator('text="Invalid verification code."')
        # 某些会话为 SPA 跳转，不触发 commit/load，直接等 wait_for_url 会卡住到超时。
        # 这里改为短轮询 URL，尽快识别是否已进入 dashboard 域名。
        entered_dashboard = False
        otp_wait_start = time.time()
        page.wait_for_timeout(700)
        deadline = time.time() + 22.0
        while time.time() < deadline:
            current_url = page.url
            if self._get_url_host(current_url) == "dashboard.exa.ai":
                entered_dashboard = True
                break
            if invalid_tip.count() and invalid_tip.first.is_visible():
                raise RuntimeError("OTP 无效，Exa 返回 Invalid verification code")
            try:
                page.wait_for_load_state("domcontentloaded", timeout=1200)
            except Exception:
                pass
            page.wait_for_timeout(300)

        if not entered_dashboard:
            current_url = page.url
            if invalid_tip.count() and invalid_tip.first.is_visible():
                raise RuntimeError("OTP 无效，Exa 返回 Invalid verification code")
            # 若仍停留在 auth.exa.ai，尝试手动打开 dashboard 触发会话写入
            if self._get_url_host(current_url) == "auth.exa.ai":
                self._log("warning", "⚠️ OTP 后未自动跳转 Dashboard，尝试手动打开...")
                self._safe_goto(
                    page,
                    "https://dashboard.exa.ai/",
                    wait_until="domcontentloaded",
                    timeout=self.timeout_ms,
                    retries=2,
                )
                if self._get_url_host(page.url) == "dashboard.exa.ai":
                    entered_dashboard = True
            if not entered_dashboard:
                raise RuntimeError(f"OTP 提交后未进入 Exa Dashboard，当前页面: {current_url}")

        otp_wait_cost = time.time() - otp_wait_start
        if otp_wait_cost > 8:
            self._log("warning", f"⚠️ OTP 跳转耗时较长: {otp_wait_cost:.1f}s，当前页面: {page.url}")

        if invalid_tip.count() and invalid_tip.first.is_visible():
            raise RuntimeError("OTP 无效，Exa 返回 Invalid verification code")

        self._log("info", f"✅ OTP 提交后已进入: {page.url}")

    def _complete_onboarding(self, page) -> Optional[str]:
        onboarding_key = None
        self._safe_goto(
            page,
            "https://dashboard.exa.ai/onboarding",
            wait_until="domcontentloaded",
            timeout=self.timeout_ms,
            retries=2,
        )
        page.wait_for_timeout(1200)

        if "onboarding" not in page.url:
            return None

        next_selectors = [
            'button:has-text("Next")',
            'button:has-text("Continue")',
            'button:has-text("Continue to Step 2")',
            'button:has-text("Proceed")',
        ]

        # Step 1 需要完成三组选择才能点亮 Next：
        # 1) coding with   2) API client   3) building
        # 按旧流程优先选择 Codex / Python / Coding agent，并做重试。
        step1_choice_groups = [
            ["Codex", "Cursor", "Claude", "Devin", "Other"],
            ["Python", "OpenAI SDK", "JavaScript", "cURL", "MCP", "Other"],
            ["Coding agent", "Coding Agent", "Web search tool", "News monitoring", "E-commerce", "People + Company search", "Other"],
        ]

        step1_deadline = time.time() + 15.0
        while time.time() < step1_deadline:
            for labels in step1_choice_groups:
                selectors = [f'button:has-text("{label}")' for label in labels]
                if self._click_any_visible(page, selectors):
                    page.wait_for_timeout(250)

            next_btn = self._first_visible_locator(page, next_selectors)
            if next_btn is not None and next_btn.is_enabled():
                next_btn.click()
                page.wait_for_timeout(1200)
                break

            page.wait_for_timeout(350)

        # Step 2（生成代码）使用多文案兼容，直到拿到 key 或超时。
        generate_selectors = [
            'button:has-text("Generate Code")',
            'button:has-text("Generate")',
            'button:has-text("Generate API Key")',
            'button:has-text("Create Code")',
        ]
        generate_deadline = time.time() + 18.0
        while time.time() < generate_deadline and onboarding_key is None and "onboarding" in page.url:
            if self._click_any_visible(page, generate_selectors):
                page.wait_for_timeout(2200)

            onboarding_key = self._extract_first_uuid(page.inner_text("body"))
            if onboarding_key:
                break

            # 如果仍在 step1，继续尝试推进到下一步
            next_btn = self._first_visible_locator(page, next_selectors)
            if next_btn is not None and next_btn.is_enabled():
                next_btn.click()
                page.wait_for_timeout(1000)
            else:
                page.wait_for_timeout(400)

        go_dashboard_selectors = [
            'button:has-text("Go to Dashboard")',
            'button:has-text("Continue to Dashboard")',
            'button:has-text("Back to Dashboard")',
            'button:has-text("Open Dashboard")',
            'a:has-text("Go to Dashboard")',
            'a:has-text("Continue to Dashboard")',
        ]
        exit_deadline = time.time() + 10.0
        while time.time() < exit_deadline and "onboarding" in page.url:
            if self._click_any_visible(page, go_dashboard_selectors):
                page.wait_for_timeout(900)
                continue
            page.wait_for_timeout(300)

        onboarding_completed = self._is_onboarding_completed(page)

        # Exa 新版完成态可能仍保留在 /onboarding 页面，此时主动跳往 API Keys，
        # 避免仅因 URL 未变化而将已完成的新手引导误判为失败。
        if onboarding_completed and "onboarding" in page.url:
            self._log("info", "✅ 检测到 onboarding 已完成，继续跳转 API Keys 页面")
            try:
                self._safe_goto(
                    page,
                    "https://dashboard.exa.ai/api-keys",
                    wait_until="domcontentloaded",
                    timeout=self.timeout_ms,
                    retries=1,
                )
                page.wait_for_timeout(900)
            except Exception as exc:
                self._log("warning", f"⚠️ onboarding 完成后跳转 API Keys 失败，将继续后续流程: {exc}")

        # 只有在既没有完成信号、又仍停留在 onboarding 时，才判定为真正失败。
        if "dashboard.exa.ai" in page.url and "onboarding" in page.url and not onboarding_completed:
            raise RuntimeError(f"onboarding 未完成，仍停留在: {page.url}")

        if not onboarding_key:
            self._log("warning", "⚠️ onboarding 未提取到生成 key，可能未触发完整新手引导奖励")

        return onboarding_key

    def _redeem_coupon(self, page, coupon_code: str) -> tuple[Optional[str], str]:
        self._safe_goto(
            page,
            "https://dashboard.exa.ai/billing",
            wait_until="domcontentloaded",
            timeout=self.timeout_ms,
            retries=1,
        )
        page.wait_for_timeout(1200)

        def read_balance_with_retry(timeout_sec: float = 12.0) -> Optional[str]:
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                text = page.inner_text("body")
                bal = self._extract_balance(text)
                if bal:
                    return bal
                page.wait_for_timeout(400)
            return None

        coupon_status = "not_attempted"
        balance_before = read_balance_with_retry()

        # 先尝试展开优惠码输入区域，兼容折叠/弹窗样式。
        coupon_expand_selectors = [
            'button:has-text("Have a coupon")',
            'button:has-text("Add coupon")',
            'button:has-text("Add Coupon")',
            'button:has-text("Promo code")',
            'button:has-text("Coupon code")',
            'button:has-text("Redeem code")',
        ]
        self._click_any_visible(page, coupon_expand_selectors)
        page.wait_for_timeout(300)

        coupon_input_selectors = [
            'input[placeholder="Enter coupon code"]',
            'input[placeholder*="coupon" i]',
            'input[placeholder*="promo" i]',
            'input[aria-label*="coupon" i]',
            'input[aria-label*="promo" i]',
            'input[name*="coupon" i]',
            'input[name*="promo" i]',
            'input[id*="coupon" i]',
            'input[id*="promo" i]',
        ]
        coupon_input = self._first_visible_locator(page, coupon_input_selectors)

        if coupon_input is not None:
            coupon_input.fill(coupon_code)
            page.wait_for_timeout(250)

            redeem_btn_selectors = [
                'button:has-text("Redeem")',
                'button:has-text("Apply")',
                'button:has-text("Apply Code")',
                'button:has-text("Apply Coupon")',
                'button:has-text("Use Code")',
            ]
            redeem_btn = self._first_visible_locator(page, redeem_btn_selectors)
            if redeem_btn is not None and redeem_btn.is_enabled():
                redeem_btn.click()
                coupon_status = "submitted"
                page.wait_for_timeout(3400)
            else:
                try:
                    coupon_input.press("Enter")
                    coupon_status = "submitted"
                    page.wait_for_timeout(3400)
                except Exception:
                    coupon_status = "redeem_disabled"
        else:
            coupon_status = "coupon_input_not_found"

        body = page.inner_text("body").lower()
        if (
            "successfully redeemed" in body
            or ("redeemed" in body and "coupon" in body)
            or ("applied" in body and ("coupon" in body or "promo" in body))
        ):
            coupon_status = "redeemed_successfully"
        elif ("already redeemed" in body) or ("already used" in body) or ("already" in body and "coupon" in body):
            coupon_status = "already_redeemed"
        elif ("invalid" in body and ("coupon" in body or "promo" in body)) or ("expired" in body and "coupon" in body):
            coupon_status = "invalid_coupon"

        balance = read_balance_with_retry() or balance_before
        self._log("info", f"🎟️ 优惠码状态: {coupon_status}")
        if balance_before:
            self._log("info", f"💰 兑换前余额: {balance_before}")
        if balance:
            self._log("info", f"💰 兑换后余额: {balance}")
        if balance_before and balance:
            try:
                before_num = float(balance_before.replace(",", ""))
                after_num = float(balance.replace(",", ""))
                delta = after_num - before_num
                self._log("info", f"📈 余额变化: {delta:+.2f}")
            except Exception:
                pass

        return balance, coupon_status

    def _create_api_key(self, page) -> str:
        self._safe_goto(
            page,
            "https://dashboard.exa.ai/api-keys",
            wait_until="domcontentloaded",
            timeout=self.timeout_ms,
            retries=1,
        )
        page.wait_for_timeout(1000)

        create_btn = page.locator('button:has-text("Create Key")').first
        if not create_btn.count() or not create_btn.is_visible():
            raise RuntimeError("未找到 Create Key 按钮")
        create_btn.click()
        page.wait_for_timeout(500)

        name_input = page.locator('input[placeholder="Project name"]').first
        if not name_input.count() or not name_input.is_visible():
            raise RuntimeError("未找到 API key 名称输入框")
        name_input.fill(f"pool-{int(time.time())}-{random.randint(100, 999)}")

        create_confirm = page.locator('button:has-text("Create a Key")').first
        if not create_confirm.count() or not create_confirm.is_enabled():
            raise RuntimeError("Create a Key 按钮不可用")
        create_confirm.click()

        key_input = page.locator("input[readonly]").first
        key_input.wait_for(timeout=15_000)
        key_value = key_input.input_value().strip()
        if not UUID_RE.fullmatch(key_value):
            raise RuntimeError(f"创建后的 Key 格式异常: {key_value[:24]}")

        self._click_if_visible(page, 'button:has-text("Done")') or self._click_if_visible(page, 'button:has-text("Close")')
        page.wait_for_timeout(300)
        masked = f"{key_value[:6]}...{key_value[-4:]}"
        self._log("info", f"🔑 已提取 API key: {masked}")
        return key_value

    def _build_account_config(
        self,
        email: str,
        api_key: str,
        coupon_status: str,
        balance: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "id": email,
            "exa_api_key": api_key,
            "coupon_status": coupon_status,
            "balance": balance,
            # 保留旧字段，兼容当前前端与账户加载逻辑
            "secure_c_ses": api_key,
            "host_c_oses": "",
            "csesidx": "exa",
            "config_id": "exa",
            "expires_at": None,
            "disabled": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _extract_first_uuid(text: str) -> Optional[str]:
        m = UUID_RE.search(text or "")
        return m.group(0) if m else None

    @staticmethod
    def _extract_balance(text: str) -> Optional[str]:
        m = re.search(r"Remaining Balance\s*\$([0-9][0-9,]*(?:\.[0-9]{2})?)", text or "", flags=re.I)
        return m.group(1) if m else None

    @staticmethod
    def _is_onboarding_completed(page) -> bool:
        body_text = (page.inner_text("body") or "").lower()
        completion_markers = [
            "you're all set",
            "you’re all set",
            "exa api setup guide",
            "manage keys at",
            "api keys",
            "your api key:",
            "mcp.exa.ai/mcp",
        ]
        return any(marker in body_text for marker in completion_markers)

    @staticmethod
    def _click_if_visible(page, selector: str) -> bool:
        loc = page.locator(selector).first
        if loc.count() and loc.is_visible() and loc.is_enabled():
            loc.click()
            return True
        return False

    @staticmethod
    def _click_any_visible(page, selectors) -> bool:
        for selector in selectors:
            if ExaAutomation._click_if_visible(page, selector):
                return True
        return False

    @staticmethod
    def _first_visible_locator(page, selectors):
        for selector in selectors:
            loc = page.locator(selector).first
            if loc.count() and loc.is_visible():
                return loc
        return None

    @staticmethod
    def _get_url_host(url: str) -> str:
        try:
            return urlparse(url).hostname or ""
        except Exception:
            return ""

    def _safe_goto(
        self,
        page,
        url: str,
        wait_until: str = "domcontentloaded",
        timeout: Optional[int] = None,
        retries: int = 1,
    ) -> None:
        last_exc = None
        effective_timeout = timeout or self.timeout_ms
        for attempt in range(retries + 1):
            try:
                page.goto(url, wait_until=wait_until, timeout=effective_timeout)
                return
            except Exception as exc:
                last_exc = exc
                if "net::ERR_ABORTED" not in str(exc) or attempt >= retries:
                    raise
                self._log("warning", f"⚠️ 页面跳转被中止，重试 {attempt + 1}/{retries}")
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=3000)
                except Exception:
                    pass
                page.wait_for_timeout(500 + attempt * 300)
        if last_exc:
            raise last_exc

    def _log(self, level: str, message: str) -> None:
        if not self.log_callback:
            return
        try:
            self.log_callback(level, message)
        except Exception:
            return
