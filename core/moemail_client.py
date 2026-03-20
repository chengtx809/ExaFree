"""
Moemail临时邮箱客户端

API文档参考:
- 获取系统配置: GET /api/config
- 生成临时邮箱: POST /api/emails/generate
- 获取邮件列表: GET /api/emails/{emailId}
- 获取单封邮件: GET /api/emails/{emailId}/{messageId}
"""

import random
import string
import time
from typing import Optional

import requests

from core.mail_utils import extract_verification_code
from core.proxy_utils import request_with_proxy_fallback


class MoemailClient:
    """Moemail临时邮箱客户端"""

    def __init__(
        self,
        base_url: str = "https://moemail.nanohajimi.mom",
        proxy: str = "",
        api_key: str = "",
        domain: str = "",
        log_callback=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.proxies = {"http": proxy, "https": proxy} if proxy else None
        self.api_key = api_key.strip()
        self.domain = domain.strip() if domain else ""
        self.log_callback = log_callback

        self.email: Optional[str] = None
        self.email_id: Optional[str] = None
        self.password: Optional[str] = None  # 兼容 DuckMailClient 接口

        # 缓存可用域名列表
        self._available_domains: list = []

    def set_credentials(self, email: str, password: str = "") -> None:
        """设置凭据（兼容 DuckMailClient 接口）"""
        self.email = email
        self.password = password

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """发送请求并打印详细日志"""
        headers = kwargs.pop("headers", None) or {}
        if self.api_key and "X-API-Key" not in headers:
            headers["X-API-Key"] = self.api_key
        headers.setdefault("Content-Type", "application/json")
        kwargs["headers"] = headers

        self._log("info", f"📤 发送 {method} 请求: {url}")
        if "json" in kwargs:
            self._log("info", f"📦 请求体: {kwargs['json']}")

        try:
            res = request_with_proxy_fallback(
                requests.request,
                method,
                url,
                proxies=self.proxies,
                timeout=kwargs.pop("timeout", 30),
                **kwargs,
            )
            self._log("info", f"📥 收到响应: HTTP {res.status_code}")
            if res.content and res.status_code >= 400:
                try:
                    self._log("error", f"📄 响应内容: {res.text[:500]}")
                except Exception:
                    pass
            return res
        except Exception as e:
            self._log("error", f"❌ 网络请求失败: {e}")
            raise

    def _get_available_domains(self) -> list:
        """获取可用的邮箱域名列表"""
        if self._available_domains:
            return self._available_domains

        try:
            res = self._request("GET", f"{self.base_url}/api/config")
            if res.status_code == 200:
                data = res.json()
                email_domains_str = data.get("emailDomains", "")
                if email_domains_str:
                    self._available_domains = [d.strip() for d in email_domains_str.split(",") if d.strip()]
                    self._log("info", f"🌐 Moemail 可用域名: {self._available_domains}")
                    return self._available_domains
        except Exception as e:
            self._log("error", f"❌ 获取可用域名失败: {e}")

        # 默认域名
        self._available_domains = ["moemail.app"]
        return self._available_domains

    def register_account(self, domain: Optional[str] = None) -> bool:
        """注册新邮箱账号

        API: POST /api/emails/generate
        """
        # 确定使用的域名
        selected_domain = domain
        if not selected_domain:
            selected_domain = self.domain

        if not selected_domain:
            # 从可用域名中随机选择
            available = self._get_available_domains()
            if available:
                selected_domain = random.choice(available)
            else:
                selected_domain = "moemail.app"

        self._log("info", f"📧 使用域名: {selected_domain}")

        # 生成随机邮箱名称
        rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        timestamp = str(int(time.time()))[-4:]
        name = f"t{timestamp}{rand}"

        self._log("info", f"🎲 生成邮箱: {name}@{selected_domain}")

        try:
            # 设置为 0 表示永久有效
            self._log("info", f"⏰ 设置过期时间: 永久有效")

            res = self._request(
                "POST",
                f"{self.base_url}/api/emails/generate",
                json={
                    "name": name,
                    "expiryTime": 0,
                    "domain": selected_domain,
                },
            )

            if res.status_code in (200, 201):
                data = res.json() if res.content else {}
                self.email = data.get("email", "")
                self.email_id = data.get("id", "")
                self.password = self.email_id  # 用 email_id 作为 password 存储

                if self.email and self.email_id:
                    self._log("info", f"✅ Moemail 注册成功: {self.email}")
                    self._log("info", f"🔑 Email ID: {self.email_id}")
                    return True

            self._log("error", f"❌ Moemail 注册失败: HTTP {res.status_code}")
            if res.content:
                self._log("error", f"📄 响应内容: {res.text[:500]}")
            return False

        except Exception as e:
            self._log("error", f"❌ Moemail 注册异常: {e}")
            return False

    def login(self) -> bool:
        """登录（Moemail 无需登录，返回 True）"""
        # Moemail 使用 API Key 认证，无需单独登录
        return True

    def fetch_verification_code(self, since_time=None) -> Optional[str]:
        """获取验证码

        API: GET /api/emails/{emailId}
        API: GET /api/emails/{emailId}/{messageId}
        """
        if not self.email_id:
            self._log("error", "❌ 缺少 email_id，无法获取邮件")
            return None

        try:
            self._log("info", "📬 正在拉取 Moemail 邮件列表...")

            # 获取邮件列表
            res = self._request(
                "GET",
                f"{self.base_url}/api/emails/{self.email_id}",
            )

            if res.status_code != 200:
                self._log("error", f"❌ 获取邮件列表失败: HTTP {res.status_code}")
                return None

            data = res.json() if res.content else {}
            messages = data.get("messages", [])

            if not messages:
                self._log("info", "📭 邮箱为空，暂无邮件")
                return None

            self._log("info", f"📨 收到 {len(messages)} 封邮件，开始检查验证码...")

            from datetime import datetime

            def _parse_message_time(msg_obj) -> Optional[datetime]:
                import re

                time_keys = [
                    "createdAt",
                    "receivedAt",
                    "sentAt",
                    "created_at",
                    "received_at",
                    "sent_at",
                ]
                raw_time = None
                for key in time_keys:
                    if msg_obj.get(key) is not None:
                        raw_time = msg_obj.get(key)
                        break

                if raw_time is None:
                    return None

                if isinstance(raw_time, (int, float)):
                    timestamp = float(raw_time)
                    if timestamp > 1e12:
                        timestamp = timestamp / 1000.0
                    return datetime.fromtimestamp(timestamp)

                if isinstance(raw_time, str):
                    raw_time = raw_time.strip()
                    if raw_time.isdigit():
                        timestamp = float(raw_time)
                        if timestamp > 1e12:
                            timestamp = timestamp / 1000.0
                        return datetime.fromtimestamp(timestamp)

                    # 处理 ISO 时间字符串
                    try:
                        # 截断纳秒到微秒
                        raw_time = re.sub(r"(\.\d{6})\d+", r"\1", raw_time)
                        return datetime.fromisoformat(raw_time.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
                    except Exception:
                        return None

                return None

            def _looks_like_verification(msg_obj) -> bool:
                subject = (msg_obj.get("subject") or "").strip()
                if not subject:
                    return False
                import re
                return re.search(r"(验证码|验证|verification|verify|passcode|security\s*code|one[-\s]?time|otp)", subject, re.IGNORECASE) is not None

            messages_with_time = [(msg, _parse_message_time(msg)) for msg in messages]
            if any(item[1] for item in messages_with_time):
                messages_with_time.sort(key=lambda item: item[1] or datetime.min, reverse=True)
                messages = [item[0] for item in messages_with_time]

            # 按验证码相关性排序：subject 匹配验证关键词的排前面
            messages = sorted(messages, key=lambda m: (0 if _looks_like_verification(m) else 1))

            # 遍历邮件
            for idx, msg in enumerate(messages, 1):
                msg_id = msg.get("id")
                if not msg_id:
                    continue

                # 时间过滤
                if since_time:
                    msg_time = _parse_message_time(msg)
                    if msg_time and msg_time < since_time:
                        continue

                # 优先从邮件列表的 content 字段提取验证码（更高效）
                list_content = msg.get("content") or ""
                if list_content:
                    code = extract_verification_code(list_content)
                    if code:
                        self._log("info", f"✅ 找到验证码: {code}")
                        return code

                # 如果列表没有 content，则获取邮件详情
                self._log("info", f"🔍 正在读取邮件 {idx}/{len(messages)} 详情...")
                detail_res = self._request(
                    "GET",
                    f"{self.base_url}/api/emails/{self.email_id}/{msg_id}",
                )

                if detail_res.status_code != 200:
                    self._log("warning", f"⚠️ 读取邮件详情失败: HTTP {detail_res.status_code}")
                    continue

                detail = detail_res.json() if detail_res.content else {}

                # 处理 {'message': {...}} 格式
                if "message" in detail and isinstance(detail["message"], dict):
                    detail = detail["message"]

                # 获取邮件内容
                text_content = detail.get("text") or detail.get("textContent") or detail.get("content") or ""
                html_content = detail.get("html") or detail.get("htmlContent") or ""

                if isinstance(html_content, list):
                    html_content = "".join(str(item) for item in html_content)
                if isinstance(text_content, list):
                    text_content = "".join(str(item) for item in text_content)

                content = text_content + html_content
                if content:
                    code = extract_verification_code(content)
                    if code:
                        self._log("info", f"✅ 找到验证码: {code}")
                        return code
                    else:
                        self._log("info", f"❌ 邮件 {idx} 中未找到验证码")

            self._log("warning", "⚠️ 所有邮件中均未找到验证码")
            return None

        except Exception as e:
            self._log("error", f"❌ 获取验证码异常: {e}")
            return None

    def poll_for_code(
        self,
        timeout: int = 120,
        interval: int = 4,
        since_time=None,
    ) -> Optional[str]:
        """轮询获取验证码"""
        max_retries = max(1, timeout // interval)
        self._log("info", f"⏱️ 开始轮询验证码 (超时 {timeout}秒, 间隔 {interval}秒, 最多 {max_retries} 次)")

        for i in range(1, max_retries + 1):
            self._log("info", f"🔄 第 {i}/{max_retries} 次轮询...")
            code = self.fetch_verification_code(since_time=since_time)
            if code:
                self._log("info", f"🎉 验证码获取成功: {code}")
                return code

            if i < max_retries:
                self._log("info", f"⏳ 等待 {interval} 秒后重试...")
                time.sleep(interval)

        self._log("error", f"⏰ 验证码获取超时 ({timeout}秒)")
        return None

    def _log(self, level: str, message: str) -> None:
        if self.log_callback:
            try:
                self.log_callback(level, message)
            except Exception:
                pass
