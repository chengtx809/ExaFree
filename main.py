import json, time, os, asyncio, uuid, re, secrets
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from urllib.parse import urlencode, quote
from dotenv import load_dotenv
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Header, Request, Body, Form, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from collections import deque
from threading import Lock
from core.database import stats_db

# ---------- 数据目录配置 ----------
DATA_DIR = "./data"

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# 导入认证模块
from core.session_auth import (
    is_logged_in,
    login_user,
    logout_user,
    require_login,
    get_session_user,
    generate_session_secret,
)
from core.user_auth import (
    normalize_username,
    is_valid_username,
    is_valid_password,
    hash_password,
    verify_password,
    generate_api_key,
    hash_api_key,
    key_prefix,
)

# 导入核心模块
from core.account import (
    AccountManager,
    RetryPolicy,
    CooldownConfig,
    load_multi_account_config,
    load_accounts_from_source,
    update_accounts_config as _update_accounts_config,
    delete_account as _delete_account,
    update_account_disabled_status as _update_account_disabled_status,
    bulk_update_account_disabled_status as _bulk_update_account_disabled_status,
    bulk_delete_accounts as _bulk_delete_accounts
)
from core.proxy_utils import parse_proxy_setting
from core.exa_automation import ExaAutomation

# 导入 Uptime 追踪器
from core import uptime as uptime_tracker

# 导入配置管理和模板系统
from core.config import config_manager, config

# 数据库存储支持
from core import storage, account

# ---------- 日志配置 ----------

# 内存日志缓冲区 (保留最近 3000 条日志，重启后清空)
log_buffer = deque(maxlen=3000)
log_lock = Lock()

# 统计数据持久化
stats_lock = asyncio.Lock()  # 改为异步锁

async def load_stats():
    """加载统计数据（异步）。数据库不可用时使用内存默认值。"""
    data = None
    if storage.is_database_enabled():
        try:
            has_stats = await asyncio.to_thread(storage.has_stats_sync)
            if has_stats:
                data = await asyncio.to_thread(storage.load_stats_sync)
                if not isinstance(data, dict):
                    data = None
        except Exception as e:
            logger.error(f"[STATS] 数据库加载失败: {str(e)[:50]}")

    if data is None:
        data = {
            "total_visitors": 0,
            "total_requests": 0,
            "success_count": 0,
            "failed_count": 0,
            "request_timestamps": [],
            "model_request_timestamps": {},
            "failure_timestamps": [],
            "visitor_ips": {},
            "account_conversations": {},
            "account_failures": {},
            "recent_conversations": []
        }

    if isinstance(data.get("request_timestamps"), list):
        data["request_timestamps"] = deque(data["request_timestamps"], maxlen=20000)
    if isinstance(data.get("failure_timestamps"), list):
        data["failure_timestamps"] = deque(data["failure_timestamps"], maxlen=10000)

    return data

async def save_stats(stats):
    """保存统计数据(异步)。数据库不可用时不落盘。"""
    def convert_deques(obj):
        """递归转换所有 deque 对象为 list"""
        if isinstance(obj, deque):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_deques(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_deques(item) for item in obj]
        return obj

    stats_to_save = convert_deques(stats)

    if storage.is_database_enabled():
        try:
            saved = await asyncio.to_thread(storage.save_stats_sync, stats_to_save)
            if saved:
                return
        except Exception as e:
            logger.error(f"[STATS] 数据库保存失败: {str(e)[:50]}")
    return

# 初始化统计数据（需要在启动时异步加载）
global_stats = {
    "total_visitors": 0,
    "total_requests": 0,
    "success_count": 0,
    "failed_count": 0,
    "request_timestamps": deque(maxlen=20000),
    "model_request_timestamps": {},
    "failure_timestamps": deque(maxlen=10000),
    "visitor_ips": {},
    "account_conversations": {},
    "account_failures": {},
    "recent_conversations": []
}

# 任务历史记录（内存存储，容器重启后清空）
task_history = deque(maxlen=100)  # 最多保留100条历史记录
task_history_lock = Lock()

def get_beijing_time_str(ts: Optional[float] = None) -> str:
    tz = timezone(timedelta(hours=8))
    current = datetime.fromtimestamp(ts or time.time(), tz=tz)
    return current.strftime("%Y-%m-%d %H:%M:%S")


def save_task_to_history(task_type: str, task_data: dict) -> None:
    """保存任务历史记录（只存储简要信息）"""
    with task_history_lock:
        history_entry = _build_history_entry(task_type, task_data)
        entry_id = history_entry.get("id")
        if entry_id:
            for i in range(len(task_history) - 1, -1, -1):
                if task_history[i].get("id") == entry_id:
                    task_history.remove(task_history[i])
                    break
        task_history.append(history_entry)
        _persist_task_history()
        logger.info(f"[HISTORY] Saved {task_type} task to history: {history_entry['id']}")


def _build_history_entry(task_type: str, task_data: dict, is_live: bool = False) -> dict:
    total_value = task_data.get("count") if task_type == "register" else len(task_data.get("account_ids", []))
    return {
        "id": task_data.get("id", ""),
        "type": task_type,  # "register" or "login"
        "status": task_data.get("status", ""),
        "progress": task_data.get("progress", 0),
        "total": total_value,
        "success_count": task_data.get("success_count", 0),
        "fail_count": task_data.get("fail_count", 0),
        "created_at": task_data.get("created_at", time.time()),
        "finished_at": task_data.get("finished_at"),
        "is_live": is_live,
    }


def _persist_task_history() -> None:
    """持久化任务历史到数据库（仅数据库模式）。"""
    if not storage.is_database_enabled():
        return
    try:
        if not task_history:
            storage.clear_task_history_sync()
            return
        storage.save_task_history_entry_sync(task_history[-1])
    except Exception as exc:
        logger.warning(f"[HISTORY] Persist task history failed: {exc}")


def _load_task_history() -> None:
    """从数据库加载任务历史（仅数据库模式）。"""
    if not storage.is_database_enabled():
        return
    try:
        history = storage.load_task_history_sync(limit=100)
        if not isinstance(history, list):
            return
        with task_history_lock:
            task_history.clear()
            for entry in history:
                if isinstance(entry, dict):
                    task_history.append(entry)
    except Exception as exc:
        logger.warning(f"[HISTORY] Load task history failed: {exc}")


def build_recent_conversation_entry(
    request_id: str,
    model: Optional[str],
    message_count: Optional[int],
    start_ts: float,
    status: str,
    duration_s: Optional[float] = None,
    error_detail: Optional[str] = None,
) -> dict:
    start_time = get_beijing_time_str(start_ts)
    if model:
        start_content = f"{model}"
        if message_count:
            start_content = f"{model} | {message_count}条消息"
    else:
        start_content = "请求处理中"

    events = [{
        "time": start_time,
        "type": "start",
        "content": start_content,
    }]

    end_time = get_beijing_time_str(start_ts + duration_s) if duration_s is not None else get_beijing_time_str()

    if status == "success":
        if duration_s is not None:
            events.append({
                "time": end_time,
                "type": "complete",
                "status": "success",
            "content": f"响应完成 | 耗时{duration_s:.2f}s",
            })
        else:
            events.append({
                "time": end_time,
                "type": "complete",
                "status": "success",
            "content": "响应完成",
            })
    elif status == "timeout":
        events.append({
            "time": end_time,
            "type": "complete",
            "status": "timeout",
            "content": "请求超时",
        })
    else:
        detail = error_detail or "请求失败"
        events.append({
            "time": end_time,
            "type": "complete",
            "status": "error",
            "content": detail[:120],
        })

    return {
        "request_id": request_id,
        "start_time": start_time,
        "start_ts": start_ts,
        "status": status,
        "events": events,
    }

class MemoryLogHandler(logging.Handler):
    """自定义日志处理器，将日志写入内存缓冲区"""
    def emit(self, record):
        log_entry = self.format(record)
        # 转换为北京时间（UTC+8）
        beijing_tz = timezone(timedelta(hours=8))
        beijing_time = datetime.fromtimestamp(record.created, tz=beijing_tz)
        with log_lock:
            log_buffer.append({
                "time": beijing_time.strftime("%Y-%m-%d %H:%M:%S"),
                "level": record.levelname,
                "message": record.getMessage()
            })

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("exa")

_load_task_history()

# ---------- Linux zombie process reaper ----------
# DrissionPage / Chromium may spawn subprocesses that exit without being waited on,
# which can accumulate as zombies (<defunct>) in long-running services.
try:
    from core.child_reaper import install_child_reaper

    install_child_reaper(log=lambda m: logger.warning(m))
except Exception:
    # Never fail startup due to optional process reaper.
    pass

# 添加内存日志处理器
memory_handler = MemoryLogHandler()
memory_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(memory_handler)

# ---------- 配置管理（使用统一配置系统）----------
# 所有配置通过 config_manager 访问，优先级：环境变量 > YAML > 默认值
TIMEOUT_SECONDS = 300
ADMIN_KEY = config.security.admin_key
_proxy_auth, _no_proxy_auth = parse_proxy_setting(config.basic.proxy_for_auth)
_proxy_chat, _no_proxy_chat = parse_proxy_setting(config.basic.proxy_for_chat)
PROXY_FOR_AUTH = _proxy_auth
PROXY_FOR_CHAT = _proxy_chat
_NO_PROXY = ",".join(filter(None, {_no_proxy_auth, _no_proxy_chat}))
if _NO_PROXY:
    os.environ["NO_PROXY"] = _NO_PROXY
else:
    os.environ.pop("NO_PROXY", None)
BASE_URL = config.basic.base_url
SESSION_SECRET_KEY = config.security.session_secret_key
SESSION_EXPIRE_HOURS = config.session.expire_hours
LINUXDO_OAUTH_ENABLED = bool(getattr(config.basic, "linuxdo_oauth_enabled", False))
LINUXDO_CLIENT_ID = str(getattr(config.basic, "linuxdo_client_id", "") or "").strip()
LINUXDO_CLIENT_SECRET = str(getattr(config.basic, "linuxdo_client_secret", "") or "").strip()
LINUXDO_AUTHORIZE_URL = str(
    getattr(config.basic, "linuxdo_authorize_url", "https://connect.linux.do/oauth2/authorize") or ""
).strip()
LINUXDO_TOKEN_URL = str(
    getattr(config.basic, "linuxdo_token_url", "https://connect.linux.do/oauth2/token") or ""
).strip()
LINUXDO_USERINFO_URL = str(
    getattr(config.basic, "linuxdo_userinfo_url", "https://connect.linux.do/api/user") or ""
).strip()
LINUXDO_REDIRECT_URI = str(getattr(config.basic, "linuxdo_redirect_uri", "") or "").strip()
LINUXDO_SCOPE = str(getattr(config.basic, "linuxdo_scope", "openid profile email") or "openid profile email").strip()

# ---------- 公开展示配置 ----------
LOGO_URL = config.public_display.logo_url
CHAT_URL = config.public_display.chat_url

# ---------- 重试配置 ----------
MAX_ACCOUNT_SWITCH_TRIES = config.retry.max_account_switch_tries
SESSION_CACHE_TTL_SECONDS = config.retry.session_cache_ttl_seconds

def build_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        cooldowns=CooldownConfig(
            text=config.retry.text_rate_limit_cooldown_seconds,
            images=config.retry.images_rate_limit_cooldown_seconds,
            videos=config.retry.videos_rate_limit_cooldown_seconds,
        ),
    )

RETRY_POLICY = build_retry_policy()


def _default_user_auth_policy() -> dict:
    return {
        "registration_enabled": True,
        "password_login_enabled": True,
        "password_registration_enabled": True,
        "linuxdo_oauth_registration_enabled": True,
        "limits": {
            "user": {
                "daily_limit": 200,
                "window_minutes": 10,
                "window_max_calls": 30,
            },
            "premium": {
                "daily_limit": 1000,
                "window_minutes": 10,
                "window_max_calls": 120,
            },
        },
    }


def _sanitize_user_auth_policy(raw: Optional[dict]) -> dict:
    base = _default_user_auth_policy()
    if not isinstance(raw, dict):
        return base
    out = dict(base)
    out["registration_enabled"] = bool(raw.get("registration_enabled", base["registration_enabled"]))
    out["password_login_enabled"] = bool(raw.get("password_login_enabled", base["password_login_enabled"]))
    out["password_registration_enabled"] = bool(
        raw.get("password_registration_enabled", base["password_registration_enabled"])
    )
    out["linuxdo_oauth_registration_enabled"] = bool(
        raw.get("linuxdo_oauth_registration_enabled", base["linuxdo_oauth_registration_enabled"])
    )
    limits = raw.get("limits")
    if isinstance(limits, dict):
        for role in ("user", "premium"):
            current = base["limits"][role]
            role_raw = limits.get(role)
            if isinstance(role_raw, dict):
                current = {
                    "daily_limit": max(1, int(role_raw.get("daily_limit", current["daily_limit"]))),
                    "window_minutes": max(1, int(role_raw.get("window_minutes", current["window_minutes"]))),
                    "window_max_calls": max(1, int(role_raw.get("window_max_calls", current["window_max_calls"]))),
                }
            out["limits"][role] = current
    return out


USER_AUTH_POLICY = _default_user_auth_policy()


REDEEM_CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _generate_random_redeem_code(length: int = 12) -> str:
    code_len = max(6, min(int(length or 12), 64))
    return "".join(secrets.choice(REDEEM_CODE_CHARS) for _ in range(code_len))

# ---------- HTTP 客户端 ----------
# 对话操作客户端（用于JWT获取、创建会话、发送消息）
http_client = httpx.AsyncClient(
    proxy=(PROXY_FOR_CHAT or None),
    verify=False,
    http2=False,
    timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=60.0),
    limits=httpx.Limits(
        max_keepalive_connections=100,
        max_connections=200
    )
)

# 对话流式客户端（用于流式响应）
http_client_chat = httpx.AsyncClient(
    proxy=(PROXY_FOR_CHAT or None),
    verify=False,
    http2=False,
    timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=60.0),
    limits=httpx.Limits(
        max_keepalive_connections=100,
        max_connections=200
    )
)

# 账户操作客户端（用于注册）
http_client_auth = httpx.AsyncClient(
    proxy=(PROXY_FOR_AUTH or None),
    verify=False,
    http2=False,
    timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=60.0),
    limits=httpx.Limits(
        max_keepalive_connections=100,
        max_connections=200
    )
)

# 打印代理配置日志
logger.info(f"[PROXY] Account operations (register): {PROXY_FOR_AUTH if PROXY_FOR_AUTH else 'disabled'}")
logger.info(f"[PROXY] Chat operations (JWT/session/messages): {PROXY_FOR_CHAT if PROXY_FOR_CHAT else 'disabled'}")

# ---------- 常量定义 ----------
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"

# ---------- 多账户支持 ----------
# (AccountConfig, AccountManager, MultiAccountManager 已移至 core/account.py)

# ---------- 配置文件管理 ----------
# (配置管理函数已移至 core/account.py)

# 初始化多账户管理器
multi_account_mgr = load_multi_account_config(
    http_client,
    USER_AGENT,
    RETRY_POLICY,
    SESSION_CACHE_TTL_SECONDS,
    global_stats
)

# ---------- 自动注册服务 ----------
register_service = None

def _set_multi_account_mgr(new_mgr):
    global multi_account_mgr
    multi_account_mgr = new_mgr
    if register_service:
        register_service.multi_account_mgr = new_mgr

def _get_global_stats():
    return global_stats

try:
    from core.register_service import RegisterService
    register_service = RegisterService(
        multi_account_mgr,
        http_client_auth,
        USER_AGENT,
        RETRY_POLICY,
        SESSION_CACHE_TTL_SECONDS,
        _get_global_stats,
        _set_multi_account_mgr,
    )
except Exception as e:
    logger.warning("[SYSTEM] 自动注册服务不可用: %s", e)
    register_service = None

# 验证必需的环境变量
if not ADMIN_KEY:
    ADMIN_KEY = "123456"
    logger.warning("[SYSTEM] 未配置 ADMIN_KEY，已回退为默认值 123456（仅用于旧版 /login 接口）")

# 启动日志
logger.info("[SYSTEM] API端点: /search /answer /contents /findSimilar /research/v1")
logger.info("[SYSTEM] Admin API endpoints: /admin/*")
logger.info("[SYSTEM] User API endpoints: /auth/register /auth/login /auth/apikeys/new")
logger.info("[SYSTEM] Public endpoints: /public/log, /public/stats, /public/uptime")
logger.info(f"[SYSTEM] Session过期时间: {SESSION_EXPIRE_HOURS}小时")
logger.info("[SYSTEM] 系统初始化完成")
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # Startup
    global global_stats, USER_AUTH_POLICY
    global_stats = await load_stats()
    admin_default_password_hash = hash_password("123456")
    admin_api_user = await storage.ensure_admin_api_user("admin", password_hash=admin_default_password_hash)
    if admin_api_user:
        logger.info("[AUTH] 管理员用户已初始化: admin")
    else:
        logger.warning("[AUTH] 管理员用户初始化失败（不影响 ADMIN_KEY 登录）")
    policy_from_db = await storage.load_user_auth_policy()
    legacy_redeem_code = ""
    if isinstance(policy_from_db, dict):
        legacy_redeem_code = str(policy_from_db.get("redeem_code") or "").strip()
    USER_AUTH_POLICY = _sanitize_user_auth_policy(policy_from_db)
    if policy_from_db is None:
        await storage.save_user_auth_policy(USER_AUTH_POLICY)
    if legacy_redeem_code:
        legacy_import = await storage.create_redeem_codes([legacy_redeem_code], created_by="legacy-policy")
        if legacy_import.get("created"):
            logger.info("[AUTH] 已迁移旧版固定兑换码到新兑换码表")
    logger.info(
        "[AUTH] 用户策略: register=%s pwd_login=%s pwd_register=%s oauth_register=%s user(daily=%s,%sm/%s) premium(daily=%s,%sm/%s)",
        USER_AUTH_POLICY["registration_enabled"],
        USER_AUTH_POLICY["password_login_enabled"],
        USER_AUTH_POLICY["password_registration_enabled"],
        USER_AUTH_POLICY["linuxdo_oauth_registration_enabled"],
        USER_AUTH_POLICY["limits"]["user"]["daily_limit"],
        USER_AUTH_POLICY["limits"]["user"]["window_minutes"],
        USER_AUTH_POLICY["limits"]["user"]["window_max_calls"],
        USER_AUTH_POLICY["limits"]["premium"]["daily_limit"],
        USER_AUTH_POLICY["limits"]["premium"]["window_minutes"],
        USER_AUTH_POLICY["limits"]["premium"]["window_max_calls"],
    )
    global_stats.setdefault("request_timestamps", [])
    global_stats.setdefault("model_request_timestamps", {})
    global_stats.setdefault("failure_timestamps", [])
    global_stats.setdefault("recent_conversations", [])
    global_stats.setdefault("success_count", 0)
    global_stats.setdefault("failed_count", 0)
    global_stats.setdefault("account_conversations", {})
    global_stats.setdefault("account_failures", {})
    uptime_tracker.configure_storage(os.path.join(DATA_DIR, "uptime.json"))
    uptime_tracker.load_heartbeats()
    for account_id, account_mgr in multi_account_mgr.accounts.items():
        account_mgr.conversation_count = global_stats["account_conversations"].get(account_id, 0)
        account_mgr.failure_count = global_stats["account_failures"].get(account_id, 0)
    logger.info("[SYSTEM] 已恢复账户成功/失败统计")
    logger.info(f"[SYSTEM] 统计数据已加载: {global_stats['total_requests']} 次请求, {global_stats['total_visitors']} 位访客")
    asyncio.create_task(multi_account_mgr.start_background_cleanup())
    logger.info("[SYSTEM] 后台缓存清理任务已启动（间隔: 5分钟）")
    asyncio.create_task(cleanup_database_task())
    logger.info("[SYSTEM] 数据库清理任务已启动（每天清理一次，保留30天数据）")
    if storage.is_database_enabled():
        asyncio.create_task(save_cooldown_states_task())
        logger.info("[SYSTEM] 冷却状态定期保存任务已启动（间隔: 5分钟）")

    mcp_session_manager = None
    try:
        from core.mcp_server import get_mcp_session_manager

        mcp_session_manager = get_mcp_session_manager()
    except Exception as e:
        logger.warning("[SYSTEM] MCP session manager unavailable: %s", e)

    if mcp_session_manager:
        async with mcp_session_manager.run():
            yield
    else:
        yield

    # Shutdown
    if storage.is_database_enabled():
        try:
            success_count = await account.save_all_cooldown_states(multi_account_mgr)
            logger.info(f"[SYSTEM] 应用关闭，已保存 {success_count}/{len(multi_account_mgr.accounts)} 个账户的冷却状态")
        except Exception as e:
            logger.error(f"[SYSTEM] 关闭时保存冷却状态失败: {e}")

app = FastAPI(title="ExaFree", lifespan=lifespan)

frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip()
DISABLE_ADMIN_PANEL = os.getenv("DISABLE_ADMIN_PANEL", "0") == "1"
allow_all_origins = os.getenv("ALLOW_ALL_ORIGINS", "0") == "1"
if allow_all_origins and not frontend_origin:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
elif frontend_origin:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[frontend_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if not DISABLE_ADMIN_PANEL:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    if os.path.exists(os.path.join("static", "assets")):
        app.mount("/assets", StaticFiles(directory=os.path.join("static", "assets")), name="assets")
    if os.path.exists(os.path.join("static", "vendor")):
        app.mount("/vendor", StaticFiles(directory=os.path.join("static", "vendor")), name="vendor")
try:
    from core.mcp_server import get_mcp_http_app

    app.mount("/mcp", get_mcp_http_app(), name="mcp")
    logger.info("[SYSTEM] MCP endpoint mounted at /mcp")
except Exception as e:
    logger.warning("[SYSTEM] MCP endpoint disabled: %s", e)

@app.api_route("/mcp", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def mcp_redirect():
    return RedirectResponse(url="/mcp/", status_code=307)

if not DISABLE_ADMIN_PANEL:
    @app.get("/")
    async def serve_frontend_index():
        index_path = os.path.join("static", "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        raise HTTPException(404, "Not Found")

    @app.get("/logo.svg")
    async def serve_logo():
        logo_path = os.path.join("static", "logo.svg")
        if os.path.exists(logo_path):
            return FileResponse(logo_path)
        raise HTTPException(404, "Not Found")

@app.get("/health")
async def health_check():
    """健康检查端点，用于 Docker HEALTHCHECK"""
    return {"status": "ok"}

# ---------- Session 中间件配置 ----------
from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    max_age=SESSION_EXPIRE_HOURS * 3600,  # 转换为秒
    same_site="lax",
    https_only=False  # 本地开发可设为False，生产环境建议True
)

# ---------- Uptime 追踪中间件 ----------
@app.middleware("http")
async def track_uptime_middleware(request: Request, call_next):
    """Uptime 监控：跟踪非对话接口的请求结果。"""
    path = request.url.path
    if (
        path.startswith("/public/")
        or path.startswith("/favicon")
    ):
        return await call_next(request)

    start_time = time.time()

    try:
        response = await call_next(request)
        latency_ms = int((time.time() - start_time) * 1000)
        success = response.status_code < 400
        uptime_tracker.record_request("api_service", success, latency_ms, response.status_code)
        return response

    except Exception:
        uptime_tracker.record_request("api_service", False)
        raise


# ---------- 后台任务启动 ----------


async def save_cooldown_states_task():
    """定期保存所有账户的冷却状态到数据库"""
    while True:
        try:
            await asyncio.sleep(300)  # 每5分钟执行一次
            for attempt in range(3):
                try:
                    success_count = await account.save_all_cooldown_states(multi_account_mgr)
                    logger.debug(f"[COOLDOWN] 定期保存: {success_count}/{len(multi_account_mgr.accounts)} 个账户")
                    break
                except Exception as retry_err:
                    err_msg = str(retry_err)
                    if "another operation" in err_msg or "ConnectionDoesNotExist" in err_msg or "connection was closed" in err_msg:
                        if attempt < 2:
                            logger.warning(f"[COOLDOWN] 数据库连接繁忙，{attempt+1}/3 次重试...")
                            await asyncio.sleep(5 * (attempt + 1))
                            continue
                    raise
        except Exception as e:
            logger.error(f"[COOLDOWN] 定期保存失败: {e}")


async def cleanup_database_task():
    """定时清理数据库过期数据"""
    while True:
        try:
            await asyncio.sleep(24 * 3600)  # 每天执行一次
            deleted_count = await stats_db.cleanup_old_data(days=30)
            logger.info(f"[DATABASE] 清理了 {deleted_count} 条过期数据（保留30天）")
        except Exception as e:
            logger.error(f"[DATABASE] 清理数据失败: {e}")

# ---------- 日志脱敏函数 ----------
def get_sanitized_logs(limit: int = 100) -> list:
    """获取脱敏后的日志列表，按请求ID分组并提取关键事件"""
    with log_lock:
        logs = list(log_buffer)

    # 按请求ID分组（支持两种格式：带[req_xxx]和不带的）
    request_logs = {}
    orphan_logs = []  # 没有request_id的日志（如选择账户）

    for log in logs:
        message = log["message"]
        req_match = re.search(r'\[req_([a-z0-9]+)\]', message)

        if req_match:
            request_id = req_match.group(1)
            if request_id not in request_logs:
                request_logs[request_id] = []
            request_logs[request_id].append(log)
        else:
            # 没有request_id的日志（如选择账户），暂存
            orphan_logs.append(log)

    # 将orphan_logs（如选择账户）关联到对应的请求
    # 策略：将orphan日志关联到时间上最接近的后续请求
    for orphan in orphan_logs:
        orphan_time = orphan["time"]
        # 找到时间上最接近且在orphan之后的请求
        closest_request_id = None
        min_time_diff = None

        for request_id, req_logs in request_logs.items():
            if req_logs:
                first_log_time = req_logs[0]["time"]
                # orphan应该在请求之前或同时
                if first_log_time >= orphan_time:
                    if min_time_diff is None or first_log_time < min_time_diff:
                        min_time_diff = first_log_time
                        closest_request_id = request_id

        # 如果找到最接近的请求，将orphan日志插入到该请求的日志列表开头
        if closest_request_id:
            request_logs[closest_request_id].insert(0, orphan)

    # 为每个请求提取关键事件
    sanitized = []
    for request_id, req_logs in request_logs.items():
        # 收集关键信息
        model = None
        message_count = None
        retry_events = []
        final_status = "in_progress"
        duration = None
        start_time = req_logs[0]["time"]

        # 遍历该请求的所有日志
        for log in req_logs:
            message = log["message"]

            # 提取模型名称和消息数量（开始对话）
            if '收到请求:' in message and not model:
                model_match = re.search(r'收到请求: ([^ |]+)', message)
                if model_match:
                    model = model_match.group(1)
                count_match = re.search(r'(\d+)条消息', message)
                if count_match:
                    message_count = int(count_match.group(1))

            # 提取重试事件（包括失败尝试、账户切换、选择账户）
            # 注意：不提取"正在重试"日志，因为它和"失败 (尝试"是配套的
            if any(keyword in message for keyword in ['切换账户', '选择账户', '失败 (尝试']):
                retry_events.append({
                    "time": log["time"],
                    "message": message
                })

            # 提取响应完成（最高优先级 - 最终成功则忽略中间错误）
            if '响应完成:' in message:
                time_match = re.search(r'响应完成: ([\d.]+)秒', message)
                if time_match:
                    duration = time_match.group(1) + 's'
                    final_status = "success"

            # 检测非流式响应完成
            if '非流式响应完成' in message:
                final_status = "success"

            # 检测失败状态（仅在非success状态下）
            if final_status != "success" and (log['level'] == 'ERROR' or '失败' in message):
                final_status = "error"

            # 检测超时（仅在非success状态下）
            if final_status != "success" and '超时' in message:
                final_status = "timeout"

        # 如果没有模型信息但有错误，仍然显示
        if not model and final_status == "in_progress":
            continue

        # 构建关键事件列表
        events = []

        # 1. 开始对话
        if model:
            events.append({
                "time": start_time,
                "type": "start",
                "content": f"{model} | {message_count}条消息" if message_count else model
            })
        else:
            # 没有模型信息但有错误的情况
            events.append({
                "time": start_time,
                "type": "start",
                "content": "请求处理中"
            })

        # 2. 重试事件
        failure_count = 0  # 失败重试计数
        account_select_count = 0  # 账户选择计数

        for i, retry in enumerate(retry_events):
            msg = retry["message"]

            # 识别不同类型的重试事件（按优先级匹配）
            if '失败 (尝试' in msg:
                # 创建会话失败
                failure_count += 1
                events.append({
                    "time": retry["time"],
                    "type": "retry",
                    "content": f"服务异常，正在重试（{failure_count}）"
                })
            elif '选择账户' in msg:
                # 账户选择/切换
                account_select_count += 1

                # 检查下一条日志是否是"切换账户"，如果是则跳过当前"选择账户"（避免重复）
                next_is_switch = (i + 1 < len(retry_events) and '切换账户' in retry_events[i + 1]["message"])

                if not next_is_switch:
                    if account_select_count == 1:
                        # 第一次选择：显示为"选择服务节点"
                        events.append({
                            "time": retry["time"],
                            "type": "select",
                            "content": "选择服务节点"
                        })
                    else:
                        # 第二次及以后：显示为"切换服务节点"
                        events.append({
                            "time": retry["time"],
                            "type": "switch",
                            "content": "切换服务节点"
                        })
            elif '切换账户' in msg:
                # 运行时切换账户（显示为"切换服务节点"）
                events.append({
                    "time": retry["time"],
                    "type": "switch",
                    "content": "切换服务节点"
                })

        # 3. 完成事件
        if final_status == "success":
            if duration:
                events.append({
                    "time": req_logs[-1]["time"],
                    "type": "complete",
                    "status": "success",
                    "content": f"响应完成 | 耗时{duration}"
                })
            else:
                events.append({
                    "time": req_logs[-1]["time"],
                    "type": "complete",
                    "status": "success",
                    "content": "响应完成"
                })
        elif final_status == "error":
            events.append({
                "time": req_logs[-1]["time"],
                "type": "complete",
                "status": "error",
                "content": "请求失败"
            })
        elif final_status == "timeout":
            events.append({
                "time": req_logs[-1]["time"],
                "type": "complete",
                "status": "timeout",
                "content": "请求超时"
            })

        sanitized.append({
            "request_id": request_id,
            "start_time": start_time,
            "status": final_status,
            "events": events
        })

    # 按时间排序并限制数量
    sanitized.sort(key=lambda x: x["start_time"], reverse=True)
    return sanitized[:limit]

# ---------- Auth endpoints (API) ----------

@app.post("/login")
async def admin_login_post(request: Request, admin_key: str = Form(...)):
    """Admin login (API)"""
    if admin_key == ADMIN_KEY:
        admin_user = await storage.get_api_user_by_username("admin") if storage.is_database_enabled() else None
        login_user(
            request,
            user_id=(admin_user or {}).get("user_id", ""),
            username="admin",
            role="admin",
        )
        logger.info("[AUTH] Admin login success")
        return {"success": True}
    logger.warning("[AUTH] Login failed - invalid key")
    raise HTTPException(401, "Invalid key")


@app.post("/logout")
@require_login(redirect_to_login=False, admin_only=False)
async def admin_logout(request: Request):
    """Portal logout (API)"""
    logout_user(request)
    logger.info("[AUTH] Logout")
    return {"success": True}


@app.post("/auth/logout")
@require_login(redirect_to_login=False, admin_only=False)
async def user_logout(request: Request):
    logout_user(request)
    logger.info("[AUTH] Logout")
    return {"success": True}


def _require_user_storage() -> None:
    if not storage.is_database_enabled():
        raise HTTPException(503, "Database storage is required for multi-user auth")


def _resolve_role_limits(role: str) -> Optional[dict]:
    if role == "admin":
        return None
    role_key = "premium" if role == "premium" else "user"
    return USER_AUTH_POLICY["limits"].get(role_key, USER_AUTH_POLICY["limits"]["user"])


def _day_start_ts_local() -> int:
    now = datetime.now()
    return int(datetime(now.year, now.month, now.day, 0, 0, 0).timestamp())


async def _build_limit_snapshot(user_id: str, role: str) -> dict:
    limits = _resolve_role_limits(role)
    if limits is None:
        return {
            "role": role,
            "daily_limit": None,
            "window_minutes": None,
            "window_max_calls": None,
            "today_call_count": 0,
            "window_call_count": 0,
            "remaining_today": None,
            "remaining_window": None,
        }

    window_minutes = int(limits["window_minutes"])
    now_ts = int(time.time())
    counts = await storage.get_user_request_counts(
        user_id,
        day_start_ts=_day_start_ts_local(),
        window_start_ts=now_ts - window_minutes * 60,
    )
    daily_limit = int(limits["daily_limit"])
    window_max_calls = int(limits["window_max_calls"])
    return {
        "role": role,
        "daily_limit": daily_limit,
        "window_minutes": window_minutes,
        "window_max_calls": window_max_calls,
        "today_call_count": counts["day_count"],
        "window_call_count": counts["window_count"],
        "remaining_today": max(0, daily_limit - counts["day_count"]),
        "remaining_window": max(0, window_max_calls - counts["window_count"]),
        "last_call_ts": counts.get("last_call_ts"),
    }


def _linuxdo_oauth_ready() -> bool:
    return bool(
        LINUXDO_OAUTH_ENABLED
        and LINUXDO_CLIENT_ID
        and LINUXDO_CLIENT_SECRET
        and LINUXDO_AUTHORIZE_URL
        and LINUXDO_TOKEN_URL
        and LINUXDO_USERINFO_URL
    )


def _resolve_linuxdo_redirect_uri(request: Request) -> str:
    configured = (LINUXDO_REDIRECT_URI or "").strip()
    if configured:
        return configured
    base = (config.basic.base_url or "").strip().rstrip("/")
    if not base:
        base = str(request.base_url).rstrip("/")
    return f"{base}/auth/linuxdo/callback"


def _sanitize_linuxdo_username(raw: str) -> str:
    value = (raw or "").strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "-", value)
    value = value.strip("._-")
    if len(value) < 3:
        value = f"ld_{value}".strip("._-")
    if len(value) < 3:
        value = f"ld_user"
    if len(value) > 32:
        value = value[:32].strip("._-")
    if len(value) < 3:
        value = (value + "user")[:3]
    return value


async def _build_unique_linuxdo_username(profile: dict, subject: str) -> str:
    candidates = [
        str(profile.get("username") or "").strip(),
        str(profile.get("name") or "").strip(),
        f"linuxdo_{subject}",
        f"ld_{subject}",
    ]
    base = ""
    for candidate in candidates:
        normalized = _sanitize_linuxdo_username(candidate)
        if is_valid_username(normalized) and normalized != "admin":
            base = normalized
            break
    if not base:
        base = _sanitize_linuxdo_username(f"linuxdo_{subject}")
    if base == "admin":
        base = "linuxdo_user"

    if not await storage.get_api_user_by_username(base):
        return base

    for idx in range(1, 1000):
        suffix = f"_{idx}"
        max_base_len = 32 - len(suffix)
        candidate = f"{base[:max_base_len]}{suffix}"
        candidate = _sanitize_linuxdo_username(candidate)
        if candidate != "admin" and is_valid_username(candidate):
            if not await storage.get_api_user_by_username(candidate):
                return candidate

    return f"ld_{uuid.uuid4().hex[:20]}"


@app.get("/auth/me")
@require_login(redirect_to_login=False, admin_only=False)
async def auth_me(request: Request):
    _require_user_storage()
    session_user = get_session_user(request)
    user = await storage.get_api_user_by_id(session_user.get("user_id", ""))
    if not user or not user.get("is_active"):
        logout_user(request)
        raise HTTPException(401, "Unauthorized")

    keys = await storage.list_user_api_keys(user["user_id"], include_inactive=True)
    limits = await _build_limit_snapshot(user["user_id"], user["role"])
    return {
        "success": True,
        "user": {
            "user_id": user["user_id"],
            "username": user["username"],
            "role": user["role"],
            "is_active": user["is_active"],
            "created_at": user["created_at"],
        },
        "limits": limits,
        "keys": keys,
    }


@app.get("/auth/options")
async def auth_options():
    _require_user_storage()
    registration_enabled = bool(USER_AUTH_POLICY.get("registration_enabled", True))
    password_login_enabled = bool(USER_AUTH_POLICY.get("password_login_enabled", True))
    password_registration_enabled = bool(USER_AUTH_POLICY.get("password_registration_enabled", True))
    linuxdo_oauth_registration_enabled = bool(
        USER_AUTH_POLICY.get("linuxdo_oauth_registration_enabled", True)
    )
    oauth_ready = _linuxdo_oauth_ready()
    return {
        "registration_enabled": registration_enabled,
        "password_login_enabled": password_login_enabled,
        "password_registration_enabled": password_registration_enabled,
        "linuxdo_oauth_registration_enabled": linuxdo_oauth_registration_enabled,
        "linuxdo_oauth_login_enabled": oauth_ready,
        "linuxdo_oauth_ready": oauth_ready,
        "linuxdo_oauth_start_url": "/auth/linuxdo/start",
    }


def _build_oauth_login_redirect_url(error_message: Optional[str] = None) -> str:
    base = "/#/login"
    if not error_message:
        return base
    return f"{base}?oauth_error={quote(str(error_message)[:200])}"


@app.get("/auth/linuxdo/start")
async def auth_linuxdo_start(request: Request):
    _require_user_storage()
    if not _linuxdo_oauth_ready():
        raise HTTPException(503, "Linux DO OAuth is not configured")

    state = secrets.token_urlsafe(32)
    redirect_uri = _resolve_linuxdo_redirect_uri(request)
    request.session["linuxdo_oauth_state"] = state
    request.session["linuxdo_oauth_ts"] = int(time.time())

    params = {
        "client_id": LINUXDO_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": LINUXDO_SCOPE or "openid profile email",
        "state": state,
    }
    auth_url = f"{LINUXDO_AUTHORIZE_URL}?{urlencode(params)}"
    return RedirectResponse(url=auth_url, status_code=302)


@app.get("/auth/linuxdo/callback")
async def auth_linuxdo_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
):
    _require_user_storage()
    if error:
        return RedirectResponse(url=_build_oauth_login_redirect_url(f"oauth_error:{error}"), status_code=302)
    if not _linuxdo_oauth_ready():
        return RedirectResponse(url=_build_oauth_login_redirect_url("OAuth configuration invalid"), status_code=302)

    expected_state = request.session.pop("linuxdo_oauth_state", None)
    request.session.pop("linuxdo_oauth_ts", None)
    if not expected_state or not state or expected_state != state:
        return RedirectResponse(url=_build_oauth_login_redirect_url("Invalid OAuth state"), status_code=302)
    if not code:
        return RedirectResponse(url=_build_oauth_login_redirect_url("Missing OAuth code"), status_code=302)

    redirect_uri = _resolve_linuxdo_redirect_uri(request)
    token_payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": LINUXDO_CLIENT_ID,
        "client_secret": LINUXDO_CLIENT_SECRET,
    }

    try:
        async with httpx.AsyncClient(
            timeout=20,
            follow_redirects=True,
            proxy=(PROXY_FOR_AUTH or None),
            verify=False,
        ) as oauth_client:
            token_resp = await oauth_client.post(
                LINUXDO_TOKEN_URL,
                data=token_payload,
                headers={"Accept": "application/json"},
            )
            if token_resp.status_code >= 400:
                logger.warning(
                    "[AUTH] LinuxDO token exchange failed: HTTP %s, body=%s",
                    token_resp.status_code,
                    token_resp.text[:300],
                )
                return RedirectResponse(url=_build_oauth_login_redirect_url("OAuth token exchange failed"), status_code=302)
            token_data = token_resp.json()
            access_token = str(token_data.get("access_token") or "").strip()
            if not access_token:
                logger.warning("[AUTH] LinuxDO token exchange missing access_token: %s", token_data)
                return RedirectResponse(url=_build_oauth_login_redirect_url("OAuth access token missing"), status_code=302)

            profile_resp = await oauth_client.get(
                LINUXDO_USERINFO_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )
            if profile_resp.status_code >= 400:
                logger.warning(
                    "[AUTH] LinuxDO userinfo failed: HTTP %s, body=%s",
                    profile_resp.status_code,
                    profile_resp.text[:300],
                )
                return RedirectResponse(url=_build_oauth_login_redirect_url("OAuth userinfo failed"), status_code=302)
            profile = profile_resp.json()
    except httpx.RequestError as exc:
        logger.warning("[AUTH] LinuxDO OAuth request error: %s", repr(exc))
        return RedirectResponse(url=_build_oauth_login_redirect_url("OAuth request failed"), status_code=302)
    except Exception as exc:
        logger.warning("[AUTH] LinuxDO OAuth callback exception: %s", repr(exc))
        return RedirectResponse(url=_build_oauth_login_redirect_url("OAuth request failed"), status_code=302)

    subject = str(profile.get("id") or profile.get("sub") or profile.get("user_id") or "").strip()
    if not subject:
        return RedirectResponse(url=_build_oauth_login_redirect_url("OAuth user id missing"), status_code=302)

    provider = "linuxdo"
    user = await storage.get_api_user_by_oauth(provider, subject)
    if not user:
        allow_register = bool(USER_AUTH_POLICY.get("registration_enabled", True)) and bool(
            USER_AUTH_POLICY.get("linuxdo_oauth_registration_enabled", True)
        )
        if not allow_register:
            return RedirectResponse(url=_build_oauth_login_redirect_url("OAuth registration is disabled"), status_code=302)

        username = await _build_unique_linuxdo_username(profile, subject)
        password_seed = secrets.token_urlsafe(32)
        user = await storage.create_api_user(
            username=username,
            password_hash=hash_password(password_seed),
            role="user",
        )
        if not user:
            return RedirectResponse(url=_build_oauth_login_redirect_url("Create OAuth user failed"), status_code=302)

    identity = await storage.save_oauth_identity(
        user_id=user["user_id"],
        provider=provider,
        provider_user_id=subject,
        profile=profile if isinstance(profile, dict) else {},
    )
    if not identity:
        fallback_user = await storage.get_api_user_by_oauth(provider, subject)
        if fallback_user:
            user = fallback_user
        else:
            return RedirectResponse(url=_build_oauth_login_redirect_url("Bind OAuth identity failed"), status_code=302)

    if not user.get("is_active"):
        return RedirectResponse(url=_build_oauth_login_redirect_url("User is disabled"), status_code=302)

    login_user(
        request,
        user_id=user["user_id"],
        username=user["username"],
        role=user["role"],
    )
    target = "/#/dashboard" if user["role"] == "admin" else "/#/apikeys"
    return RedirectResponse(url=target, status_code=302)


@app.post("/auth/register")
async def user_register(payload: dict = Body(...)):
    """用户注册并创建首个 API Key。"""
    _require_user_storage()
    if not USER_AUTH_POLICY.get("registration_enabled", True):
        raise HTTPException(403, "Registration is disabled")
    if not USER_AUTH_POLICY.get("password_registration_enabled", True):
        raise HTTPException(403, "Password registration is disabled")
    username = normalize_username(payload.get("username"))
    password = payload.get("password") or ""

    if not is_valid_username(username):
        raise HTTPException(400, "Invalid username. Use 3-32 chars: letters, digits, _ . -")
    if username == "admin":
        raise HTTPException(400, "Username 'admin' is reserved")
    if not is_valid_password(password):
        raise HTTPException(400, "Invalid password. Length must be 8-128")

    exists = await storage.get_api_user_by_username(username)
    if exists:
        raise HTTPException(409, "Username already exists")

    user = await storage.create_api_user(username=username, password_hash=hash_password(password), role="user")
    if not user:
        raise HTTPException(409, "Username already exists")

    api_key_plain = generate_api_key()
    key_row = await storage.create_api_key(
        user_id=user["user_id"],
        key_hash=hash_api_key(api_key_plain),
        key_prefix=key_prefix(api_key_plain),
        name="default",
    )
    if not key_row:
        raise HTTPException(500, "Failed to create API key")

    logger.info(f"[AUTH] user registered: {username}")
    return {
        "success": True,
        "user": {
            "user_id": user["user_id"],
            "username": user["username"],
            "role": user["role"],
        },
        "api_key": api_key_plain,
    }


@app.post("/auth/login")
async def user_login(request: Request, payload: dict = Body(...)):
    """门户登录（管理员/普通用户/高级用户）。"""
    _require_user_storage()
    username = normalize_username(payload.get("username"))
    password = payload.get("password") or ""
    if not USER_AUTH_POLICY.get("password_login_enabled", True) and username != "admin":
        raise HTTPException(403, "Password login is disabled")

    user = await storage.get_api_user_by_username(username)
    if not user or not user.get("is_active"):
        raise HTTPException(401, "Invalid username or password")
    if not verify_password(password, user.get("password_hash", "")):
        raise HTTPException(401, "Invalid username or password")

    login_user(
        request,
        user_id=user["user_id"],
        username=user["username"],
        role=user["role"],
    )
    keys = await storage.list_user_api_keys(user["user_id"], include_inactive=True)
    limits = await _build_limit_snapshot(user["user_id"], user["role"])
    return {
        "success": True,
        "user": {
            "user_id": user["user_id"],
            "username": user["username"],
            "role": user["role"],
            "is_active": user["is_active"],
        },
        "limits": limits,
        "keys": keys,
    }


@app.post("/auth/change-password")
@require_login(redirect_to_login=False, admin_only=False)
async def auth_change_password(request: Request, payload: dict = Body(...)):
    _require_user_storage()
    old_password = payload.get("old_password") or ""
    new_password = payload.get("new_password") or ""
    if not is_valid_password(new_password):
        raise HTTPException(400, "Invalid password. Length must be 8-128")

    session_user = get_session_user(request)
    user = await storage.get_api_user_by_id(session_user.get("user_id", ""))
    if not user or not user.get("is_active"):
        raise HTTPException(401, "Unauthorized")
    if not verify_password(old_password, user.get("password_hash", "")):
        raise HTTPException(401, "Old password is incorrect")

    ok = await storage.update_api_user_password(user["user_id"], hash_password(new_password))
    if not ok:
        raise HTTPException(500, "Failed to update password")
    return {"success": True}


@app.post("/auth/redeem")
@require_login(redirect_to_login=False, admin_only=False)
async def auth_redeem(request: Request, payload: dict = Body(...)):
    _require_user_storage()
    code = str(payload.get("code") or "").strip()
    if not code:
        raise HTTPException(400, "redeem code required")

    session_user = get_session_user(request)
    user = await storage.get_api_user_by_id(session_user.get("user_id", ""))
    if not user or not user.get("is_active"):
        raise HTTPException(401, "Unauthorized")
    if user["role"] == "admin":
        raise HTTPException(400, "Admin does not need redeem code")
    if user["role"] == "premium":
        return {"success": True, "message": "already premium"}

    redeem_result = await storage.consume_redeem_code_for_user(code, user["user_id"])
    if not redeem_result.get("ok"):
        reason = redeem_result.get("reason")
        if reason == "already_used":
            raise HTTPException(400, "Redeem code already used")
        if reason == "already_premium":
            return {"success": True, "message": "already premium"}
        if reason == "admin_forbidden":
            raise HTTPException(400, "Admin does not need redeem code")
        if reason == "user_not_found":
            raise HTTPException(401, "Unauthorized")
        if reason == "invalid_code":
            raise HTTPException(400, "Invalid redeem code")
        raise HTTPException(500, "Upgrade failed")

    request.session["role"] = "premium"
    limits = await _build_limit_snapshot(user["user_id"], "premium")
    return {"success": True, "role": "premium", "limits": limits}


@app.get("/auth/apikeys")
@require_login(redirect_to_login=False, admin_only=False)
async def user_list_api_keys(request: Request):
    _require_user_storage()
    session_user = get_session_user(request)
    user = await storage.get_api_user_by_id(session_user.get("user_id", ""))
    if not user or not user.get("is_active"):
        raise HTTPException(401, "Unauthorized")
    keys = await storage.list_user_api_keys(user["user_id"], include_inactive=True)
    limits = await _build_limit_snapshot(user["user_id"], user["role"])
    return {"total": len(keys), "keys": keys, "limits": limits}


@app.post("/auth/apikeys/new")
@require_login(redirect_to_login=False, admin_only=False)
async def user_create_api_key(request: Request, payload: dict = Body(...)):
    """用户新增 API Key。"""
    _require_user_storage()
    key_name = (payload.get("name") or "manual").strip() or "manual"

    session_user = get_session_user(request)
    user = await storage.get_api_user_by_id(session_user.get("user_id", ""))
    if not user or not user.get("is_active"):
        raise HTTPException(401, "Unauthorized")

    api_key_plain = generate_api_key()
    key_row = await storage.create_api_key(
        user_id=user["user_id"],
        key_hash=hash_api_key(api_key_plain),
        key_prefix=key_prefix(api_key_plain),
        name=key_name[:40],
    )
    if not key_row:
        raise HTTPException(500, "Failed to create API key")
    return {
        "success": True,
        "api_key": api_key_plain,
        "key": key_row,
    }


@app.post("/auth/apikeys/revoke")
@require_login(redirect_to_login=False, admin_only=False)
async def user_revoke_api_key(request: Request, payload: dict = Body(...)):
    """用户吊销自己的 API Key。"""
    _require_user_storage()
    key_id = (payload.get("key_id") or "").strip()

    if not key_id:
        raise HTTPException(400, "key_id is required")

    session_user = get_session_user(request)
    user = await storage.get_api_user_by_id(session_user.get("user_id", ""))
    if not user or not user.get("is_active"):
        raise HTTPException(401, "Unauthorized")

    revoked = await storage.deactivate_api_key(key_id, user_id=user["user_id"])
    if not revoked:
        raise HTTPException(404, "API key not found")
    return {"success": True}


@app.get("/admin/users")
@require_login()
async def admin_list_users(request: Request, limit: int = 200):
    _require_user_storage()
    users = await storage.list_api_users_with_usage(limit=limit)
    for user in users:
        user["last_call_time"] = get_beijing_time_str(user["last_call_ts"]) if user.get("last_call_ts") else None
    return {"total": len(users), "users": users}


@app.post("/admin/users")
@require_login()
async def admin_create_user(request: Request, payload: dict = Body(...)):
    _require_user_storage()
    username = normalize_username(payload.get("username"))
    password = payload.get("password") or ""
    role = (payload.get("role") or "user").strip().lower()
    create_key = bool(payload.get("create_key", False))

    if role not in ("user", "premium"):
        role = "user"
    if not is_valid_username(username):
        raise HTTPException(400, "Invalid username")
    if username == "admin":
        raise HTTPException(400, "Username 'admin' is reserved")
    if not is_valid_password(password):
        raise HTTPException(400, "Invalid password. Length must be 8-128")
    if await storage.get_api_user_by_username(username):
        raise HTTPException(409, "Username already exists")

    user = await storage.create_api_user(username, hash_password(password), role=role)
    if not user:
        raise HTTPException(500, "Create user failed")

    api_key_plain = None
    if create_key:
        api_key_plain = generate_api_key()
        created = await storage.create_api_key(
            user["user_id"],
            hash_api_key(api_key_plain),
            key_prefix(api_key_plain),
            name="default",
        )
        if not created:
            raise HTTPException(500, "User created but key create failed")
    return {
        "success": True,
        "user": {
            "user_id": user["user_id"],
            "username": user["username"],
            "role": user["role"],
            "is_active": user["is_active"],
        },
        "api_key": api_key_plain,
    }


@app.delete("/admin/users/{user_id}")
@require_login()
async def admin_delete_user(request: Request, user_id: str):
    _require_user_storage()
    user = await storage.get_api_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "user not found")
    if user.get("role") == "admin":
        raise HTTPException(400, "cannot delete admin user")
    ok = await storage.delete_api_user(user_id)
    if not ok:
        raise HTTPException(500, "delete user failed")
    return {"success": True}


@app.put("/admin/users/{user_id}/enable")
@require_login()
async def admin_enable_user(request: Request, user_id: str):
    _require_user_storage()
    ok = await storage.set_api_user_active(user_id, True)
    if not ok:
        raise HTTPException(404, "user not found")
    return {"success": True}


@app.put("/admin/users/{user_id}/disable")
@require_login()
async def admin_disable_user(request: Request, user_id: str):
    _require_user_storage()
    user = await storage.get_api_user_by_id(user_id)
    if user and user.get("role") == "admin":
        raise HTTPException(400, "cannot disable admin user")
    ok = await storage.set_api_user_active(user_id, False)
    if not ok:
        raise HTTPException(404, "user not found")
    return {"success": True}


@app.get("/admin/user-policy")
@require_login()
async def admin_get_user_policy(request: Request):
    _require_user_storage()
    return {"policy": USER_AUTH_POLICY}


@app.put("/admin/user-policy")
@require_login()
async def admin_update_user_policy(request: Request, payload: dict = Body(...)):
    _require_user_storage()
    global USER_AUTH_POLICY
    merged = json.loads(json.dumps(USER_AUTH_POLICY))
    payload = payload or {}
    if "registration_enabled" in payload:
        merged["registration_enabled"] = payload.get("registration_enabled")
    if "password_login_enabled" in payload:
        merged["password_login_enabled"] = payload.get("password_login_enabled")
    if "password_registration_enabled" in payload:
        merged["password_registration_enabled"] = payload.get("password_registration_enabled")
    if "linuxdo_oauth_registration_enabled" in payload:
        merged["linuxdo_oauth_registration_enabled"] = payload.get("linuxdo_oauth_registration_enabled")
    if isinstance(payload.get("limits"), dict):
        limits = payload["limits"]
        for role in ("user", "premium"):
            if isinstance(limits.get(role), dict):
                merged.setdefault("limits", {}).setdefault(role, {}).update(limits[role])
    policy = _sanitize_user_auth_policy(merged)
    saved = await storage.save_user_auth_policy(policy)
    if not saved:
        raise HTTPException(500, "save user policy failed")
    USER_AUTH_POLICY = policy
    return {"success": True, "policy": USER_AUTH_POLICY}


@app.get("/admin/redeem-codes")
@require_login()
async def admin_list_redeem_codes(request: Request, limit: int = 1000, include_used: bool = True):
    _require_user_storage()
    codes = await storage.list_redeem_codes(limit=limit, include_used=include_used)
    used_count = sum(1 for item in codes if item.get("is_used"))
    return {
        "total": len(codes),
        "used_count": used_count,
        "unused_count": len(codes) - used_count,
        "codes": codes,
    }


@app.post("/admin/redeem-codes/generate")
@require_login()
async def admin_generate_redeem_codes(request: Request, payload: dict = Body(...)):
    _require_user_storage()
    payload = payload or {}
    requested = max(1, min(int(payload.get("count", 10)), 1000))
    length = max(6, min(int(payload.get("length", 12)), 64))

    session_user = get_session_user(request)
    creator = str(session_user.get("username") or "").strip() or "admin"

    created_rows = []
    duplicates = []
    attempts = 0
    max_attempts = requested * 20
    generated_pool = set()

    while len(created_rows) < requested and attempts < max_attempts:
        need = requested - len(created_rows)
        batch = []
        for _ in range(min(need * 2, 2000)):
            code = _generate_random_redeem_code(length=length)
            if code in generated_pool:
                continue
            generated_pool.add(code)
            batch.append(code)
        if not batch:
            break
        result = await storage.create_redeem_codes(batch, created_by=creator)
        created_rows.extend(result.get("created", []))
        duplicates.extend(result.get("duplicates", []))
        attempts += len(batch)

    created_rows = created_rows[:requested]
    return {
        "success": True,
        "requested": requested,
        "created_count": len(created_rows),
        "duplicates_count": len(duplicates),
        "codes": created_rows,
        "plain_codes": [item.get("code") for item in created_rows],
    }


@app.post("/admin/redeem-codes/import")
@require_login()
async def admin_import_redeem_codes(request: Request, payload: dict = Body(...)):
    _require_user_storage()
    payload = payload or {}

    raw_codes = payload.get("codes")
    if isinstance(raw_codes, str):
        candidate_codes = re.split(r"[\s,;]+", raw_codes)
    elif isinstance(raw_codes, list):
        candidate_codes = [str(item or "") for item in raw_codes]
    else:
        raise HTTPException(400, "codes is required")

    max_import = 5000
    candidate_codes = [code for code in candidate_codes if str(code or "").strip()]
    if len(candidate_codes) > max_import:
        raise HTTPException(400, f"too many codes, max {max_import}")

    session_user = get_session_user(request)
    creator = str(session_user.get("username") or "").strip() or "admin"

    result = await storage.create_redeem_codes(candidate_codes, created_by=creator)
    return {
        "success": True,
        "imported": len(candidate_codes),
        "created_count": len(result.get("created", [])),
        "duplicates_count": len(result.get("duplicates", [])),
        "invalid_count": len(result.get("invalid", [])),
        "codes": result.get("created", []),
        "duplicates": result.get("duplicates", []),
        "invalid": result.get("invalid", []),
    }


@app.delete("/admin/redeem-codes/{code_id}")
@require_login()
async def admin_delete_redeem_code(request: Request, code_id: str):
    _require_user_storage()
    ok = await storage.delete_redeem_code(code_id)
    if not ok:
        raise HTTPException(404, "redeem code not found")
    return {"success": True}


@app.get("/admin/redeem-codes/export")
@require_login()
async def admin_export_redeem_codes(
    request: Request,
    format: str = "txt",
    include_used: bool = True,
    only_unused: bool = False,
):
    _require_user_storage()
    export_format = str(format or "txt").strip().lower()
    if export_format not in ("txt", "json"):
        raise HTTPException(400, "format must be txt or json")

    codes = await storage.list_redeem_codes(limit=5000, include_used=include_used or not only_unused)
    if only_unused:
        codes = [item for item in codes if not item.get("is_used")]

    exported_at = datetime.now(timezone.utc).isoformat()
    if export_format == "json":
        return {
            "exported_at": exported_at,
            "total": len(codes),
            "codes": codes,
        }

    txt_lines = [item.get("code", "") for item in codes if item.get("code")]
    filename = f"redeem-codes-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    return Response(
        content="\n".join(txt_lines),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )



@app.get("/admin/stats")
@require_login()
async def admin_stats(request: Request, time_range: str = "24h"):
    """
    获取统计数据

    Args:
        time_range: 时间范围 "24h", "7d", "30d"
    """
    active_accounts = 0
    failed_accounts = 0
    idle_accounts = 0

    for account_manager in multi_account_mgr.accounts.values():
        config = account_manager.config
        # 仅区分可用/失效（禁用或过期）
        is_failed = config.is_expired() or config.disabled
        is_active = not is_failed

        if is_failed:
            failed_accounts += 1
        elif is_active:
            active_accounts += 1
        else:
            idle_accounts += 1

    total_accounts = len(multi_account_mgr.accounts)

    # 从数据库获取统计数据
    trend_data = await stats_db.get_stats_by_time_range(time_range)
    success_count, failed_count = await stats_db.get_total_counts()

    return {
        "total_accounts": total_accounts,
        "active_accounts": active_accounts,
        "failed_accounts": failed_accounts,
        "idle_accounts": idle_accounts,
        "success_count": success_count,
        "failed_count": failed_count,
        "trend": trend_data,
    }

@app.get("/admin/accounts")
@require_login()
async def admin_get_accounts(request: Request):
    """获取所有账户的状态信息"""
    accounts_info = []
    for account_id, account_manager in multi_account_mgr.accounts.items():
        config = account_manager.config
        disabled_reason = getattr(account_manager, 'disabled_reason', None) or getattr(config, 'disabled_reason', None)
        if config.disabled:
            reason_text = str(disabled_reason or "").lower()
            if "invalid_api_key" in reason_text or "insufficient_balance" in reason_text:
                status = "API Key 失效"
            elif "403" in reason_text:
                status = "403 禁用"
            else:
                status = "已禁用"
        else:
            status = "正常"

        api_key = _get_exa_upstream_key(account_manager)
        accounts_info.append({
            "id": config.account_id,
            "status": status,
            "api_key": api_key,
            "is_available": account_manager.is_available,
            "failure_count": account_manager.failure_count,
            "disabled": config.disabled,
            "disabled_reason": disabled_reason,
            "cooldown_seconds": 0,
            "cooldown_reason": None,
            "conversation_count": account_manager.conversation_count,
            "session_usage_count": account_manager.session_usage_count,
            "trial_end": config.trial_end,
            "trial_days_remaining": config.get_trial_days_remaining(),
        })

    return {"total": len(accounts_info), "accounts": accounts_info}


@app.get("/admin/accounts-config")
@require_login()
async def admin_get_config(request: Request):
    """获取完整账户配置"""
    try:
        accounts_data = load_accounts_from_source()
        return {"accounts": accounts_data}
    except Exception as e:
        logger.error(f"[CONFIG] 获取配置失败: {str(e)}")
        raise HTTPException(500, f"获取失败: {str(e)}")

@app.put("/admin/accounts-config")
@require_login()
async def admin_update_config(request: Request, accounts_data: list = Body(...)):
    """更新整个账户配置"""
    global multi_account_mgr
    try:
        multi_account_mgr = _update_accounts_config(
            accounts_data, multi_account_mgr, http_client, USER_AGENT,
            RETRY_POLICY,
            SESSION_CACHE_TTL_SECONDS, global_stats
        )
        return {"status": "success", "message": "配置已更新", "account_count": len(multi_account_mgr.accounts)}
    except Exception as e:
        logger.error(f"[CONFIG] 更新配置失败: {str(e)}")
        raise HTTPException(500, f"更新失败: {str(e)}")

@app.post("/admin/register/start")
@require_login()
async def admin_start_register(request: Request, count: Optional[int] = Body(default=None), domain: Optional[str] = Body(default=None), mail_provider: Optional[str] = Body(default=None)):
    if not register_service:
        raise HTTPException(503, "register service unavailable")
    task = await register_service.start_register(count=count, domain=domain, mail_provider=mail_provider)
    return task.to_dict()


@app.post("/admin/register/cancel/{task_id}")
@require_login()
async def admin_cancel_register_task(request: Request, task_id: str, payload: dict = Body(default=None)):
    if not register_service:
        raise HTTPException(503, "register service unavailable")
    payload = payload or {}
    reason = payload.get("reason") or "cancelled"
    task = await register_service.cancel_task(task_id, reason=reason)
    if not task:
        raise HTTPException(404, "task not found")
    return task.to_dict()

@app.get("/admin/register/task/{task_id}")
@require_login()
async def admin_get_register_task(request: Request, task_id: str):
    if not register_service:
        raise HTTPException(503, "register service unavailable")
    task = register_service.get_task(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    return task.to_dict()

@app.get("/admin/register/current")
@require_login()
async def admin_get_current_register_task(request: Request):
    if not register_service:
        raise HTTPException(503, "register service unavailable")
    task = register_service.get_current_task()
    if not task:
        return {"status": "idle"}
    return task.to_dict()

@app.delete("/admin/accounts/{account_id}")
@require_login()
async def admin_delete_account(request: Request, account_id: str):
    """删除单个账户"""
    global multi_account_mgr
    try:
        multi_account_mgr = _delete_account(
            account_id, multi_account_mgr, http_client, USER_AGENT,
            RETRY_POLICY,
            SESSION_CACHE_TTL_SECONDS, global_stats
        )
        return {"status": "success", "message": f"账户 {account_id} 已删除", "account_count": len(multi_account_mgr.accounts)}
    except Exception as e:
        logger.error(f"[CONFIG] 删除账户失败: {str(e)}")
        raise HTTPException(500, f"删除失败: {str(e)}")

@app.put("/admin/accounts/bulk-delete")
@require_login()
async def admin_bulk_delete_accounts(request: Request, account_ids: list[str]):
    """批量删除账户，单次最多50个"""
    global multi_account_mgr

    # 数量限制验证
    if len(account_ids) > 50:
        raise HTTPException(400, f"单次最多删除50个账户，当前请求 {len(account_ids)} 个")
    if not account_ids:
        raise HTTPException(400, "账户ID列表不能为空")

    try:
        multi_account_mgr, success_count, errors = _bulk_delete_accounts(
            account_ids,
            multi_account_mgr,
            http_client,
            USER_AGENT,
            RETRY_POLICY,
            SESSION_CACHE_TTL_SECONDS,
            global_stats
        )
        return {"status": "success", "success_count": success_count, "errors": errors}
    except Exception as e:
        logger.error(f"[CONFIG] 批量删除账户失败: {str(e)}")
        raise HTTPException(500, f"删除失败: {str(e)}")

@app.put("/admin/accounts/{account_id}/disable")
@require_login()
async def admin_disable_account(request: Request, account_id: str):
    """手动禁用账户"""
    global multi_account_mgr
    try:
        multi_account_mgr = _update_account_disabled_status(
            account_id, True, multi_account_mgr
        )

        # 立即保存当前状态到数据库，防止后台任务覆盖
        if account_id in multi_account_mgr.accounts:
            account_mgr = multi_account_mgr.accounts[account_id]
            await account.save_account_cooldown_state(account_id, account_mgr)

        return {"status": "success", "message": f"账户 {account_id} 已禁用", "account_count": len(multi_account_mgr.accounts)}
    except Exception as e:
        logger.error(f"[CONFIG] 禁用账户失败: {str(e)}")
        raise HTTPException(500, f"禁用失败: {str(e)}")

@app.put("/admin/accounts/{account_id}/enable")
@require_login()
async def admin_enable_account(request: Request, account_id: str):
    """启用账户（同时重置冷却状态）"""
    global multi_account_mgr
    try:
        multi_account_mgr = _update_account_disabled_status(
            account_id, False, multi_account_mgr
        )

        # 重置运行时冷却状态（允许手动恢复冷却中的账户）
        if account_id in multi_account_mgr.accounts:
            account_mgr = multi_account_mgr.accounts[account_id]
            account_mgr.quota_cooldowns = {}
            logger.info(f"[CONFIG] 账户 {account_id} 冷却状态已重置")

            # 立即保存清空的冷却状态到数据库，防止后台任务覆盖
            await account.save_account_cooldown_state(account_id, account_mgr)

        return {"status": "success", "message": f"账户 {account_id} 已启用", "account_count": len(multi_account_mgr.accounts)}
    except Exception as e:
        logger.error(f"[CONFIG] 启用账户失败: {str(e)}")
        raise HTTPException(500, f"启用失败: {str(e)}")

@app.put("/admin/accounts/bulk-enable")
@require_login()
async def admin_bulk_enable_accounts(request: Request, account_ids: list[str]):
    """批量启用账户，单次最多50个"""
    global multi_account_mgr
    success_count, errors = _bulk_update_account_disabled_status(
        account_ids, False, multi_account_mgr
    )
    # 重置运行时错误状态
    for account_id in account_ids:
        if account_id in multi_account_mgr.accounts:
            account_mgr = multi_account_mgr.accounts[account_id]
            account_mgr.quota_cooldowns = {}
    return {"status": "success", "success_count": success_count, "errors": errors}

@app.put("/admin/accounts/bulk-disable")
@require_login()
async def admin_bulk_disable_accounts(request: Request, account_ids: list[str]):
    """批量禁用账户，单次最多50个"""
    global multi_account_mgr
    success_count, errors = _bulk_update_account_disabled_status(
        account_ids, True, multi_account_mgr
    )
    return {"status": "success", "success_count": success_count, "errors": errors}

# ---------- Auth endpoints (API) ----------
@app.get("/api/admin/settings")
@require_login()
async def admin_get_settings(request: Request):
    """获取系统设置"""
    # 返回当前配置（转换为字典格式）
    return {
        "basic": {
            "base_url": config.basic.base_url,
            "proxy_for_auth": "",
            "proxy_for_chat": "",
            "linuxdo_oauth_enabled": bool(getattr(config.basic, "linuxdo_oauth_enabled", False)),
            "linuxdo_client_id": str(getattr(config.basic, "linuxdo_client_id", "") or ""),
            "linuxdo_client_secret": str(getattr(config.basic, "linuxdo_client_secret", "") or ""),
            "linuxdo_authorize_url": str(
                getattr(config.basic, "linuxdo_authorize_url", "https://connect.linux.do/oauth2/authorize") or ""
            ),
            "linuxdo_token_url": str(
                getattr(config.basic, "linuxdo_token_url", "https://connect.linux.do/oauth2/token") or ""
            ),
            "linuxdo_userinfo_url": str(
                getattr(config.basic, "linuxdo_userinfo_url", "https://connect.linux.do/api/user") or ""
            ),
            "linuxdo_redirect_uri": str(getattr(config.basic, "linuxdo_redirect_uri", "") or ""),
            "linuxdo_scope": str(getattr(config.basic, "linuxdo_scope", "openid profile email") or ""),
            "duckmail_base_url": config.basic.duckmail_base_url,
            "duckmail_api_key": config.basic.duckmail_api_key,
            "duckmail_verify_ssl": config.basic.duckmail_verify_ssl,
            "temp_mail_provider": config.basic.temp_mail_provider,
            "moemail_base_url": config.basic.moemail_base_url,
            "moemail_api_key": config.basic.moemail_api_key,
            "moemail_domain": config.basic.moemail_domain,
            "freemail_base_url": config.basic.freemail_base_url,
            "freemail_jwt_token": config.basic.freemail_jwt_token,
            "freemail_verify_ssl": config.basic.freemail_verify_ssl,
            "freemail_domain": config.basic.freemail_domain,
            "mail_proxy_enabled": config.basic.mail_proxy_enabled,
            "exa_browser_mode": str(getattr(config.basic, "exa_browser_mode", "headless") or "headless"),
            "gptmail_base_url": config.basic.gptmail_base_url,
            "gptmail_api_key": config.basic.gptmail_api_key,
            "gptmail_verify_ssl": config.basic.gptmail_verify_ssl,
            "gptmail_domain": config.basic.gptmail_domain,
            "cfmail_base_url": config.basic.cfmail_base_url,
            "cfmail_api_key": config.basic.cfmail_api_key,
            "cfmail_verify_ssl": config.basic.cfmail_verify_ssl,
            "cfmail_domain": config.basic.cfmail_domain,
            "refresh_window_hours": 0,
            "register_default_count": config.basic.register_default_count,
            "register_domain": config.basic.register_domain,
            "image_expire_hours": -1,
        },
        "image_generation": {
            "enabled": config.image_generation.enabled,
            "supported_models": config.image_generation.supported_models,
            "output_format": config.image_generation.output_format
        },
        "video_generation": {
            "output_format": config.video_generation.output_format
        },
        "retry": {
            "max_account_switch_tries": config.retry.max_account_switch_tries,
            "text_rate_limit_cooldown_seconds": config.retry.text_rate_limit_cooldown_seconds,
            "images_rate_limit_cooldown_seconds": config.retry.images_rate_limit_cooldown_seconds,
            "videos_rate_limit_cooldown_seconds": config.retry.videos_rate_limit_cooldown_seconds,
            "session_cache_ttl_seconds": config.retry.session_cache_ttl_seconds,
            "auto_refresh_accounts_seconds": 0,
            "scheduled_refresh_enabled": False,
            "scheduled_refresh_interval_minutes": 0,
            "scheduled_refresh_cron": "",
            "refresh_batch_size": 1,
            "refresh_batch_interval_minutes": 60,
            "refresh_cooldown_hours": 48,
        },
        "quota_limits": {
            "enabled": False,
            "text_daily_limit": 0,
            "images_daily_limit": 0,
            "videos_daily_limit": 0
        },
        "public_display": {
            "logo_url": config.public_display.logo_url,
            "chat_url": config.public_display.chat_url
        },
        "session": {
            "expire_hours": config.session.expire_hours
        }
    }

@app.put("/api/admin/settings")
@require_login()
async def admin_update_settings(request: Request, new_settings: dict = Body(...)):
    """更新系统设置"""
    global PROXY_FOR_AUTH, PROXY_FOR_CHAT, BASE_URL, LOGO_URL, CHAT_URL
    global LINUXDO_OAUTH_ENABLED, LINUXDO_CLIENT_ID, LINUXDO_CLIENT_SECRET
    global LINUXDO_AUTHORIZE_URL, LINUXDO_TOKEN_URL, LINUXDO_USERINFO_URL, LINUXDO_REDIRECT_URI, LINUXDO_SCOPE
    global MAX_ACCOUNT_SWITCH_TRIES
    global RETRY_POLICY
    global SESSION_CACHE_TTL_SECONDS
    global SESSION_EXPIRE_HOURS, multi_account_mgr, http_client, http_client_chat, http_client_auth

    try:
        basic = dict(new_settings.get("basic") or {})
        basic.setdefault("duckmail_base_url", config.basic.duckmail_base_url)
        basic.setdefault("duckmail_api_key", config.basic.duckmail_api_key)
        basic.setdefault("duckmail_verify_ssl", config.basic.duckmail_verify_ssl)
        basic.setdefault("temp_mail_provider", config.basic.temp_mail_provider)
        basic.setdefault("moemail_base_url", config.basic.moemail_base_url)
        basic.setdefault("moemail_api_key", config.basic.moemail_api_key)
        basic.setdefault("moemail_domain", config.basic.moemail_domain)
        basic.setdefault("freemail_base_url", config.basic.freemail_base_url)
        basic.setdefault("freemail_jwt_token", config.basic.freemail_jwt_token)
        basic.setdefault("freemail_verify_ssl", config.basic.freemail_verify_ssl)
        basic.setdefault("freemail_domain", config.basic.freemail_domain)
        basic.setdefault("mail_proxy_enabled", config.basic.mail_proxy_enabled)
        basic.setdefault("exa_browser_mode", str(getattr(config.basic, "exa_browser_mode", "headless") or "headless"))
        basic.setdefault("gptmail_base_url", config.basic.gptmail_base_url)
        basic.setdefault("gptmail_api_key", config.basic.gptmail_api_key)
        basic.setdefault("gptmail_verify_ssl", config.basic.gptmail_verify_ssl)
        basic.setdefault("gptmail_domain", config.basic.gptmail_domain)
        basic.setdefault("cfmail_base_url", config.basic.cfmail_base_url)
        basic.setdefault("cfmail_api_key", config.basic.cfmail_api_key)
        basic.setdefault("cfmail_verify_ssl", config.basic.cfmail_verify_ssl)
        basic.setdefault("cfmail_domain", config.basic.cfmail_domain)
        basic.setdefault("linuxdo_oauth_enabled", bool(getattr(config.basic, "linuxdo_oauth_enabled", False)))
        basic.setdefault("linuxdo_client_id", str(getattr(config.basic, "linuxdo_client_id", "") or ""))
        basic.setdefault("linuxdo_client_secret", str(getattr(config.basic, "linuxdo_client_secret", "") or ""))
        basic.setdefault(
            "linuxdo_authorize_url",
            str(getattr(config.basic, "linuxdo_authorize_url", "https://connect.linux.do/oauth2/authorize") or ""),
        )
        basic.setdefault(
            "linuxdo_token_url",
            str(getattr(config.basic, "linuxdo_token_url", "https://connect.linux.do/oauth2/token") or ""),
        )
        basic.setdefault(
            "linuxdo_userinfo_url",
            str(getattr(config.basic, "linuxdo_userinfo_url", "https://connect.linux.do/api/user") or ""),
        )
        basic.setdefault("linuxdo_redirect_uri", str(getattr(config.basic, "linuxdo_redirect_uri", "") or ""))
        basic.setdefault("linuxdo_scope", str(getattr(config.basic, "linuxdo_scope", "openid profile email") or ""))
        basic.setdefault("refresh_window_hours", 0)
        basic.setdefault("register_default_count", config.basic.register_default_count)
        basic.setdefault("register_domain", config.basic.register_domain)
        basic.setdefault("image_expire_hours", -1)
        basic.pop("api_key", None)
        basic["proxy_for_auth"] = ""
        basic["proxy_for_chat"] = ""
        basic["refresh_window_hours"] = 0
        basic["image_expire_hours"] = -1
        if not isinstance(basic.get("register_domain"), str):
            basic["register_domain"] = ""
        basic["linuxdo_oauth_enabled"] = bool(basic.get("linuxdo_oauth_enabled", False))
        basic["linuxdo_client_id"] = str(basic.get("linuxdo_client_id") or "").strip()
        basic["linuxdo_client_secret"] = str(basic.get("linuxdo_client_secret") or "").strip()
        basic["linuxdo_authorize_url"] = str(
            basic.get("linuxdo_authorize_url") or "https://connect.linux.do/oauth2/authorize"
        ).strip()
        basic["linuxdo_token_url"] = str(
            basic.get("linuxdo_token_url") or "https://connect.linux.do/oauth2/token"
        ).strip()
        basic["linuxdo_userinfo_url"] = str(
            basic.get("linuxdo_userinfo_url") or "https://connect.linux.do/api/user"
        ).strip()
        basic["linuxdo_redirect_uri"] = str(basic.get("linuxdo_redirect_uri") or "").strip()
        basic["linuxdo_scope"] = str(basic.get("linuxdo_scope") or "openid profile email").strip()
        basic["exa_browser_mode"] = str(basic.get("exa_browser_mode") or "headless").strip().lower()
        if basic["exa_browser_mode"] not in ("headless", "headful"):
            basic["exa_browser_mode"] = "headless"
        basic.pop("duckmail_proxy", None)
        basic.pop("browser_engine", None)
        basic.pop("browser_mode", None)
        new_settings["basic"] = basic

        image_generation = dict(new_settings.get("image_generation") or {})
        output_format = str(image_generation.get("output_format") or config_manager.image_output_format).lower()
        if output_format not in ("base64", "url"):
            output_format = "base64"
        image_generation["output_format"] = output_format
        new_settings["image_generation"] = image_generation

        video_generation = dict(new_settings.get("video_generation") or {})
        video_output_format = str(video_generation.get("output_format") or config_manager.video_output_format).lower()
        if video_output_format not in ("html", "url", "markdown"):
            video_output_format = "html"
        video_generation["output_format"] = video_output_format
        new_settings["video_generation"] = video_generation

        retry = dict(new_settings.get("retry") or {})
        retry.setdefault("auto_refresh_accounts_seconds", config.retry.auto_refresh_accounts_seconds)
        retry.setdefault("scheduled_refresh_enabled", config.retry.scheduled_refresh_enabled)
        retry.setdefault("scheduled_refresh_interval_minutes", config.retry.scheduled_refresh_interval_minutes)
        retry.setdefault("text_rate_limit_cooldown_seconds", config.retry.text_rate_limit_cooldown_seconds)
        retry.setdefault("images_rate_limit_cooldown_seconds", config.retry.images_rate_limit_cooldown_seconds)
        retry.setdefault("videos_rate_limit_cooldown_seconds", config.retry.videos_rate_limit_cooldown_seconds)
        retry["auto_refresh_accounts_seconds"] = 0
        retry["scheduled_refresh_enabled"] = False
        retry["scheduled_refresh_interval_minutes"] = 0
        retry["scheduled_refresh_cron"] = ""
        retry["refresh_batch_size"] = 1
        retry["refresh_batch_interval_minutes"] = 60
        retry["refresh_cooldown_hours"] = 48
        new_settings["retry"] = retry

        # 配额上限配置
        quota_limits = dict(new_settings.get("quota_limits") or {})
        quota_limits["enabled"] = False
        quota_limits["text_daily_limit"] = 0
        quota_limits["images_daily_limit"] = 0
        quota_limits["videos_daily_limit"] = 0
        new_settings["quota_limits"] = quota_limits

        # 保存旧配置用于对比
        old_proxy_for_auth = PROXY_FOR_AUTH
        old_proxy_for_chat = PROXY_FOR_CHAT
        old_retry_config = {
            "text_rate_limit_cooldown_seconds": RETRY_POLICY.cooldowns.text,
            "images_rate_limit_cooldown_seconds": RETRY_POLICY.cooldowns.images,
            "videos_rate_limit_cooldown_seconds": RETRY_POLICY.cooldowns.videos,
            "session_cache_ttl_seconds": SESSION_CACHE_TTL_SECONDS
        }

        # 保存到 YAML
        config_manager.save_yaml(new_settings)

        # 热更新配置
        config_manager.reload()

        # 更新全局变量（实时生效）
        _proxy_auth, _no_proxy_auth = parse_proxy_setting(config.basic.proxy_for_auth)
        _proxy_chat, _no_proxy_chat = parse_proxy_setting(config.basic.proxy_for_chat)
        PROXY_FOR_AUTH = _proxy_auth
        PROXY_FOR_CHAT = _proxy_chat
        _NO_PROXY = ",".join(filter(None, {_no_proxy_auth, _no_proxy_chat}))
        if _NO_PROXY:
            os.environ["NO_PROXY"] = _NO_PROXY
        else:
            os.environ.pop("NO_PROXY", None)
        BASE_URL = config.basic.base_url
        LOGO_URL = config.public_display.logo_url
        CHAT_URL = config.public_display.chat_url
        LINUXDO_OAUTH_ENABLED = bool(getattr(config.basic, "linuxdo_oauth_enabled", False))
        LINUXDO_CLIENT_ID = str(getattr(config.basic, "linuxdo_client_id", "") or "").strip()
        LINUXDO_CLIENT_SECRET = str(getattr(config.basic, "linuxdo_client_secret", "") or "").strip()
        LINUXDO_AUTHORIZE_URL = str(
            getattr(config.basic, "linuxdo_authorize_url", "https://connect.linux.do/oauth2/authorize") or ""
        ).strip()
        LINUXDO_TOKEN_URL = str(
            getattr(config.basic, "linuxdo_token_url", "https://connect.linux.do/oauth2/token") or ""
        ).strip()
        LINUXDO_USERINFO_URL = str(
            getattr(config.basic, "linuxdo_userinfo_url", "https://connect.linux.do/api/user") or ""
        ).strip()
        LINUXDO_REDIRECT_URI = str(getattr(config.basic, "linuxdo_redirect_uri", "") or "").strip()
        LINUXDO_SCOPE = str(getattr(config.basic, "linuxdo_scope", "openid profile email") or "openid profile email").strip()
        MAX_ACCOUNT_SWITCH_TRIES = config.retry.max_account_switch_tries
        RETRY_POLICY = build_retry_policy()
        SESSION_CACHE_TTL_SECONDS = config.retry.session_cache_ttl_seconds
        SESSION_EXPIRE_HOURS = config.session.expire_hours

        # 检查是否需要重建 HTTP 客户端（代理变化）
        if old_proxy_for_auth != PROXY_FOR_AUTH or old_proxy_for_chat != PROXY_FOR_CHAT:
            logger.info(f"[CONFIG] Proxy configuration changed, rebuilding HTTP clients")
            await http_client.aclose()
            await http_client_chat.aclose()
            await http_client_auth.aclose()

            # 重新创建对话客户端
            http_client = httpx.AsyncClient(
                proxy=(PROXY_FOR_CHAT or None),
                verify=False,
                http2=False,
                timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=60.0),
                limits=httpx.Limits(
                    max_keepalive_connections=100,
                    max_connections=200
                )
            )

            # 重新创建对话流式客户端
            http_client_chat = httpx.AsyncClient(
                proxy=(PROXY_FOR_CHAT or None),
                verify=False,
                http2=False,
                timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=60.0),
                limits=httpx.Limits(
                    max_keepalive_connections=100,
                    max_connections=200
                )
            )

            # 重新创建账户操作客户端
            http_client_auth = httpx.AsyncClient(
                proxy=(PROXY_FOR_AUTH or None),
                verify=False,
                http2=False,
                timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=60.0),
                limits=httpx.Limits(
                    max_keepalive_connections=100,
                    max_connections=200
                )
            )

            # 打印新的代理配置
            logger.info(f"[PROXY] Account operations (register): {PROXY_FOR_AUTH if PROXY_FOR_AUTH else 'disabled'}")
            logger.info(f"[PROXY] Chat operations (JWT/session/messages): {PROXY_FOR_CHAT if PROXY_FOR_CHAT else 'disabled'}")

            # 更新所有账户的 http_client 引用（对话用）
            multi_account_mgr.update_http_client(http_client)

            # 更新注册/登录服务的 http_client 引用（账户操作用）
            if register_service:
                register_service.http_client = http_client_auth

        # 检查是否需要更新账户管理器配置（重试策略变化）
        retry_changed = (
            old_retry_config["text_rate_limit_cooldown_seconds"] != RETRY_POLICY.cooldowns.text or
            old_retry_config["images_rate_limit_cooldown_seconds"] != RETRY_POLICY.cooldowns.images or
            old_retry_config["videos_rate_limit_cooldown_seconds"] != RETRY_POLICY.cooldowns.videos or
            old_retry_config["session_cache_ttl_seconds"] != SESSION_CACHE_TTL_SECONDS
        )

        if retry_changed:
            logger.info(f"[CONFIG] 重试策略已变化，更新账户管理器配置")
            # 更新所有账户管理器的配置
            multi_account_mgr.cache_ttl = SESSION_CACHE_TTL_SECONDS
            for account_id, account_mgr in multi_account_mgr.accounts.items():
                account_mgr.apply_retry_policy(RETRY_POLICY)
            if register_service:
                register_service.retry_policy = RETRY_POLICY

        logger.info(f"[CONFIG] 系统设置已更新并实时生效")
        return {"status": "success", "message": "设置已保存并实时生效！"}
    except Exception as e:
        logger.error(f"[CONFIG] 更新设置失败: {str(e)}")
        raise HTTPException(500, f"更新失败: {str(e)}")


@app.post("/api/admin/exa/browser-check")
@require_login()
async def admin_check_exa_browser(request: Request, payload: dict = Body(default=None)):
    """检查当前浏览器模式下 Exa 登录页可达性。"""
    requested_mode = str((payload or {}).get("browser_mode") or "").strip().lower()
    if requested_mode not in ("", "headless", "headful"):
        requested_mode = ""

    proxy_for_auth, _ = parse_proxy_setting(config.basic.proxy_for_auth)
    automation = ExaAutomation(
        proxy=proxy_for_auth,
        timeout_ms=45_000,
        headless=(requested_mode == "headless") if requested_mode else None,
    )
    return await asyncio.to_thread(automation.check_browser_environment)


@app.get("/api/admin/database/export")
@require_login()
async def admin_export_database(request: Request):
    """导出当前 SQLite 数据库文件。"""
    _require_user_storage()
    if storage.get_database_backend() != "sqlite":
        raise HTTPException(400, "Only sqlite backend supports browser export/import")

    db_bytes = storage.export_sqlite_db_bytes()
    if not db_bytes:
        raise HTTPException(500, "Export database failed")

    filename = f"exafree-db-{datetime.now().strftime('%Y%m%d-%H%M%S')}.db"
    return Response(
        content=db_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/admin/database/import")
@require_login()
async def admin_import_database(request: Request, file: UploadFile = File(...)):
    """导入并覆盖当前 SQLite 数据库文件。"""
    _require_user_storage()
    if storage.get_database_backend() != "sqlite":
        raise HTTPException(400, "Only sqlite backend supports browser export/import")

    if not file:
        raise HTTPException(400, "database file required")
    content = await file.read()
    ok, msg = storage.import_sqlite_db_bytes(content)
    if not ok:
        raise HTTPException(400, f"Import database failed: {msg}")
    logger.info("[DB] SQLite database imported and replaced by admin upload")
    return {"success": True, "message": "Database imported and replaced successfully"}


@app.get("/admin/log")
@require_login()
async def admin_get_logs(
    request: Request,
    limit: int = 300,
    level: str = None,
    search: str = None,
    start_time: str = None,
    end_time: str = None
):
    with log_lock:
        logs = list(log_buffer)

    stats_by_level = {}
    error_logs = []
    request_ids = set()
    legacy_chat_count = 0
    for log in logs:
        level_name = log.get("level", "INFO")
        stats_by_level[level_name] = stats_by_level.get(level_name, 0) + 1
        if level_name in ["ERROR", "CRITICAL"]:
            error_logs.append(log)
        message = log.get("message", "")
        request_ids.update(re.findall(r"\[req_([a-z0-9]+)\]", message, flags=re.I))
        if "收到请求" in message or "Received request" in message:
            legacy_chat_count += 1

    request_count = len(request_ids) if request_ids else legacy_chat_count

    if level:
        level = level.upper()
        logs = [log for log in logs if log["level"] == level]
    if search:
        logs = [log for log in logs if search.lower() in log["message"].lower()]
    if start_time:
        logs = [log for log in logs if log["time"] >= start_time]
    if end_time:
        logs = [log for log in logs if log["time"] <= end_time]

    limit = min(limit, log_buffer.maxlen)
    filtered_logs = logs[-limit:]

    return {
        "total": len(filtered_logs),
        "limit": limit,
        "filters": {"level": level, "search": search, "start_time": start_time, "end_time": end_time},
        "logs": filtered_logs,
        "stats": {
            "memory": {"total": len(log_buffer), "by_level": stats_by_level, "capacity": log_buffer.maxlen},
            "errors": {"count": len(error_logs), "recent": error_logs[-10:]},
            "request_count": request_count,
            "chat_count": request_count,
        }
    }

@app.delete("/admin/log")
@require_login()
async def admin_clear_logs(request: Request, confirm: str = None):
    if confirm != "yes":
        raise HTTPException(400, "需要 confirm=yes 参数确认清空操作")
    with log_lock:
        cleared_count = len(log_buffer)
        log_buffer.clear()
    logger.info("[LOG] 日志已清空")
    return {"status": "success", "message": "已清空内存日志", "cleared_count": cleared_count}

@app.get("/admin/task-history")
@require_login()
async def admin_get_task_history(request: Request, limit: int = 100):
    """获取任务历史记录"""
    _load_task_history()
    with task_history_lock:
        history = list(task_history)

    live_entries = []
    try:
        if register_service:
            current_register = register_service.get_current_task()
            if current_register and current_register.status in ("running", "pending"):
                live_entries.append(_build_history_entry("register", current_register.to_dict(), is_live=True))
    except Exception as exc:
        logger.warning(f"[HISTORY] build live entries failed: {exc}")

    merged = {}
    for entry in live_entries + history:
        entry_id = entry.get("id") or str(uuid.uuid4())
        if entry_id not in merged:
            merged[entry_id] = entry

    # 按创建时间倒序排序
    history = list(merged.values())
    history.sort(key=lambda x: x.get("created_at", 0), reverse=True)

    # 限制返回数量
    limit = min(limit, 100)
    return {
        "total": len(history),
        "limit": limit,
        "history": history[:limit]
    }

@app.delete("/admin/task-history")
@require_login()
async def admin_clear_task_history(request: Request, confirm: str = None):
    """清空任务历史记录"""
    if confirm != "yes":
        raise HTTPException(400, "需要 confirm=yes 参数确认清空操作")
    with task_history_lock:
        cleared_count = len(task_history)
        task_history.clear()
        _persist_task_history()
    logger.info("[HISTORY] 任务历史已清空")
    return {"status": "success", "message": "已清空任务历史", "cleared_count": cleared_count}

# ---------- Auth endpoints (API) ----------

research_task_account_map: Dict[str, Dict[str, str]] = {}
research_task_lock = asyncio.Lock()
exa_round_robin_lock = Lock()
exa_round_robin_counter = 0


def _get_exa_upstream_key(account_manager: AccountManager) -> str:
    key = (account_manager.config.exa_api_key or "").strip()
    if not key:
        key = (account_manager.config.secure_c_ses or "").strip()
    return key


def _extract_client_api_token(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[str]:
    if x_api_key:
        return x_api_key
    if authorization:
        if authorization.startswith("Bearer "):
            return authorization[7:]
        return authorization
    return None


async def _authenticate_client_api_user(authorization: Optional[str], x_api_key: Optional[str]) -> dict:
    token = _extract_client_api_token(authorization, x_api_key)
    if not token:
        raise HTTPException(401, "Missing Authorization header")

    user_ctx = await storage.authenticate_api_key(hash_api_key(token))
    if user_ctx:
        return user_ctx
    raise HTTPException(401, "Invalid API Key")


async def _enforce_user_request_limits(client_user: dict) -> None:
    role = (client_user or {}).get("role", "")
    user_id = (client_user or {}).get("user_id", "")
    if role == "admin" or user_id in ("legacy", ""):
        return
    limits = _resolve_role_limits(role)
    if not limits:
        return
    window_minutes = int(limits["window_minutes"])
    counts = await storage.get_user_request_counts(
        user_id,
        day_start_ts=_day_start_ts_local(),
        window_start_ts=int(time.time()) - window_minutes * 60,
    )
    daily_limit = int(limits["daily_limit"])
    window_max_calls = int(limits["window_max_calls"])
    if counts["day_count"] >= daily_limit:
        raise HTTPException(
            429,
            f"Daily limit exceeded ({counts['day_count']}/{daily_limit})",
        )
    if counts["window_count"] >= window_max_calls:
        raise HTTPException(
            429,
            f"Rate limit exceeded ({counts['window_count']}/{window_max_calls} in {window_minutes}m)",
        )


def _is_exa_account_available(account_manager: AccountManager) -> bool:
    if account_manager.config.disabled:
        return False
    if not _get_exa_upstream_key(account_manager):
        return False
    return True


def _pick_exa_account(
    *,
    stick_account_id: Optional[str],
    tried_accounts: set[str],
) -> Optional[AccountManager]:
    global exa_round_robin_counter

    if stick_account_id and stick_account_id in multi_account_mgr.accounts:
        stick = multi_account_mgr.accounts[stick_account_id]
        if _is_exa_account_available(stick) and stick_account_id not in tried_accounts:
            return stick

    candidates = [
        mgr for mgr in multi_account_mgr.accounts.values()
        if _is_exa_account_available(mgr) and mgr.config.account_id not in tried_accounts
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda mgr: mgr.config.account_id)
    with exa_round_robin_lock:
        index = exa_round_robin_counter % len(candidates)
        exa_round_robin_counter += 1
    return candidates[index]


async def _mark_exa_account_disabled(account_id: str, reason: str) -> None:
    global multi_account_mgr
    try:
        multi_account_mgr = _update_account_disabled_status(account_id, True, multi_account_mgr)
        if account_id in multi_account_mgr.accounts:
            multi_account_mgr.accounts[account_id].disabled_reason = reason
            await account.save_account_cooldown_state(account_id, multi_account_mgr.accounts[account_id])
        logger.warning(f"[EXA] 账号已禁用: {account_id} ({reason})")
    except Exception as exc:
        logger.warning(f"[EXA] 禁用账号失败 {account_id}: {exc}")


def _copy_exa_response_headers(resp: httpx.Response) -> Dict[str, str]:
    headers = {"Content-Type": resp.headers.get("content-type", "application/json")}
    cache_control = resp.headers.get("cache-control")
    if cache_control:
        headers["Cache-Control"] = cache_control
    return headers


async def _proxy_exa_request(
    request: Request,
    path: str,
    *,
    method: Optional[str] = None,
    body_override: Optional[bytes] = None,
    stick_account_id: Optional[str] = None,
    service_name: str = "exa-answer",
    max_retries: int = 3,
    client_user: Optional[dict] = None,
) -> tuple[Response, Optional[str]]:
    request_id = str(uuid.uuid4())[:6]
    request_started_at = time.time()
    body_bytes = body_override if body_override is not None else await request.body()
    query = request.url.query
    upstream_url = f"https://api.exa.ai{path}"
    if query:
        upstream_url = f"{upstream_url}?{query}"

    attempts = 0
    last_error_body = None
    last_status = 502
    tried_accounts = set()
    logger.info(f"[EXA] [req_{request_id}] 收到请求: {service_name}")

    async def _record_request_stats(status: str, status_code: Optional[int], elapsed_ms: Optional[int] = None) -> None:
        try:
            success_elapsed = elapsed_ms if status == "success" else None
            await stats_db.insert_request_log(
                timestamp=time.time(),
                model=service_name,
                ttfb_ms=success_elapsed,
                total_ms=success_elapsed,
                status=status,
                status_code=status_code,
                user_id=(client_user or {}).get("user_id"),
                user_name=(client_user or {}).get("username"),
            )
        except Exception as exc:
            logger.debug(f"[STATS] 写入请求统计失败: {exc}")

    while attempts < max_retries:
        picked_id = stick_account_id if attempts == 0 and stick_account_id else None
        account_manager = _pick_exa_account(stick_account_id=picked_id, tried_accounts=tried_accounts)
        if not account_manager:
            break

        account_id = account_manager.config.account_id
        tried_accounts.add(account_id)
        if attempts == 0:
            logger.info(f"[EXA] [{account_id}] [req_{request_id}] 选择账户")
        else:
            logger.info(f"[EXA] [{account_id}] [req_{request_id}] 切换账户")

        exa_key = _get_exa_upstream_key(account_manager)
        if not exa_key:
            await _mark_exa_account_disabled(account_id, "missing_exa_api_key")
            attempts += 1
            continue

        headers = {"x-api-key": exa_key}
        content_type = request.headers.get("content-type")
        if content_type:
            headers["content-type"] = content_type

        try:
            resp = await http_client.request(
                method=method or request.method,
                url=upstream_url,
                headers=headers,
                content=body_bytes if body_bytes else None,
            )
        except Exception as exc:
            logger.warning(
                f"[EXA] [{account_id}] [req_{request_id}] 失败 (尝试 {attempts + 1}/{max_retries}): "
                f"{type(exc).__name__}: {exc}"
            )
            attempts += 1
            continue

        status = resp.status_code
        if 200 <= status < 300:
            account_manager.conversation_count += 1
            uptime_tracker.record_request(service_name, True, status_code=status)
            elapsed_ms = int((time.time() - request_started_at) * 1000)
            logger.info(f"[EXA] [{account_id}] [req_{request_id}] 响应完成: {elapsed_ms / 1000:.2f}秒")
            await _record_request_stats("success", status, elapsed_ms)
            return (
                Response(content=resp.content, status_code=status, headers=_copy_exa_response_headers(resp)),
                account_id,
            )

        uptime_tracker.record_request(service_name, False, status_code=status)
        logger.warning(f"[EXA] [{account_id}] [req_{request_id}] 失败 (尝试 {attempts + 1}/{max_retries}): HTTP {status}")
        last_status = status
        last_error_body = resp.content

        if status in (401, 402):
            reason = "invalid_api_key" if status == 401 else "insufficient_balance"
            await _mark_exa_account_disabled(account_id, reason)

        if status in (400, 403):
            break

        attempts += 1

    if last_error_body is not None:
        await _record_request_stats("failed", last_status)
        return (
            Response(content=last_error_body, status_code=last_status, media_type="application/json"),
            None,
        )
    await _record_request_stats("error", 503)
    raise HTTPException(503, "No available Exa accounts")


@app.post("/search")
async def exa_search(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    client_user = await _authenticate_client_api_user(authorization, x_api_key)
    await _enforce_user_request_limits(client_user)
    proxied, _ = await _proxy_exa_request(
        request,
        "/search",
        service_name="exa-search",
        client_user=client_user,
    )
    return proxied


@app.post("/contents")
async def exa_contents(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    client_user = await _authenticate_client_api_user(authorization, x_api_key)
    await _enforce_user_request_limits(client_user)
    proxied, _ = await _proxy_exa_request(
        request,
        "/contents",
        service_name="exa-contents",
        client_user=client_user,
    )
    return proxied


@app.post("/findSimilar")
async def exa_find_similar(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    client_user = await _authenticate_client_api_user(authorization, x_api_key)
    await _enforce_user_request_limits(client_user)
    proxied, _ = await _proxy_exa_request(
        request,
        "/findSimilar",
        service_name="exa-findSimilar",
        client_user=client_user,
    )
    return proxied


@app.post("/answer")
async def exa_answer(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    client_user = await _authenticate_client_api_user(authorization, x_api_key)
    await _enforce_user_request_limits(client_user)
    proxied, _ = await _proxy_exa_request(
        request,
        "/answer",
        service_name="exa-answer",
        client_user=client_user,
    )
    return proxied


@app.post("/research/v1")
async def exa_research_create(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    client_user = await _authenticate_client_api_user(authorization, x_api_key)
    await _enforce_user_request_limits(client_user)
    proxied, account_id = await _proxy_exa_request(
        request,
        "/research/v1",
        service_name="exa-research",
        client_user=client_user,
    )

    if proxied.status_code >= 200 and proxied.status_code < 300 and account_id:
        try:
            data = json.loads(proxied.body.decode("utf-8"))
            task_id = data.get("researchId") or data.get("id")
            if task_id:
                async with research_task_lock:
                    research_task_account_map[str(task_id)] = {
                        "account_id": account_id,
                        "user_id": client_user["user_id"],
                    }
        except Exception:
            pass
    return proxied


@app.get("/research/v1")
async def exa_research_list(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    client_user = await _authenticate_client_api_user(authorization, x_api_key)
    await _enforce_user_request_limits(client_user)
    proxied, _ = await _proxy_exa_request(
        request,
        "/research/v1",
        method="GET",
        service_name="exa-research",
        client_user=client_user,
    )
    return proxied


@app.get("/research/v1/{research_id}")
async def exa_research_get(
    request: Request,
    research_id: str,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    client_user = await _authenticate_client_api_user(authorization, x_api_key)
    await _enforce_user_request_limits(client_user)
    async with research_task_lock:
        mapping = research_task_account_map.get(research_id)
        stick_id = None
        if mapping and mapping.get("user_id") == client_user["user_id"]:
            stick_id = mapping.get("account_id")
    proxied, _ = await _proxy_exa_request(
        request,
        f"/research/v1/{research_id}",
        method="GET",
        service_name="exa-research",
        stick_account_id=stick_id,
        client_user=client_user,
    )
    return proxied

# ---------- 公开端点（无需认证） ----------
@app.get("/public/uptime")
async def get_public_uptime(days: int = 90):
    """获取 Uptime 监控数据（JSON格式）"""
    if days < 1 or days > 90:
        days = 90
    return await uptime_tracker.get_uptime_summary(days)


@app.get("/public/stats")
async def get_public_stats():
    """获取公开统计信息"""
    async with stats_lock:
        # 清理1小时前的请求时间戳
        current_time = time.time()
        recent_requests = [
            ts for ts in global_stats["request_timestamps"]
            if current_time - ts < 3600
        ]

        # 计算每分钟请求数
        recent_minute = [
            ts for ts in recent_requests
            if current_time - ts < 60
        ]
        requests_per_minute = len(recent_minute)

        # 计算负载状态
        if requests_per_minute < 10:
            load_status = "low"
            load_color = "#10b981"  # 绿色
        elif requests_per_minute < 30:
            load_status = "medium"
            load_color = "#f59e0b"  # 黄色
        else:
            load_status = "high"
            load_color = "#ef4444"  # 红色

        return {
            "total_visitors": global_stats["total_visitors"],
            "total_requests": global_stats["total_requests"],
            "requests_per_minute": requests_per_minute,
            "load_status": load_status,
            "load_color": load_color
        }

@app.get("/public/display")
async def get_public_display():
    """获取公开展示信息"""
    return {
        "logo_url": LOGO_URL,
        "chat_url": CHAT_URL
    }

@app.get("/public/log")
async def get_public_logs(request: Request, limit: int = 100):
    try:
        # 基于IP的访问统计（24小时内去重）
        client_ip = request.client.host
        current_time = time.time()

        async with stats_lock:
            # 清理24小时前的IP记录
            if "visitor_ips" not in global_stats:
                global_stats["visitor_ips"] = {}
            global_stats["visitor_ips"] = {
                ip: timestamp for ip, timestamp in global_stats["visitor_ips"].items()
                if current_time - timestamp <= 86400
            }

            # 记录新访问（24小时内同一IP只计数一次）
            if client_ip not in global_stats["visitor_ips"]:
                global_stats["visitor_ips"][client_ip] = current_time
                global_stats["total_visitors"] = global_stats.get("total_visitors", 0) + 1

            global_stats.setdefault("recent_conversations", [])
            await save_stats(global_stats)

            stored_logs = list(global_stats.get("recent_conversations", []))

        sanitized_logs = get_sanitized_logs(limit=min(limit, 1000))

        log_map = {log.get("request_id"): log for log in sanitized_logs}
        for log in stored_logs:
            request_id = log.get("request_id")
            if request_id and request_id not in log_map:
                log_map[request_id] = log

        def get_log_ts(item: dict) -> float:
            if "start_ts" in item:
                return float(item["start_ts"])
            try:
                return datetime.strptime(item.get("start_time", ""), "%Y-%m-%d %H:%M:%S").timestamp()
            except Exception:
                return 0.0

        merged_logs = sorted(log_map.values(), key=get_log_ts, reverse=True)[:min(limit, 1000)]
        output_logs = []
        for log in merged_logs:
            if "start_ts" in log:
                log = dict(log)
                log.pop("start_ts", None)
            output_logs.append(log)

        return {
            "total": len(output_logs),
            "logs": output_logs
        }
    except Exception as e:
        logger.error(f"[LOG] 获取公开日志失败: {e}")
        return {"total": 0, "logs": [], "error": str(e)}

# ---------- 全局 404 处理（必须在最后） ----------

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """全局 404 处理器"""
    return JSONResponse(
        status_code=404,
        content={"detail": "Not Found"}
    )

# SPA fallback: 所有非 API 路由返回 index.html
if not DISABLE_ADMIN_PANEL:
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """处理所有前端路由，返回 index.html"""
        index_path = os.path.join("static", "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        raise HTTPException(404, "Not Found")


if __name__ == "__main__":
    import uvicorn
    import subprocess
    import sys

    port = int(os.getenv("PORT", "7860"))

    # 检查端口占用并终止进程
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                f'netstat -ano | findstr ":{port}" | findstr "LISTENING"',
                shell=True, capture_output=True, text=True
            )
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        print(f"端口 {port} 被进程 {pid} 占用，正在终止...")
                        subprocess.run(f"taskkill /F /PID {pid}", shell=True)
        else:
            result = subprocess.run(
                f"lsof -ti:{port}",
                shell=True, capture_output=True, text=True
            )
            if result.stdout:
                pid = result.stdout.strip()
                print(f"端口 {port} 被进程 {pid} 占用，正在终止...")
                subprocess.run(f"kill -9 {pid}", shell=True)
    except Exception as e:
        print(f"检查端口占用时出错: {e}")

    uvicorn.run(app, host="0.0.0.0", port=port)
