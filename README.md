<p align="center">
  <img src="docs/logo.svg" width="120" alt="ExaFree logo" />
</p>
<h1 align="center">ExaFree</h1>
<p align="center">
  <strong>简体中文</strong>
</p>
<p align="center">
  Exa API 代理与管理面板，支持多账号、用户级 API Key、Linux DO OAuth 登录/注册。
</p>

## 开源协议与声明

- 协议：MIT（见 [LICENSE](LICENSE)）
- 免责声明与使用限制：见 [docs/DISCLAIMER.md](docs/DISCLAIMER.md)
- 本项目基于[gemini-business2api](https://github.com/Dreamy-rain/gemini-business2api)魔改而来，感谢大佬！

---

## 核心功能

- Exa API 代理：`/search`、`/answer`、`/contents`、`/findSimilar`、`/research/v1`
- MCP 工具入口：`/mcp`（HTTP）
- 多账号轮询与失败切换
- 管理面板（账号、用户、策略、日志、监控、系统设置）
- 用户系统（会话登录、用户 API Key、角色/限流）
- 升级兑换码（批量生成、导入、导出、一次性使用）
- Linux DO OAuth2 登录/注册
- SQLite / PostgreSQL 持久化

---

## Docker部署（推荐）

```bash
docker pull ghcr.io/chengtx809/exafree:main
docker run --rm -p 7860:7860 -v ./data:/app/data ghcr.io/chengtx809/exafree:main
```

访问：

- 管理面板：`http://localhost:7860/#/login`
- 健康检查：`http://localhost:7860/health`

默认管理员（首次启动自动创建）：

- 用户名：`admin`
- 密码：`123456`

建议首次登录后立即修改密码。

## 鉴权模型（重要）

当前仅支持**用户 API Key**访问业务接口：

- 请求头：`Authorization: Bearer <user_api_key>`
- 不再支持管理面板配置“全局 API_KEY”直连调用

示例：

```bash
curl http://localhost:7860/search \
  -H "Authorization: Bearer your-user-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query":"latest linux do news","numResults":3}'
```

---

## MCP 工具（HTTP）

项目内置 MCP 服务，入口为：`/mcp`。  
可通过 MCP 客户端调用工具（search、contents、findSimilar、answer、research）。

**环境变量**

- `EXA_POOL_API_KEY`：用于访问本服务的用户 API Key（必填）

说明：

- MCP 仅需配置 `EXA_POOL_API_KEY`，无需额外参数。
- 未配置 `EXA_POOL_API_KEY` 时，MCP 请求将直接返回错误提示。

### 2. 配置（以 Claude Code 为例）

```bash
claude mcp add --transport http exa-pool http://localhost:7860/mcp \
  --env EXA_POOL_API_KEY=your-user-api-key
```

重启 Claude Code 以应用 MCP 的配置变更。

---

## Linux DO OAuth2 配置

在管理员面板 `系统设置 -> 基础 -> Linux DO OAuth 登录` 中填写：

- `linuxdo_oauth_enabled`
- `linuxdo_client_id`
- `linuxdo_client_secret`
- `linuxdo_authorize_url`（默认 `https://connect.linux.do/oauth2/authorize`）
- `linuxdo_token_url`（默认 `https://connect.linux.do/oauth2/token`）
- `linuxdo_userinfo_url`（默认 `https://connect.linux.do/api/user`）
- `linuxdo_redirect_uri`（可选，留空自动推导）
- `linuxdo_scope`（默认 `openid profile email`）

回调地址填写规则：

- 若手动填写 `linuxdo_redirect_uri`：使用该值
- 若留空：自动使用 `{base_url}/auth/linuxdo/callback`
- `base_url` 为空时，运行时按当前服务地址推导

登录页会在可用时显示“使用 Linux DO OAuth 登录 / 注册登录”按钮。

参考文档：<https://wiki.linux.do/Community/LinuxDoConnect>

---

## 注册策略开关

管理员可在“用户管理 -> 用户策略”控制：

- `registration_enabled`：总注册开关
- `password_registration_enabled`：账号密码注册开关
- `linuxdo_oauth_registration_enabled`：Linux DO OAuth 注册开关

说明：

- 关闭总开关后，所有注册方式都不可用
- 关闭密码注册后，`/auth/register` 会返回 `403`
- 关闭 OAuth 注册后，已有 OAuth 绑定用户仍可 OAuth 登录，新用户不可通过 OAuth 自动注册

---

## 数据库与备份恢复

### 存储后端

- 设置 `DATABASE_URL`：使用 PostgreSQL
- 未设置 `DATABASE_URL`：默认 SQLite（`data/data.db`）

### 管理面板一键备份/恢复（SQLite）

系统设置页支持：

- 一键导出数据库（下载 `.db` 文件）
- 一键导入数据库（上传并**覆盖**旧数据库）

对应接口：

- `GET /api/admin/database/export`
- `POST /api/admin/database/import`（`multipart/form-data`，字段名 `file`）

注意：导入覆盖不可逆，建议先导出备份。

---

## 主要 API

### 业务接口（需用户 API Key）

| 接口 | 方法 | 说明 |
|---|---|---|
| `/search` | POST | 搜索 |
| `/answer` | POST | 回答 |
| `/contents` | POST | 内容抓取 |
| `/findSimilar` | POST | 相似内容检索 |
| `/research/v1` | POST/GET | 创建研究任务 / 列表 |
| `/research/v1/{research_id}` | GET | 查询研究任务 |
| `/health` | GET | 健康检查 |

### 认证与用户

| 接口 | 方法 | 说明 |
|---|---|---|
| `/auth/options` | GET | 登录/注册能力开关（登录页使用） |
| `/auth/register` | POST | 账号密码注册并返回首个 API Key |
| `/auth/login` | POST | 用户/管理员登录 |
| `/auth/logout` | POST | 退出登录 |
| `/auth/me` | GET | 当前登录用户信息 |
| `/auth/change-password` | POST | 修改密码 |
| `/auth/redeem` | POST | 普通用户兑换升级码 |
| `/auth/apikeys` | GET | 列出当前用户 API Key |
| `/auth/apikeys/new` | POST | 新建 API Key |
| `/auth/apikeys/revoke` | POST | 吊销 API Key |
| `/auth/linuxdo/start` | GET | 跳转 Linux DO OAuth 授权 |
| `/auth/linuxdo/callback` | GET | Linux DO OAuth 回调 |

### 管理员（需管理员会话）

| 接口 | 方法 | 说明 |
|---|---|---|
| `/admin/users` | GET/POST | 用户列表 / 新建用户 |
| `/admin/users/{user_id}` | DELETE | 删除用户 |
| `/admin/users/{user_id}/enable` | PUT | 启用用户 |
| `/admin/users/{user_id}/disable` | PUT | 禁用用户 |
| `/admin/user-policy` | GET/PUT | 用户策略与限流 |
| `/admin/redeem-codes` | GET | 兑换码列表 |
| `/admin/redeem-codes/generate` | POST | 批量生成兑换码 |
| `/admin/redeem-codes/import` | POST | 批量导入兑换码 |
| `/admin/redeem-codes/export` | GET | 导出兑换码 |
| `/admin/redeem-codes/{code_id}` | DELETE | 删除兑换码 |
| `/api/admin/settings` | GET/PUT | 系统设置 |
| `/api/admin/database/export` | GET | 导出 SQLite |
| `/api/admin/database/import` | POST | 导入并覆盖 SQLite |

---

## 本地开发（非 Docker）

```bash
git clone https://github.com/Dreamy-rain/exafree.git
cd exafree

# 前端
cd frontend
npm install
npm run build
cd ..

# 后端
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```

---

## 常见问题

1. 登录页不显示 Linux DO OAuth 按钮  
   原因：`linuxdo_oauth_enabled` 未开启，或 `client_id/client_secret` 等未配置完整。

2. OAuth 回调失败  
   检查 Linux DO Connect 配置的回调地址是否与系统设置中的实际回调完全一致。

3. 导入数据库失败  
   目前浏览器导入仅支持 SQLite `.db` 文件，且会覆盖当前库。

---

## 相关文档

- 免责声明：[docs/DISCLAIMER.md](docs/DISCLAIMER.md)
- 支持文件类型：[docs/SUPPORTED_FILE_TYPES.md](docs/SUPPORTED_FILE_TYPES.md)
