# Docker 部署与打包清单

本项目默认按单进程、单张 NVIDIA GPU 部署。`output` 和 RVC 模型位于宿主机，重建镜像不会删除它们。

## 1. 服务器准备

- Docker Engine 和 Docker Compose v2
- NVIDIA 驱动及 NVIDIA Container Toolkit
- 可访问 Debian 软件源、PyPI、PyTorch wheel 源和 `ghcr.io` 的构建网络（容器运行时代理不自动覆盖镜像构建和拉取）
- 建议至少预留 20 GB 磁盘空间；CUDA/PyTorch 镜像较大

首次部署前确认 GPU 容器可用：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## 2. 必须打包的文件

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `pyproject.toml`
- `uv.lock`
- `app/`
- `.env.example`（模板，不含真实密钥）
- `DEPLOYMENT.md`

RVC 功能还需要单独携带这些大文件：

- `assets/rvc/weights/9971-700.pth`
- `assets/rvc/indices/9971-700_added_IVF2037_Flat_nprobe_1_9971-700_v2.index`
- `assets/rvc/base_model/hubert_base.pt`
- `assets/rvc/base_model/rmvpe.pt`
- `assets/rvc/base_model/rmvpe.onnx`

不要打包 `.env`、`.venv`、`.git`、缓存和现有 `output/`；只有迁移历史任务时才单独备份 `output/`。

## 3. 制作发布包

在项目根目录执行：

```bash
tar -czf ai-band-generate-module.tar.gz \
  Dockerfile docker-compose.yml .dockerignore DEPLOYMENT.md \
  pyproject.toml uv.lock app .env.example

tar -czf ai-band-rvc-assets.tar.gz assets/rvc
```

如果需要迁移历史生成结果，再额外执行：

```bash
tar -czf ai-band-output-backup.tar.gz output
```

## 4. 在线构建并启动

将发布包传到服务器后：

```bash
mkdir -p ai-band && cd ai-band
tar -xzf ../ai-band-generate-module.tar.gz
tar -xzf ../ai-band-rvc-assets.tar.gz
cp .env.example .env
chmod 600 .env
mkdir -p output/.bs-roformer-models
```

编辑 `.env`，至少设置真实的 `LLM_API_KEY`、音乐 Provider 配置。宿主机端口可用 `APP_PORT` 修改；宿主机用户不是 UID/GID 1000 时，设置 `APP_UID` 和 `APP_GID`，确保容器能写入 `output`。

缓存目录也必须由容器用户写入。若目录由 root 创建或从其他机器迁移，启动前按 `.env`
中的实际 UID/GID 修正权限，例如默认配置执行 `sudo chown -R 1000:1000 output`。
模型缓存目录 `output/.bs-roformer-models` 也必须由容器用户写入，供 BS-RoFormer 权重下载与校验。
当前 Compose 默认启用 SSH 隧道代理，发起联网业务前先完成下面的代理配置；服务器可以直接联网时，
按“停用代理”移除代理配置后再启动。

```bash
docker compose config --quiet
docker compose build
docker compose up -d
docker compose ps
curl http://127.0.0.1:${APP_PORT:-8010}/api/health
```

查看日志和升级：

```bash
docker compose logs -f --tail=200 api
docker compose build --pull
docker compose up -d
```

### SSH 反向隧道代理

为了不在服务器安装第三方代理软件，使用服务器已有的 OpenSSH，将**本机已有代理**的
7897 端口通过 SSH 反向端口转发提供给服务器上的容器。这里的“本机”是执行 SSH 命令的
个人电脑，`server_ip` 是运行 Docker 的服务器；这是出网代理，与应用入口的 HTTPS 反向代理无关。

请求路径为：`容器 → host.docker.internal:7897（服务器）→ SSH 隧道 → 本机 localhost:7897 → 目标服务`。
SSH 只转发 TCP，本机 7897 必须提供 HTTP 代理或兼容 HTTP 的 mixed 端口；纯 SOCKS 端口
不能直接使用当前的 `http://` 配置。本机代理只需监听回环地址，无需向本机局域网开放。

#### 1. 配置服务器 sshd

在**服务器**的 `/etc/ssh/sshd_config` 或其实际加载的配置片段中设置：

```text
GatewayPorts clientspecified
```

`GatewayPorts` 默认是 `no`，会将远程转发限制在服务器回环地址；`clientspecified` 允许 SSH
客户端指定监听地址，才能让 Docker bridge 中的容器连接宿主机的 7897。
`yes` 会强制通配地址监听，不建议替代这里的 `clientspecified`。
配置含义见 [OpenSSH sshd_config 手册](https://man.openbsd.org/sshd_config#GatewayPorts)。

修改前备份原配置，检查 `Include` 和 `Match` 中是否已有设置，修改实际生效项，不要重复追加。
如只供一个账号使用，建议将 `GatewayPorts clientspecified` 放在该账号的 `Match User user`
配置范围内，并用 `PermitListen 实际bridge地址:7897` 限制可监听的地址和端口；
地址须按下一步核实，例如当前服务器为 `PermitListen 172.17.0.1:7897`。
该账号还必须允许远程 TCP 转发（`AllowTcpForwarding yes` 或 `remote`），且不能被
`DisableForwarding`、`PermitListen` 或 `authorized_keys` 的转发限制阻止；不要为此取消其他账号的限制。

验证语法并重载（Debian/Ubuntu 通常使用 `ssh`，其他发行版可能使用 `sshd` 服务名）：

```bash
sudo /usr/sbin/sshd -t && sudo systemctl reload ssh
sudo /usr/sbin/sshd -T | grep -E '^(gatewayports|allowtcpforwarding|disableforwarding|permitlisten) '
```

使用 `Match` 时，需向 `sshd -T` 加上 `-C user=user,host=client_name,addr=client_ip`，
替换为实际 SSH 用户、本机主机名和服务器看到的客户端 IP，检查该连接的有效配置。
保留当前管理会话，用新会话验证 SSH 仍可登录；重载后需重新建立隧道才能应用新配置。

#### 2. 确认地址并先限制访问

完成镜像构建后，可以先启动容器核对网络，此时不要发起需要代理的业务请求。
在**服务器项目目录**执行：

```bash
docker compose up -d api
docker compose exec -T api python -c 'import socket; print(socket.gethostbyname("host.docker.internal"))'
ip -4 addr show
docker inspect "$(docker compose ps -q api)" --format '{{json .NetworkSettings.Networks}}'
```

隧道应绑定容器解析到的宿主机 bridge 地址，防火墙应匹配应用所属的网桥，两者可能不同。
当前服务器解析到 `172.17.0.1`（`docker0`），应用属于 `br-e71e5f2bd0ad`（网关 `172.20.0.1`）。
可以根据容器网络的 `NetworkID` 与实际 `br-` 网卡核对；有自定义 bridge 名称时以实际配置为准。
以下地址和网桥仅是该服务器的已验证示例，其他机器必须替换为核实后的值。

**建立隧道前先限制 TCP 7897 的访问来源。** 本次排查时服务器使用 iptables-nft，
IPv4/IPv6 INPUT 默认放行、UFW 未启用，尚无 7897 来源限制，Tailscale 规则也未限制该端口。
使用 iptables 在 INPUT 首位添加规则，仅允许应用网桥访问；规则已存在时只检查其顺序，
不重复添加：

```bash
sudo iptables -I INPUT 1 ! -i br-e71e5f2bd0ad -p tcp --dport 7897 -j DROP
sudo iptables -S INPUT
```

确认 DROP 规则位于 `-j ts-input` 等放行规则之前；不要直接修改 Docker 管理的 nft 表。
规则也阻止宿主机回环和其他网桥访问 7897，同一应用网桥上的其他容器仍能访问。
这是 IPv4 规则，后续不应新增 IPv6 的 7897 监听而不配置相应限制。
手动规则通常在重启后丢失；服务器重启或防火墙被重置后，需在重建隧道前检查/恢复。

#### 3. 在本机建立隧道

先结束原有监听 `0.0.0.0` 或错误地址的隧道，启动本机代理并确认 `localhost:7897` 可用，
然后在**本机**终端运行（当前服务器已核实监听地址为 `172.17.0.1`）：

```bash
ssh -v -N -R 172.17.0.1:7897:localhost:7897 user@server_ip
```

- `-v` 输出连接和转发调试信息；共享日志前隐藏用户名、主机地址和密钥路径。
- `-N` 不执行远程命令，终端保持运行以维持隧道。
- `-R 172.17.0.1:7897:localhost:7897` 在**服务器已核实的 bridge 地址**监听 7897，将连接转发到
  **本机**的 `localhost:7897`；这里的 `localhost` 不指服务器或容器。
- `user@server_ip` 替换成真实账号和地址；SSH 端口非 22 时加 `-p 实际端口`。

原命令 `ssh -v -N -R 0.0.0.0:7897:localhost:7897 user@server_ip` 会监听服务器所有 IPv4
网卡，存在外部借用代理的风险，不作为默认示例。仅在确有需要且已验证严格的来源限制后使用。

建议使用下面的命令，让监听失败时退出，并及时检测 SSH 连接失效：

```bash
ssh -v -N -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 172.17.0.1:7897:localhost:7897 user@server_ip
```

这些选项不会自动重连，也不保证本机代理或目标网站可用。关闭终端、电脑休眠、网络中断
都会影响代理；连接失败后需手动重新执行。选项说明见
[OpenSSH ssh 手册](https://man.openbsd.org/ssh#R)。

#### 4. 容器配置与验证

当前 `docker-compose.yml` 的 `api` 服务包含：

```yaml
environment:
  HTTP_PROXY: http://host.docker.internal:7897
  HTTPS_PROXY: http://host.docker.internal:7897
  NO_PROXY: localhost,127.0.0.1,::1,hf-mirror.com
extra_hosts:
  - "host.docker.internal:host-gateway"
```

`host.docker.internal` 通过 `host-gateway` 指向 Docker 宿主机内部地址，而不是个人电脑，
见 [Docker host-gateway 文档](https://docs.docker.com/reference/cli/docker/container/run/#add-entries-to-container-hosts-file---add-host)。
容器中的 `localhost` 指容器自身，不能将代理地址改为 `http://localhost:7897`。
`HTTPS_PROXY` 仍使用 `http://`，表示通过 HTTP 代理的 CONNECT 隧道访问 HTTPS 目标。
`NO_PROXY` 中的地址绕过代理；当前 `hf-mirror.com` 直接访问服务器网络。内网服务如需直连，
将其实际域名/IP 加入列表；镜像站也需要代理时，从列表中移除 `hf-mirror.com`。
这些值写在 Compose 的 `environment` 中，优先于 `env_file: .env`，只改 `.env` 不会覆盖它们，
见 [Compose 环境变量优先级](https://docs.docker.com/compose/how-tos/environment-variables/envvars-precedence/)。

应用中遵循代理环境变量的客户端会使用隧道；ElevenLabs 默认
`ELEVENLABS_BYPASS_GLOBAL_PROXY=true`，其生成请求直连，需要代理时在 `.env` 改为 `false`。
MiniMax 使用独立直连客户端，不随这组环境变量切换。模型下载是否走代理取决于实际下载客户端。

在**服务器**检查监听和访问规则，并从容器验证代理链路：

```bash
sudo ss -lntp 'sport = :7897'
sudo iptables -S INPUT
docker compose exec -T api python -c 'import httpx; r = httpx.head("https://example.com/", proxy="http://host.docker.internal:7897", trust_env=False, timeout=15); r.raise_for_status(); print(r.status_code)'
```

当前服务器已验证监听为 `172.17.0.1:7897`，容器代理请求返回 `200`，DROP 规则位于
Tailscale 放行规则之前。`ss` 的监听范围看 `Local Address`，`Peer Address` 中的
`0.0.0.0:*` 不代表监听所有网卡；按上述限制配置后，宿主机回环代理测试也会被阻止。
若 SSH 报 `remote port forwarding failed`，检查端口占用、`PermitListen` 和转发权限；
容器连接被拒绝/超时时，检查实际监听是否匹配 `host.docker.internal` 的解析地址、
网桥名称和防火墙规则。只监听 `172.20.0.1` 而容器解析到 `172.17.0.1` 会造成地址不匹配。
以上显式代理测试成功只证明该目标的代理链路可用，应用 health 成功也不代表模型下载或
Provider 已联通；还需按实际业务验证。

#### 风险与访问限制

- **公网暴露：** `0.0.0.0:7897` 会监听服务器所有 IPv4 网卡。如果安全组或防火墙允许公网
  访问，任何能连接该端口的人都可能借用本机代理出网；SSH 登录认证不会替转发端口上的
  代理请求做认证。可能造成流量盗用、出口 IP 滥用，以及通过本机代理访问其可达的内网服务。
  不要在云安全组开放公网 7897；宿主机防火墙仅允许所需 Docker bridge 网段访问该端口，
  阻止公网及其他不需要的来源，并核对实际监听地址及相关 IPv6 规则。
- **配置影响范围：** 全局 `GatewayPorts clientspecified` 会允许其他具有远程转发权限的
  SSH 用户选择非回环地址；优先按账号限制。默认绑定已核实的宿主机 bridge 地址并同步
  调整 `PermitListen`。绑定 bridge 地址本身不限制请求来源，还需上述 INPUT 访问限制。
- **可用性与数据路径：** 代理依赖个人电脑、代理进程和 SSH 会话；隧道断开时，仍配置代理
  的客户端通常请求失败，不会自动回退直连。流量消耗本机带宽，受本机代理规则和出口影响。
  SSH 保护本机与服务器之间的隧道，容器到宿主机段及代理出口段不因此获得额外加密；
  使用 HTTPS 并保留证书验证，不要用关闭 TLS 校验解决连通问题。
- **模型下载：** BS-RoFormer 固定从运行库登记的 HTTPS 地址下载权重，并校验 SHA-256。
  下载使用全局 HTTP/HTTPS 代理，缓存位于 `output/.bs-roformer-models`；该运行库不读取
  `HF_ENDPOINT` 或 `HF_HOME`，不能通过这些变量更改其下载位置。离线部署应先用 README
  中的预下载命令准备模型，再把整个缓存目录随 `output` 迁移。

#### 升级时保留网桥

日常升级保持项目目录/Compose 项目名和网络配置不变，使用：

```bash
docker compose build
docker compose up -d --force-recreate api
```

这些操作复用现有网络，容器 IP 变化不会影响按网桥匹配的防火墙规则。
`docker compose down` 默认删除项目网络，再次 `up` 时网桥名称可能变化，见
[Docker down 文档](https://docs.docker.com/reference/cli/docker/compose/down/)。
网桥生命周期取决于 Docker 网络，即使服务器不关机，删除/重建网络也会使原名称失效。
网络重建后，核对解析地址和应用新网桥，先添加新的来源限制、移除旧规则，再复测代理；
解析地址变化时同步更新隧道及 `PermitListen`。旧 DROP 规则会阻止新网桥访问 7897，
表现为代理请求失败。正常升级保留现有网络即可，无需额外配置固定网桥。

#### 停用代理与恢复配置

服务器具备直连能力、需要长期独立运行或不再使用该代理时：

1. 修改 `docker-compose.yml`，删除或注释 `api.environment` 中的 `HTTP_PROXY`、
   `HTTPS_PROXY`、`NO_PROXY`，删除仅供代理使用的 `extra_hosts` 条目；块为空时一并删除块。
   同时清理 `.env` 中可能存在的 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`、`NO_PROXY`
   及对应小写变量，避免从 `env_file` 再次注入。如果仍需其他宿主机服务的映射，保留对应条目。
2. 校验并重建容器，使新环境生效：

   ```bash
   docker compose config --quiet
   docker compose up -d --force-recreate api
   docker compose exec api python -c 'import os; print(sorted(k for k in os.environ if k.lower().endswith("_proxy")))'
   ```

   彻底停用时最后一条应输出 `[]`。无需重新构建镜像，单独 `docker compose restart` 不会
   更新容器环境；重新验证实际 API 请求和模型下载。若只是临时关闭 SSH 且不改 Compose，
   代理请求会失败，因此不要把“关闭隧道”当作完整停用。
3. 在本机运行 SSH 的终端按 `Ctrl+C` 结束该隧道；若以后台方式启动，找到并结束对应进程，
   不要批量杀死其他 SSH 管理连接。在服务器用 `sudo ss -lntp 'sport = :7897'` 确认该监听已移除。
4. 将服务器的有效 `GatewayPorts` 改回原值；原来未配置时恢复为默认 `no`，或移除本次为
   专用账号增加的 `Match` 配置。按前面的 `sshd -t` 检查后重载，并再次验证有效配置。
   若还需其他 SSH 转发，保留其配置。**修改并重载 sshd 不会撤销已经建立的转发，必须结束原隧道。**
   同时撤回本次为 7897 添加的防火墙/安全组放行规则。
5. 确认隧道监听已移除后，删除本次添加的来源限制规则。当前服务器的命令为：

   ```bash
   sudo iptables -D INPUT ! -i br-e71e5f2bd0ad -p tcp --dport 7897 -j DROP
   ```

   网桥变更后应使用实际添加的规则参数；不要清空整个 INPUT 链或 Docker/Tailscale 规则。

停用代理后，模型权重下载改走直接 HTTPS；已缓存且校验通过的权重无需重复下载。
不要为停用代理删除 `output` 或模型缓存，否则会丢失结果或触发重新下载。

### 构建源与缓存说明

当前 Dockerfile 在 builder 阶段将匹配到的 Debian `http://deb.debian.org` 地址替换为清华镜像，
设置 `UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`、`UV_HTTP_TIMEOUT=3000`（秒），
并通过 BuildKit 缓存挂载复用 `/root/.cache/uv`。构建缓存与容器模型缓存是两回事：
BS-RoFormer 使用 `./output/.bs-roformer-models`（容器内 `/app/output/.bs-roformer-models`），
随已有 `output` 卷持久化；无需额外缓存挂载。

**运行时 Compose 代理不会自动用于 `docker compose build`，也不会为 Docker daemon 拉取镜像
提供代理。** 最终镜像的 `ffmpeg` 安装仍使用该阶段原有 Debian 源；项目的 `pyproject.toml`
显式配置了 PyPI/PyTorch 索引，`uv.lock` 保留下载地址，`uv sync --frozen` 不重新生成锁文件，
不能仅凭 `UV_INDEX_URL` 就认为所有依赖都已改走镜像。构建网络仍需覆盖实际访问的源，
依据构建日志确认；无法满足时按第 5 节在可联网机器构建并导入镜像。
锁文件行为见 [uv sync --frozen 文档](https://docs.astral.sh/uv/reference/cli/#uv-sync--frozen)。

不用构建镜像源时，在 Dockerfile 中移除 builder 的 `sed` 替换和 `UV_INDEX_URL` 设置，
按需要恢复 `UV_HTTP_TIMEOUT`，然后重新构建。BuildKit 缓存挂载可以保留，无需为停用代理清空。

### 安全启用歌曲保留策略

`output` 树内（含 `jobs`、`.expired`、`.trash`）不支持符号链接：保留清理拒绝沿链接操作，
跨出 `output` 的音频路径返回 404。多卷部署请直接挂载到 `OUTPUT_DIR` 或真实子目录。
启动时后台检查 `jobs`/`.expired`，不支持的布局会记录 WARNING `unsupported_layout`，
health 中 `unsupportedLayout=true` 并列出相对目录名 `unsupportedLayoutDirectories`。
启用前确认该标志为 `false`；`null` 表示检查尚未完成或无法确认。修复后等待下一次清单刷新。

**正式启用后，第一次启动会永久删除所有超过 `SONG_RETENTION_DAYS`（默认 3 天）的既有历史、
歌曲与回收区内容；策略追溯旧任务，无法恢复。** `output/` 可能是唯一副本，先备份：

```bash
tar -czf ai-band-output-before-retention.tar.gz output
```

先在 `.env` 设置 `SONG_RETENTION_ENABLED=true`，保持默认的 `SONG_RETENTION_DRY_RUN=true`，
重建/重启服务。dry-run 不删除、不移动文件，也不限制历史访问。检查
`output/logs/retention.log` 中的 `wouldDeleteJobIds` 和跳过原因，以及 `/api/health` 的
`retention.orphanJobIds`、`pendingDeletionJobIds`、`expiredActiveJobIds`。日志不可写时检查应用日志。
确认备份与删除范围后，再设置 `SONG_RETENTION_DRY_RUN=false` 并重启，执行正式清理。
正式启用前，必须由 PO 在 PR 或上线工单明确确认追溯删除，并由前端负责人确认新增字段、
SSE `done/status=expired` 无结果载荷及音频 404 的兼容性。策略关闭时不返回新增任务字段，
默认部署保持原有任务响应结构；开启 dry-run 才会暴露新增字段。
关闭开关会停止删除，但 `.expired` 残留仍列在 health/启动日志中，重新正式启用后会重试。

health 的清单来自后台快照（启动、每小时及清理后刷新），先确认 `inventoryUpdatedAt` 非空、
`inventoryStale=false` 且 `inventoryError=null`，并核对快照时间。`inventoryAttemptedAt` 只表示
尝试时间；失败不会推进成功时间，失败类别保留旧数据（或首次扫描时计数为 `null`），
对应的 `*Stale=true`。不能把旧计数为 0 当作本次已确认无残留。
每类最多展示 20 个 id；`Count` 为总数，`Truncated=true` 表示汇总并不完整。审查全部删除范围
时查看逐任务 `job_would_delete` 日志。`unmanagedJobs` 给出元数据异常原因，修复后需重启重新加载。

另核对 `lastCleanupAt/Reason/Result/Error`；它们反映最近一轮清理的结束时间、触发原因、
结果和脱敏错误，与清单扫描状态独立。`cleanupRunning=false` 和 `remainingJobs=0` 不代表清理成功。
dry-run 的 `lastCleanupResult=success` 只表示审计完成，未执行删除；同时核对 `dryRun` 和
`deletedJobIds/wouldDeleteJobIds`。只有正式模式才能实际删除文件。
日志的 `cleanup_failed` 带 `stage`（`pendingDeletionScan` 或 `run`），`cleanup_finished` 带
`result/error/pendingDeletionScanFailed`。暂存区扫描失败仍尝试处理已知过期任务；若暂存区
损坏或不可写，逐任务失败会列入 `failedJobIds`，修复后再重试，不能视为空批次。
关停只等待当前目录删除完成，批次剩余任务保留，结果为 `stopped`；意外循环异常记录
`retention_monitor_failed` 并在下一次刷新间隔重试。

## 5. 离线服务器打包镜像

在可联网、CPU 架构与目标服务器一致的机器上构建：

```bash
docker compose build
docker image save ai-band-generate-module:latest | gzip > ai-band-image.tar.gz
```

将代码发布包、RVC 资源包、镜像包一起传到服务器，然后执行：

```bash
gzip -dc ai-band-image.tar.gz | docker image load
docker compose up -d --no-build
```

`.env` 必须只在服务器上创建和保管。公网部署时再在服务前配置现有的 HTTPS 反向代理；本清单不额外绑定某一种代理。

## 6. ElevenLabs 升级注意事项

- **硬性发布门槛：本版本的按首发布与部分成功契约不兼容旧前端，必须与适配前端同版本
  发布；禁止只更新后端。无法保证前后端配套发布时，不得上线本版本。**
  发版验收必须确认：前端不会把运行中的 `result` 归一化成成功或停止 SSE；第二首未完成时
  不会因当前只有一个可用输出而夹取并写回歌曲选择；部分成功会按 `songNumber` 对应状态。
  联调须覆盖第一首可播放且第二首仍生成、第二首失败后第一首保留、第一首失败而第二首成功。
  SSE 的普通进度帧省略波形，前端需保留已取得的波形或通过单任务详情接口补取。
- 旧版 `.env` 中的 `ELEVENLABS_MUSIC_OUTPUT_FORMAT=auto` 会自动兼容为
  `pcm_44100`；其他非 `pcm_*` 值会在服务启动时报错。
- ElevenLabs 的双曲生成是串行且按两次调用计费：第一首完成后立即发布，第二首失败时
  任务保留第一首并以部分成功收尾，错误记录在 `songStates` 和 `warning` 中。
  明确的鉴权、套餐、配额或限流错误会终止后续调用并将任务标为失败，已发布的歌曲保留。
  前端应允许 `status=running` 的任务携带可播放的 `result`，并继续订阅第二首的进度。
  请优先使用 `/api/jobs` 异步任务端点；如果使用同步生成端点，反向代理超时必须
  高于两次 ElevenLabs 生成的总时间。
- 44.1 kHz、16-bit 立体声 WAV 约为 10.6 MB/分钟。在 6 分钟、两首歌的上限下，
  单任务约需 127 MB 磁盘和下载流量，部署时需相应规划 `output` 容量与出网带宽。
  前端的 `fullTrack` 已是 `.wav`，播放时返回 `audio/wav`。
