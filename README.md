# AI Band Generate Module

AI 音乐生成与 RVC 人声替换的纯 FastAPI 后端。项目使用 `uv` 固定 Python 和依赖版本，
不包含原 Vue/Vite 前端。

## 功能

- OpenAI-compatible LLM 音乐 Prompt 结构化扩写与质量校验
- ElevenLabs Music API（直接 Prompt）
- Suno sidecar 和 Generic Provider 兼容适配器
- 内置 BS-RoFormer `vocal / drums / bass / guitar / piano / other` 六轨分离实现（按需执行）
- RVC 人声替换、结果下载、软删除与恢复
- `/api/health`、`/api/generate` 和 `/output/*` 音频访问接口
- 本地内联任务执行，以及队列、缓存、事件发布的可替换接口

默认端口为 `8010`，Prompt 上限为 2000 个 Unicode 字符。

## 安装

开发、运行和测试均以 Linux 为支持平台（`uv` 已限定 `sys_platform == 'linux'`）。
Windows 请在 WSL2 的 Linux 环境或 Linux 容器中执行以下命令和 `uv run pytest`；
不支持在原生 Windows Python 中直接安装或收集测试。参考音频的跨进程去重依赖 POSIX
`fcntl` 文件锁，不提供仅进程内锁的降级，以保持并发上传的去重保证。

项目固定使用 Python 3.10：

```bash
uv python install 3.10
uv sync --frozen
cp .env.example .env
```

当前依赖固定为 Linux `torch/torchaudio 2.7.1+cu128`，用于支持 RTX 50 系列
Blackwell `sm_120`；BS-RoFormer 和 RVC 共用这一套 CUDA 环境，不再安装 CPU Torch。
运行拆轨或默认的 RVC 配置需要可用的 NVIDIA 驱动和 CUDA GPU。

编辑 `.env`，至少配置 LLM Key。`MUSIC_PROVIDER` 只是请求未传 `provider` 时的默认值；
MiniMax 和 ElevenLabs 的配置可以同时保留。真实 ElevenLabs 模式还需要配置：

```env
MUSIC_API_MODE=real
MUSIC_PROVIDER=elevenlabs_music
LLM_API_KEY=...
LLM_MAX_TOKENS=2048
LLM_TIMEOUT_SECONDS=120
LLM_DISABLE_THINKING=true
ELEVENLABS_API_KEY=...
```

ElevenLabs 直接接收完整的结构化歌词和扩充风格标签，由模型自行安排段落与演唱节奏。
未传 `durationSeconds`，且 `durationMinutes` 为 `auto`、`null` 或未传时，
不发送 `music_length_ms` 或固定目标时长，由模型根据 Prompt 决定时长。
`durationSeconds` 优先于 `durationMinutes`；显式指定时长时发送 `music_length_ms`。
自动时长返回的音频须至少为 3 秒，低于该下限会报错，不保存为成功歌曲。
后端不会裁剪歌词；组合后的 Prompt 超过接口的 4100 字符上限时会在
调用前报错。歌词相对目标时长过长时可能无法完整唱完，调用方需要缩短歌词或增加目标时长。
请求两首时后端会使用相同歌词和风格独立调用两次 ElevenLabs。音频固定请求
`pcm_44100`，再在本地封装为 16-bit 立体声 WAV，不保存 MP3。
当前产品只生成中文歌曲，不再根据请求内容检测演唱语言：ElevenLabs 默认追加中文清晰人声
要求，MiniMax 同样直接接收中文歌词与中文人声风格标签。启用
`ELEVENLABS_FORCE_INSTRUMENTAL=true` 时会自动关闭 ElevenLabs 中文人声要求，避免与纯音乐
模式冲突。

### 音频参考

`POST /api/references`（multipart 字段 `file`）上传一段参考音频，用来引导新歌的整体音色、
制作风格、配器、速度与情绪（不会复制或混音原曲）。参考音频可用于 ElevenLabs `music_v2` /
`music_v2_5` 或 Mureka（`mureka-o2` 不支持）；MiniMax 不支持。ElevenLabs 参考不能与
`ELEVENLABS_FORCE_INSTRUMENTAL=true` 同时使用。

- 上传阶段只在本地完成：后端将文件转为最长 30 秒的 MP3，生成随机本地参考 id 并保存至
  `output/references/`；**不会调用 ElevenLabs 或 Mureka 接口**。
- `GET /api/references` 列出本地参考（含 `output/references/*.mp3` 试听地址）；
  `DELETE /api/references/<id>` 仅删除本地记录与试听文件。
- 生成请求（`/api/generate`、`/api/jobs`）带 `referenceId` 时，才按当前 provider 上传本地文件：
  ElevenLabs 先调用 `POST /v1/music/upload`，再调用免费的 `POST /v1/music/plan`，在首个
  chunk 加入 `conditioning_ref` 和 `condition_strength`
  （`ELEVENLABS_REFERENCE_CONDITION_STRENGTH`，默认 `medium`），最后以 `composition_plan`
  调用 stream 接口；Mureka 使用 `POST /v1/files/upload` 的 `purpose=reference`，短于 30 秒时会先在本地循环补足。
  provider 返回的远端 id 持久化到本地索引，重启后继续复用；文件锁保证共享同一输出目录的
  进程不会并发重复上传。同一服务商下相同内容的本地参考也会复用，Mureka 和 ElevenLabs 仍相互隔离。
  ElevenLabs 不足 10 秒的参考会先循环补至 30 秒；本地仍接受至少 3 秒的音频。
  生成时会先报告参考音频准备阶段，首次上传仍可能按服务商规则计费。
  缓存与服务商地址、API 凭据绑定，变更后会重新上传；凭据只保存不可逆摘要，不写入索引原文。
  若进程在远端已受理、但本地索引尚未写回时崩溃，仍可能再次上传；目前上游未使用幂等键。
- 单文件上限 `AUDIO_REFERENCE_MAX_UPLOAD_MB`（默认 30 MB），参考至少 3 秒。
  部署时还需对齐网关/反向代理的请求体限制，并为 multipart 边界和字段开销预留余量；
  例如后端为 30 MB 时，nginx 可设置 `client_max_body_size 32m;`，其它代理层也需同步检查。
  若小于后端上限的文件仍返回 413，应检查网关限制，不能只调整应用配置。

自部署 MiniMax Music 3 使用兼容的 `/v1/audio/jobs` 接口：

```env
MUSIC_API_MODE=real
MUSIC_PROVIDER=minimax_music
MINIMAX_BASE_URL=http://127.0.0.1:8111
MINIMAX_MODEL=MiniMaxAI/MiniMax-Music3
MINIMAX_SEED=42
MINIMAX_NUM_INFERENCE_STEPS=30
MINIMAX_TIMEOUT_SECONDS=7200
LLM_API_KEY=...
```

后端将 LLM 生成的歌词作为 `input`，结构化音乐描述作为 `instructions`，并将返回的
44.1 kHz WAV 直接保存到 `output`。MiniMax 始终使用自动时长，请求不包含
`audio_duration`，由模型选择自然完整的时长；即使客户端传入固定时长也会被忽略。
除 MiniMax、ElevenLabs 和 Mureka 外，其他 Provider 在未指定时长时使用 `DEFAULT_DURATION_MINUTES`。
如果两个服务分别运行在 Docker 容器中，请将 `MINIMAX_BASE_URL` 改为可达的容器服务名
或宿主机地址。

### Mureka

```env
MUSIC_API_MODE=real
MUSIC_PROVIDER=mureka_music
MUREKA_API_KEY=...
MUREKA_MODEL=mureka-9.5
```

后端调用 `POST /v1/song/generate`（`lyrics` 为 LLM 歌词，`prompt` 为结构化风格描述，
超过 1024 字符时截断；`n=1`，请求两首时独立调用两次），再轮询
`GET /v1/song/query/{task_id}`，优先下载 `wav_url`，没有时下载 MP3 `url`。
Mureka 没有时长参数，与 MiniMax 一样忽略固定时长，歌曲最长 5 分 30 秒。
价格见 <https://platform.mureka.ai/pricing>（`mureka-9.5` 每首 $0.15，`mureka-9` / `mureka-8` /
`mureka-o2` 每首 $0.045）。

音频参考流程见上文“音频参考”；Mureka 的远端 `reference_id` 在首次生成时创建并仅在进程内缓存。

音乐 Prompt 扩写建议使用非推理模型。推理模型可能先输出很长的思考过程，增加
延迟并触发读取超时。遇到 LLM `ReadTimeout` 时，应先确认 `LLM_MODEL`，再根据
服务延迟调整 `LLM_TIMEOUT_SECONDS`；后端不会自动重试网络超时，以免产生重复调用。
合格扩写结果必须包含 6–8 个 `[Category: value]` 英文制作标签，整体覆盖风格与年代、
速度与拍号、情绪、配器、中文人声、编曲结构、制作与混音以及排除项；相邻类别可合并。
若模型以 JSON 数组返回 `styleTags`，其中不带方括号的 `Category: value` 项会在拼接前
自动补上方括号，再按同一套规则校验。
未指定的要素由 LLM 做协调一致的专业补充，短标签翻译不会再被当作扩写成功。

对于 Qwen3/Qwen3.5，默认通过 `chat_template_kwargs.enable_thinking=false` 关闭推理，
避免思考过程耗尽输出预算。`LLM_MAX_TOKENS` 直接控制“歌词 + 风格标签”合并请求的输出
预算，默认 2048、最大 4096，不建议低于 2048。自动生成的歌词正文最多 1000 个字符，
并要求模型将每个原始风格标签值控制在 200 个字符以内，避免后端生成超出音乐 Provider
的 Prompt 上限。

歌词与风格在同一次 JSON 请求中生成；响应格式或内容校验失败时，后端会携带失败原因
重试一次。
使用 `[歌词与创作内容]` 和 `[风格要求]` 区分输入：歌词为空时按用户风格创作歌词；
风格为空时根据歌词的主题、情绪、意象和语言节奏推断风格；两者都有时严格遵守用户输入。
已有歌词只添加段落标签，不改写正文；歌词和风格都为空时，生成接口返回 400，不创建任务。

Mock 模式需要把测试母带放到 `output/mock_full.mp3`，或者修改
`MOCK_FULL_SONG_PATH`。

## 启动

```bash
uv run ai-band-api
```

也可以直接启动 Uvicorn：

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8010
```

接口文档：<http://127.0.0.1:8010/docs>

健康检查：

```bash
curl http://127.0.0.1:8010/api/health
```

服务器 Docker 部署、离线镜像和发布包清单见 [`DEPLOYMENT.md`](DEPLOYMENT.md)，
SSH 出网代理与升级注意事项见下方“Docker 部署与 SSH 出网代理”。

RVC 默认自动查找 `assets/rvc` 中的模型、索引和 HuBERT/RMVPE 基础模型；也可通过
`RVC_MODEL_PATH`、`RVC_INDEX_PATH` 和 `RVC_BASE_MODEL_DIR` 显式指定。RVC 推理在
第一次转换时懒加载到 `cuda:0`；任务替换固定使用 `rms_mix_rate=0`，使替换人声沿用原人声的音量包络：

已完成拆轨的生成任务可直接启动后台人声替换，无需重新上传 vocal 文件：

```bash
curl -X POST 'http://127.0.0.1:8010/api/jobs/<jobId>/replace?song=0'
```

首次启动返回 `202`；重复启动同一任务返回当前状态，已有可读结果时返回 `200`；上一次替换恰好
超时或取消、推理线程仍在安全收尾时返回 `409`。`GET /api/jobs/<jobId>` 和任务 SSE 会返回
`replaceStatus`、`replaceSong`、`replaceError`。顶层 `status/stage/progress/message`
沿用既有兼容契约：操作 `pending/running` 期间覆盖为当前操作的四字段；操作进入终态后，
顶层回到主音乐生成任务。历史列表与落盘记录始终表示主生成任务，操作失败不会污染歌曲生成状态。
POST、GET 与 SSE 新增独立操作字段：`operation`（`split` / `replace` / `mix`）、
`operationStatus`、`operationSong`、`operationStage`、`operationProgress`、`operationMessage`。
三种操作的进行中和终态均使用这套字段；`operationSong` 与对应的 `splitSong/replaceSong/mixSong`
一致，都是从 0 起的序号。既有操作 Status/Song/Error 字段保留。
操作字段选择当前操作或本进程最近一次操作，运行态不落盘；重启后复用既有 `replaceStatus/mixStatus`
推导补齐成功状态、`completed` 和 100。同时存在替换与合轨产物时优先报告下游合轨；
同类操作只有唯一一首有可用产物时才恢复歌曲序号；多首都有时为 `null`，应查看各首结果。失败/取消记录不跨重启保留，
没有可推导操作时新字段为 `null`；分轨缓存 POST 仍会补齐成功状态和歌曲序号。
替换依次经过 `operationStage=preparing_vocal`（准备输入音频，0）、
`replacing_vocal`（RVC 整体转换，25）、
`creating_replacement`（替换音轨与波形，50）、`exporting_replacement`（进入试听导出时 75、导出完成后 99）。
RVC 不暴露独立的特征分析回调或内部推理进度，只报告阶段里程碑；
结果保存可用后才报告 `operationStage=completed`、`replaceStatus=succeeded`、
`operationStatus=succeeded`、`operationProgress=100`。完成后歌曲结果增加 `replacedVocal` 音频 URL。转换超过
`RVC_CONVERSION_TIMEOUT_SECONDS`（默认 1800 秒）会立即标记失败；由于 RVC 推理线程不能安全
强停，锁和并发额度会保留到线程实际退出，期间产物不会发布。这种"已经终态但线程还没退出"的
替换数量可以从 `GET /api/health` 的 `replacement.workersHoldingCapacityAfterTerminal` 读到，
日志里也会在进入收尾和线程退出时各记一条 warning。`PATCH /api/jobs/<jobId>` 可取消：
接口会立即把 `operationStatus/operationStage` 和 `replaceStatus` 标记为 `cancelled`，
`operationProgress` 置空；主生成任务的顶层字段及历史保持成功。当前推理安全退出后会丢弃产物、释放并发额度。

替换结果与 RVC 模型绑定：`RVC_MODEL_PATH`/`RVC_INDEX_PATH` 内容或 `RVC_MODEL_VERSION` 变化后，
已缓存的结果会被判定为失效并自动重新推理。失效到新产物落位之间旧文件仍保留在磁盘上（不再
被 `replacedVocal` 引用），因为新旧产物同名、成功时会被原子覆盖；这样即使重跑失败、超时或
被取消，上一版可用的替换人声也不会被一并删除。删除原 vocal 分轨保留已有替换人声。

替换结果由 `/output/jobs/<jobId>/song_<n>/<原音轨名>_rvc_vocal.wav` 提供，接口返回的
`replacedVocal` 就是该 URL，前端直接播放或下载。`DELETE /api/voice/result` 软删除替换产物，
`PUT /api/voice/result` 恢复；两者均传 `job_id`、`filename`、`audio_revision` 和可选的
`song`，删除后可恢复，重新替换则会覆盖同一路径。这个入口只管理派生音频：母带 `fullTrack`
与提交后的保留音轨（`stems` 里任一文件）返回 409 —— 保留音轨的软删除与撤回走 `/stems`，
那里才记撤回记录、才检查版本。

## Docker 部署与 SSH 出网代理

为了不在服务器安装第三方代理软件，使用 SSH 反向隧道复用个人电脑上已有的 HTTP/mixed
代理端口 `7897`。Compose 已将 `HTTP_PROXY`、`HTTPS_PROXY` 指向
`http://host.docker.internal:7897`，并通过 `host-gateway` 将该域名映射到 Docker 宿主机。
服务器 sshd 需要 `GatewayPorts clientspecified` 并允许该账号的远程 TCP 转发。
完整配置、风险与停用步骤见 [`DEPLOYMENT.md`](DEPLOYMENT.md#ssh-反向隧道代理)。

**必须先核实监听地址和应用网桥，并限制 7897 的访问来源，再建立隧道。** SSH 登录认证
不会保护代理端口上的请求，不使用 `0.0.0.0:7897` 作为默认监听地址。
在服务器项目目录确认：

```bash
docker compose exec -T api python -c 'import socket; print(socket.gethostbyname("host.docker.internal"))'
ip -4 addr show
docker inspect "$(docker compose ps -q api)" --format '{{json .NetworkSettings.Networks}}'
```

当前服务器已核实：`host.docker.internal` 解析为 `172.17.0.1`（`docker0`），应用位于
`br-e71e5f2bd0ad`（网关 `172.20.0.1`）。**代理监听地址与允许访问的应用网桥是两个不同值**：
隧道绑定前者，防火墙匹配后者。其他机器或网络重建后需重新核实，不能直接照抄。

在服务器添加下面的 IPv4 规则，仅允许当前应用网桥访问 TCP 7897；
规则已存在时只检查其顺序，不重复添加：

```bash
sudo iptables -I INPUT 1 ! -i br-e71e5f2bd0ad -p tcp --dport 7897 -j DROP
sudo iptables -S INPUT
```

确认这条 DROP 规则位于 `-j ts-input` 等放行规则之前。规则也阻止宿主机回环访问以及
其他网桥访问 7897；同一应用网桥上的其他容器仍可使用代理。手动添加的规则通常在重启后
丢失；当前服务器按持续运行维护，重启或防火墙规则被重置后必须在建立隧道前重新检查/添加。

先启动个人电脑上的代理，再在**个人电脑**终端建立隧道，替换 SSH 用户和服务器地址：

```bash
ssh -v -N -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 172.17.0.1:7897:localhost:7897 user@server_ip
```

如使用 `PermitListen`，同步设置为 `172.17.0.1:7897`。在服务器验证监听和容器出网：

```bash
sudo ss -lntp 'sport = :7897'
docker compose exec -T api python -c 'import httpx; r = httpx.head("https://example.com/", proxy="http://host.docker.internal:7897", trust_env=False, timeout=15); r.raise_for_status(); print(r.status_code)'
```

当前部署已验证监听为 `172.17.0.1:7897`，容器请求返回 `200`。
`ss` 的监听范围看 `Local Address`，`Peer Address` 的 `0.0.0.0:*` 不代表监听所有网卡。
个人电脑休眠、代理退出或 SSH 断开会导致代理请求失败，需要恢复隧道。
当前 `NO_PROXY` 包含 `hf-mirror.com`，镜像站直连；ElevenLabs 默认绕过全局代理，
需要走隧道时在 `.env` 设置 `ELEVENLABS_BYPASS_GLOBAL_PROXY=false`，MiniMax 保持直连。
Compose 运行时代理不覆盖镜像构建和拉取，构建源及缓存说明见部署文档。

日常升级保留项目目录/Compose 项目名和网络配置，直接重建容器：

```bash
docker compose build
docker compose up -d --force-recreate api
```

这些操作复用现有网络，容器 IP 变化不要求修改按网桥匹配的规则。
`docker compose down` 会删除默认网络，再次 `up` 时网桥名称可能变化，见
[Docker down 文档](https://docs.docker.com/reference/cli/docker/compose/down/)。
删除/重建网络后，先核实新网桥、重新添加访问限制并移除旧规则，再复测代理；
旧规则会阻止新网桥访问 7897，表现为代理连接失败。

彻底停用时，先移除 Compose 和 `.env` 中的代理变量并重建容器，再结束 SSH 隧道、
恢复 sshd 设置，最后移除本次添加的防火墙规则。当前规则的撤销命令为：

```bash
sudo iptables -D INPUT ! -i br-e71e5f2bd0ad -p tcp --dport 7897 -j DROP
```

## 工程编辑与成品提交

`POST /api/jobs/<jobId>/mix?song=0|1[&variant=mixed]` 用本次保留的音轨覆盖一首歌的成品与
音轨文件，返回 202；进度和结果通过同一任务的 GET / SSE 获取。前端 Next 代理原样透传 JSON。

**替换前后是两首歌。** 每首候选里，原曲就是结果本身（`fullTrack`/`stems`/`audioRevision`），
「替换后」那首是 `mixed` 字段里一份同构的结果（自己的 `fullTrack`、`stems`、`stemUrls`、
`playback`、`waveforms`、`splitEnabled`、`audioRevision`、`mixConfig`），两首分别编辑、
分别覆盖，互不影响：

- `editor=replace`（只能 `variant=original`）读原曲的音轨与 `replacedVocal`，产出写进
  `mixed`（第一次提交时新建，之后每次覆盖它），原曲的成品、音轨、版本与可撤回删除都不变。
- `editor=tracks` 读哪首覆盖哪首：缺省 `variant=original` 覆盖原曲，`variant=mixed` 读并
  覆盖替换后那首；`variant=mixed` 时没有 `source=replaced`（那里的 `replaced` 已是普通音轨）。
- 替换后那首的文件与原曲在同一歌曲目录，带 `mixed_` 前缀；分轨 DELETE/PUT 同样接受
  `variant=mixed`，`X-Audio-Revision` 按所选那首校验，撤回记录分开记。
- 原曲重新替换人声、删除/恢复原曲音轨都不再作废替换后那首；`/api/voice/result` 拒绝删除
  两首歌的成品与保留音轨。
- 兼容字段：响应里 `mixedTrack`、`playback.mixedTrack`、`waveforms.mix` 镜像替换后那首的
  成品、试听与波形。旧记录里挂在原曲上的 `mixedTrack` 在加载时迁成单轨的 `mixed`；与原曲
  `fullTrack` 相同（覆盖式提交的旧形状）时只是同一首歌，直接丢弃。

`tracks` 是完整的最终保留列表（不是部分更新）：前端已经排除删除、静音（M）以及音量为
−∞ 的轨道。请求里没有 `editorState`/`muted`/`solo`/`projectRevision`，后端也不保存或
恢复编辑状态 —— 再次编辑时 M/S 全部关闭、所有滑杆从 0 dB 开始。后端只按「有没有有效
输入」判定：只排除人声、保留伴奏是合法提交（前端提交前用一次确认弹窗说明会消失的车道），
空列表才返回 400。

```json
{
  "commit": true,
  "audioRevision": 2,
  "editor": "tracks",
  "tracks": [
    {"id": "vocal", "source": "stem", "stemId": "vocal", "gainDb": -6}
  ]
}
```

- `audioRevision` 是当前歌曲及可编辑文件的版本，初始缺省 0，成功覆盖后加 1；不匹配返回
  409。它同时挡住过期提交和迟到响应覆盖新文件，与编辑状态无关。
- `editor` 为 `tracks` 或 `replace`：合轨失败时前端据此退回原编辑页面，不用于恢复控制状态。
- `source=stem` 读取当前歌曲的 `stems[stemId]`。保存时一律以请求的 `id` 为输出键、读取
  文件时用 `stemId`，两者不能混淆：普通编辑器的 ID 等于 stemId；替换编辑器里原人声的 ID
  为 `original:<stemId>`，但**已经带前缀**的键（重新分轨可能落下 `original:vocal`）保持
  原样、不再叠加前缀，此时 id 与 stemId 都取它；只保留替换人声时新 stems 里只有
  `replaced`。`replaced` 成为普通音轨后就是普通车道（`source=stem` + `stemId=replaced`，
  读 `stems["replaced"]`），后端不能因为它叫 replaced 而拒绝 —— 它和 `source=replaced`
  是两回事。
- `source=replaced` 的 ID 为 `replaced`，读取当前替换结果 `replacedVocal`，仅在选择它时
  校验 RVC 模型指纹、且本轮必须确实存在替换结果。`source=full` 的 ID 为 `full`，读取当前
  `fullTrack`，只用于未分轨的单轨编辑且必须独占输入。
- 轨道 ID 必须唯一且安全，空列表返回 400。`gainDb` 只接受有限数字，范围 `[-66, 6]`，
  0 表示线性增益 1。
- 每路输入先按 `volume=10**(gainDb/20)` 渲染出**新的音轨文件**，再对这些新文件
  `amix=inputs=N:normalize=0` 合成成品（支持只有一路）：不再做人声 EQ 或原曲响度补偿，
  所以调低音量不会被整体抬回。防削波限幅（−1 dBFS 真峰）**逐轨与成品各过一次**：保留
  音轨要能直接试听而不削顶，成品由这些已带增益的文件求和，不会再把增益应用第二遍。
  输出沿用当前歌曲的采样率、声道与 PCM 位深，并补齐或裁剪到歌曲长度（短轨补静音）。
- 成品、保留音轨的预览与波形全部就绪后原子提交：被覆盖那首的 `fullTrack` 指向新成品
  `[mixed_]master_v<版本>_<唯一标识>.wav`。分轨输入会**完整替换** `stems`、`stemUrls`、
  `playback.stems` 与分轨波形，只保留请求 id 对应的增益后文件，`splitEnabled=true`，
  不与旧 stems 做字典合并；单轨输入时 `stems`/`stemUrls` 为空、`splitEnabled=false`，
  新 `fullTrack` 成为下次单轨编辑的输入。
- 基于原曲的提交（替换编辑器、原曲的普通编辑器）都清除原曲上的 `replacedVocal`、
  `playback.replacedVocal`、替换波形与 replace 操作结果：替换编辑器已经把它写进替换后那首，
  原曲覆盖后它对应的人声也已经变了。编辑替换后那首不碰这些字段。
- `audioRevision` 对每首歌单调递增：替换编辑器写出的版本是 `max(替换后那首的版本, 请求版本) + 1`。
- `audioRevision` 加 1 并持久化成功后才发布 `mixStatus=succeeded`；新文件用新 URL，避免
  客户端缓存旧音频。混音、波形、试听、发布、保存失败或取消都不更新当前成品与版本，保留
  本轮输入和可撤回文件，只清掉本轮新建的临时产物。成功后才清理无引用的旧文件，因此一次
  覆盖之后不能撤回上一版删除的文件。
- 已有 `splitEnabled=true` 的歌曲再次 `/split` 直接返回当前保留音轨，不重跑 BS-RoFormer、
  也不补回被移除的轨道。未建立分轨时首次分离会推进 `audioRevision`，让前端丢弃旧单轨页面
  的控制与撤回记录。
- 分轨 DELETE/PUT 在修改前检查 `X-Audio-Revision`，替换产物 DELETE/PUT 检查 multipart
  `audio_revision`；缺省按 0 校验，不一致返回 409。删除/恢复本身不增加版本，返回的任务
  状态始终携带当前 `audioRevision`。独立删除原人声不连带删除当前替换结果（替换页对此只
  隐藏车道，完成时以完整输入列表覆盖）；反过来，撤回一条普通音轨也只会恢复它自己 ——
  已经软删除的替换结果由 `/api/voice/result` 自己的 PUT 恢复。
- 合轨与生成、拆轨、人声替换共享容量；额度用满返回 429，已有音频操作返回 409。合轨期间
  禁止该任务的输入文件删除/恢复、替换人声与拆轨。进度依次为 mixing=76、waveform=90、
  preview=95、completed=100，超时使用 `RVC_MIX_TIMEOUT_SECONDS`。输入不可读返回 404；
  缺少所选工程输入或版本过期返回 409；非法设置、文件归属或没有任何有效输入返回 400，
  不会生成空白成品。

`mixConfig` 原样回显本轮完整请求（包括顺序、版本号与增益）：202 的对应歌曲结果就带着它，
成功终态继续保留作为本次提交凭据，受理后的失败同样保留，前端据此退回对应的编辑页。
前端不据此初始化控制：首次打开及文件版本更新后，M/S、音量与撤回栈都重新初始化。

旧客户端在成品版本仍为 0 时可以不带请求体，继续使用“替换人声 + 当前伴奏轨”的
导出；四轨历史歌曲保持兼容，六轨歌曲会同时保留 guitar / piano，已上传的录音轨也会一并混入。这条兼容路径保留原 EQ、响度补偿和固定文件名，产出落成单轨的替换后那首；它不会提交
母带或编辑状态。只有结果里还没有 `audioRevision`（本次部署之前生成、且再次 `/split` 命中
缓存因而没有推进版本）时才会被受理：首次分离音轨就会把版本推到 1 以上，此后无请求体的
导出一律 409。旧记录里的 `projectRevision`/
`editorState`/`editorFullTrack`（含 `playback.editorFullTrack`、`waveforms.editorFull`）
在加载和响应渲染时被当作未提交设置丢弃并记日志，后端不会按旧设置重新渲染。

`mixStatus` 等运行态不落盘，重启后从结果推导成功。历史列表保留 `audioRevision` 与
`mixConfig`，按既有约定不返回波形或运行态。一首候选有替换后那首时列出两条记录（原曲与
替换后），各自播放、下载与编辑；多首都有替换后那首时 `mixSong=null`，客户端按每首结果判断。
202 回显的 `mixConfig` 在读输入的那首上（替换编辑器即原曲），成功后同时写在被覆盖的那首上。

生成音乐。`provider` 可传 `minimax_music`、`elevenlabs_music` 或 `mureka_music`，不传时使用
`MUSIC_PROVIDER`：

```bash
curl -X POST http://127.0.0.1:8010/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"明亮的普通话摇滚，清晰女声和有力鼓组","durationMinutes":"auto","provider":"minimax_music"}'
```

`durationSeconds`（10–360）用于 30 秒试听这类精确短时长，传了它就会覆盖
`durationMinutes`；MiniMax 和 Mureka 仍然忽略任何固定时长。可选的 `title`（最长 100 字符）是
群聊里选定的歌曲名；未传时会在歌词与风格扩写时生成简短歌名。标题随任务持久化，
并由 `GET /api/jobs` 和任务 SSE 返回，供历史页显示。

前端使用异步任务接口，以便刷新后恢复任务并显示真实阶段进度：

| 方法 | 路径 | 语义 |
| --- | --- | --- |
| `POST` | `/api/jobs` | 创建生成任务 |
| `GET` | `/api/jobs` | 列出全部历史任务（不含波形数据） |
| `GET` | `/api/jobs/{jobId}` | 查询任务状态与结果 |
| `GET` | `/api/jobs/{jobId}/events` | 通过 SSE 接收任务阶段和终态 |
| `PATCH` | `/api/jobs/{jobId}` | 局部更新任务状态（当前用于取消） |
| `DELETE` | `/api/jobs/{jobId}/stems/{stemId}` | 删除指定分轨及其输出文件 |
| `PUT` | `/api/jobs/{jobId}/stems/{stemId}` | 撤回删除并恢复指定分轨 |

当前没有完整替换任务资源的业务，因此不提供 `PUT`；未来需要整体替换任务配置时再增加。

SSE 终态始终使用具名 `done` 事件，客户端收到后应主动关闭 EventSource。若已连接的任务
在流中到期，`done` 的数据为 `{"success":false,"status":"expired","jobId":"...",
"message":"歌曲已过期。"}`，不包含音频结果；`expired` 仅表示该 SSE 终态的原因，
不会写入任务的持久化状态。连接前已过期的任务仍返回 HTTP 404。

ElevenLabs 优先使用 `/v1/music/stream`，收到首块音频前显示等待状态；接收期间
`stage=receiving_audio`，`receivedAudioSeconds` 表示当前这一首已收到的 PCM 音频秒数，
`expectedAudioSeconds` 表示请求的目标秒数（自动时长为 `null`）。进度更新通常最多每秒一次，
首块立即推送，流结束并原子保存音频后补齐最后一次更新。固定时长的 `progress` 按整个任务
聚合已完成歌曲数和当前歌曲接收比例，后处理及歌曲切换期间保留进度，最高为 99；
自动时长保持 `progress=null`，可直接展示 `message` 中的已接收秒数。
这表示音频接收量，不代表上游模型内部推理进度。双曲切换和后处理阶段清空
`receivedAudioSeconds`/`expectedAudioSeconds`，`progress` 保留当前任务进度；
所有歌曲均已完成或失败后，只要至少一首可用，任务就以 `progress=100` 结束。
MiniMax 继续使用上游回传的 `step/totalSteps`。

**硬性发布门槛：按首发布与部分成功改变了已发布客户端的响应契约，必须与支持这些
行为的前端同版本发布，禁止只升级后端。无法保证配套发布时，本版本不得上线。**
上线前须联调确认：运行中的 `result` 不会被当作任务成功或导致 SSE 关闭；
第二首未完成时不会按当前可用输出数夹取或覆盖用户的歌曲选择；部分成功按
`songNumber` 映射输出与状态，并保留可用歌曲。本仓库的后端测试不能代替前端联调。

双曲任务按首串行生成，每首完成文件保存、试听处理和波形提取后立即发布：
`GET /api/jobs/<jobId>`、历史列表和 SSE 的 `result` 都包含当前已完成的歌曲，
此时顶层 `status` 仍可为 `running`，不能仅凭 `result` 存在就关闭 SSE。
`currentSong` 是当前正在处理的歌曲序号（从 1 开始，终态为 `null`）；
`songStates` 按序号分别提供 `songNumber/status/stage/message/progress/error`，以及
每首自己的 `step/totalSteps` 和音频接收秒数。顶层 `progress` 是任务级进度。
顶层 `stage` 增加 `song_completed`/`song_failed`，两者仍属于任务运行阶段。
`songStates` 长度为实际尝试的曲数，可能小于 `requestedCount`（例如只支持单曲的 provider）；
首个进度上报前可能为空数组，请以 `songStates[].songNumber` 区分歌曲。
`songStates[].progress` 是单首进度（固定时长可用百分比，自动时长为 `null`，完成为 100）。
客户端可立即展示第一首播放器，并按 `currentSong` 显示第二首的进度。
SSE 仅在 `song_completed` 帧及终态帧附带波形，其余中间帧的 `waveforms` 为 `{}`，
不表示已保存的波形被删除。帧可能合并推进到下一阶段；新订阅或未收到歌曲完成帧的
客户端需要立即展示波形时，请查询 `GET /api/jobs/<jobId>` 获取完整结果。

第二首失败不会撤回第一首：任务以 `status=succeeded` 收尾，`warning` 说明部分失败，
`songStates[1].status=failed` 和 `error` 保存第二首的错误。两首都失败才令任务 `failed`。
明确的 provider 全局错误例外：鉴权、套餐、配额或限流错误（HTTP 401/402/403/429）以及
缺少必需配置会终止后续歌曲调用，任务标为 `failed`，但已经发布的歌曲仍可播放和下载。
ElevenLabs 的 402/403 流式端点拒绝仍先单次回退 compose；回退也失败才按全局错误终止。
生成被取消或服务器重启时，已经发布的歌曲也保留，未完成的歌曲分别标为 `cancelled` 或 `failed`。
`result.count` 始终等于可用歌曲数（`1 + alternatives.length`），`requestedCount` 是请求数；
每个输出的 `songNumber` 表示原始序号。因此第一首失败、第二首成功时，`result.count=1`、
`result.songNumber=2`，`alternatives=[]`。编辑接口的 `song` 仍是可用输出列表的零基索引，
这种情况下编辑第二首应传 `song=0`。音频编辑操作仍需等待任务结束。

首块音频到达前，上游不提供推理进度；本服务不会周期更新等待秒数，`message` 保持等待文案。
这段等待期间 SSE 仍约每 15 秒保活；保活表示连接存活，不表示上游生成取得了新进展。
流式端点返回 402/403/404/405 时，仅回退一次到 `/v1/music`，使用相同参数；
超时、断流或其它错误不重试。回退记录写入 `prompts.json` 的
`providerRequests[].streamFallbackStatus`，被拒响应的状态码、响应头与响应体保存在
`providerRequests[].streamFallbackResponse`；条目按 `variation` 区分，`variation=0` 对应第 1 首。
实际使用的端点记录在结果的 `debug.music.endpoint`。compose 回退可能集中返回音频，
因而无法保证持续更新接收量。进度观察回调失败只记日志，取消仍向上传播；完整音频已保存后
发生取消时保留该文件。2026-09-29 使用真实 Key 实测流式端点：`music_v2`、
`pcm_44100`、`music_length_ms=3000`、纯钢琴器乐请求返回 HTTP 200；408 个音频块共
529,200 字节，保存后的 WAV 为 44.1 kHz、16 位双声道、3.000 秒。

```bash
# 创建任务（返回 202 和 jobId）
curl -X POST http://127.0.0.1:8010/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"明亮的普通话摇滚","durationMinutes":2}'

# 查询任务状态、阶段、进度和最终 result
curl http://127.0.0.1:8010/api/jobs/<jobId>

# 订阅阶段进度；收到具名 done 事件后客户端应主动关闭 EventSource
curl -N http://127.0.0.1:8010/api/jobs/<jobId>/events

# 对已完成的历史歌曲按需执行六轨分离；已有分轨时直接返回缓存
curl -X POST 'http://127.0.0.1:8010/api/jobs/<jobId>/split?song=0'

# 取消任务
curl -X PATCH http://127.0.0.1:8010/api/jobs/<jobId> \
  -H 'Content-Type: application/json' \
  -d '{"status":"cancelled"}'

# 删除已生成的某条分轨（同时删除 output 中的对应文件）
curl -X DELETE http://127.0.0.1:8010/api/jobs/<jobId>/stems/vocal

# 撤回删除
curl -X PUT http://127.0.0.1:8010/api/jobs/<jobId>/stems/vocal
```

任务元数据写入 `output/jobs/<jobId>/job.json`，完整歌曲、分轨、RVC 人声和最终混音统一
写入 `output/jobs/<jobId>/song_<n>/`。服务重启后会重新加载历史。
重启时仍为 `pending` 或 `running` 的任务会恢复为 `failed`，并标记“服务器重启，生成任务已中断”；
不会自动续跑未完成的音乐生成。执行队列和并发计数仍为单进程状态，因此仍推荐单
Uvicorn worker；需要多 worker 时再接入共享任务存储。

### 创作保留与清理日志

永久清理默认关闭（`SONG_RETENTION_ENABLED=false`），默认仅审计（`SONG_RETENTION_DRY_RUN=true`）。
**正式启用会追溯全部历史：第一次启动就会永久删除所有超过 N 天的既有任务及音频，无法恢复。**
先备份完整 `output/`，设置 `SONG_RETENTION_ENABLED=true` 并保持 `SONG_RETENTION_DRY_RUN=true`
重启，检查日志 `wouldDeleteJobIds`、跳过的任务以及 health 中的孤立目录；确认影响范围后，
才将 `SONG_RETENTION_DRY_RUN=false` 并再次重启，正式执行删除。

关闭或 dry-run 时不删除、不移动目录、不隐藏历史、不限制访问，试听缓存沿用原有策略。
关闭时任务 HTTP/SSE 响应不返回 `expiresAt`、`retentionState`、`audioAvailable`，保留原有响应结构；
开启 dry-run 时返回这三个字段，`expiresAt=null`。dry-run 启动时及每天凌晨 03:00 仅记录候选
`wouldDeleteJobIds`。正式启用前需 PO 在 PR 或上线工单明确确认追溯删除，并完成前端对新增字段、
`done/status=expired`（无结果）和音频 404 的兼容验证。`SONG_RETENTION_DAYS` 范围为 1–3650 天。

正式启用后，创作从任务 `createdAt` 起可访问连续 72 小时（`SONG_RETENTION_DAYS=3`），播放、
拆轨、换声和合轨不延长访问期。历史、详情和 SSE 响应提供 `expiresAt`，用于前端提前提示。
到期且执行结束后任务不再出现在历史中，详情、编辑和新音频下载请求返回 404；试听 MP3 的缓存时间
不超过剩余访问期。已经开始的下载可完成，已下载或此前已缓存到客户端的内容无法追回。

正式启用后，服务启动时在后台清理一次，不阻塞 API 就绪；随后每天北京时间凌晨 03:00 清理。
正常情况下文件在创建后 72–96 小时内删除，容量规划应按接近 4 天计算；仍在执行的任务
或删除失败的目录可能保留更久。访问期为 72 小时，磁盘释放时间取决于清理执行。
清理包含整个任务目录 `output/jobs/<jobId>/` 和回收区 `output/.trash/<jobId>/`，
一次任务中的所有歌曲一并永久删除。
仍在生成、拆轨、换声或合轨的任务跳过本轮，实际执行结束后在下次清理时删除；换声超时
但推理线程仍在运行时也不会删除。创建时间缺失、无时区或格式异常的历史记录不会被猜测删除，
需要根据日志人工修复。过期但仍在执行的任务继续出现在历史中，详情、SSE 和 PATCH 取消
仍可使用，SSE 不会提前发送 `done/expired`；新编辑和音频下载仍被拒绝，实际执行结束后再清理。
RVC 推理线程无法强制停止，取消仅标记取消意图，并发额度仍等到实际线程退出后释放。
没有任务元数据的孤立文件不会按文件修改时间自动删除；正式启用时，这些任务及无法计算
到期时间的任务音频返回 404。孤立目录列在启动日志和 health 中，需人工修复或处理。

策略开启后的任务响应中，`retentionState` 明确区分 `dry_run`（仅审计）、
`retained`（可访问）、`expired_active`（已过期但仍在执行）和 `unmanaged`（创建时间异常）。
`audioAvailable` 表示任务是否已有结果且保留策略允许音频访问，不代表对每个音频文件做了
磁盘校验。正式清理时，`unmanaged` 任务仍可查看以便修复，但 `audioAvailable=false`；
前端应显示“元数据异常，需修复”，禁用播放/下载，不能把 `expiresAt=null` 当作无限期可播放。
修复 `job.json` 后重启服务重新加载；不会猜测时间，也不会自动删除这些任务。

删除前先将目录移入不可访问的 `output/.expired/<jobId>/`；删除失败会保留目录，下一次清理
（含服务重启）重试，避免把部分删除的任务重新加载到历史中。
暂存前先校验全部源路径、目标路径及目标冲突。重试暂存区条目时，若任务仍被加载且按当前
保留天数尚未过期（或创建时间不可解析），跳过删除；已移走元数据的任务仍可继续重试。

`output` 树内（包括 `jobs`、`.expired`、`.trash` 和任务目录）不支持符号链接。
保留清理拒绝沿符号链接操作；指向 `output` 之外的链接，其 `/output` 音频请求返回 404。
音频放在其它卷时，请将卷直接挂载到 `OUTPUT_DIR` 或其真实子目录，不要用符号链接转接。
后台首轮清单检查 `jobs` 和 `.expired` 布局，发现链接时记录一次 WARNING 的
`unsupported_layout`；重复刷新同一问题不重复该告警。health 的缓存字段 `unsupportedLayout`
为 `true`，`unsupportedLayoutDirectories` 只列 `jobs`/`.expired` 相对名称；错误原因明确包含
`refusing symlinked directory`。首次检查前或无法确认布局时为 `null`，修复后下一次刷新恢复
`false`；这些根目录检查不代表完整扫描了树内所有符号链接。

清理日志保存到 `output/logs/retention.log`，同时接入现有应用日志；每行是一个 JSON 对象，
`time` 使用带时区的 UTC 时间。启动时记录开关、天数、清理时区和下次计划清理时间；
文件日志初始化失败时告警并降级到应用日志，不阻止 API 启动。
日志记录清理开始、删除尝试、每个任务删除成功/失败或跳过的
时间与原因，结束时汇总 `deletedJobIds`、`failedJobIds` 和 `skippedJobIds`。没有过期任务时也
记录开始和结束。单文件最大 5 MiB，保留 3 个轮转备份；日志不会随歌曲清理，也不通过
`/output` 对外提供。仍须保持单 worker、单实例写入该 output 目录。

`cleanup_failed` 记录扫描或整轮错误：`stage=pendingDeletionScan` 表示放弃暂存区残留重试，
仍继续处理内存中的已知过期任务；`stage=run` 表示本轮中止。`cleanup_finished` 包含
`result`、`error` 和 `pendingDeletionScanFailed`。暂存区本身损坏或无法写入时，无法安全移动
目录，相应任务记录 `job_delete_failed` 并进入 `failedJobIds`，保留原文件以便修复后重试。

批次汇总清单最多包含前 20 个 jobId，并附带对应的 `Count` 和 `Truncated`；dry-run 的
`wouldDeleteJobIds` 也遵循这个限制。核对完整范围时查看逐任务 `job_would_delete`、
`job_deleted`、`job_delete_failed` 和 `job_skipped` 记录，不能只依据截断后的汇总。

`GET /api/health` 的 `retention` 返回 `enabled`、`days`、`cleanupTimezone`、`nextCleanupAt`、
`cleanupRunning` 和 `remainingJobs`。关闭时下次清理时间为 `null`；剩余数量包含正在删除的
任务，仅代表本轮待处理数量（本轮跳过/失败的任务见日志）。
`lastCleanupAt` 是上一轮结束的 UTC 时间，`lastCleanupReason` 是触发原因，
`lastCleanupResult` 为 `success`、`partial_failure`（有任务处理成功，但仍有错误）、`failed`
或 `cancelled`；关停时完成当前删除后停止批次，剩余任务留待下次清理，结果为 `stopped`。
`lastCleanupError` 给出脱敏后的首个错误，`pendingDeletionScanFailed`
报告上一轮是否无法扫描暂存区。尚未执行时这些字段为 `null`；本轮执行期间继续显示上一轮结果。
dry-run 下 `success` 仅表示审计完成，未删除或移动任何内容；`partial_failure` 也仅表示部分
审计完成。正式模式的 `success` 表示本轮删除完成或无需删除。判断删除是否已执行必须同时
核对 `dryRun`、`deletedJobIds` 和 `wouldDeleteJobIds`，不能只依据结果枚举。

health 只读取缓存，不在请求中扫描目录或解析全部任务时间。后台线程在启动时、每小时、
以及清理后刷新清单，策略关闭时也会刷新。`inventoryUpdatedAt` 仅在整轮扫描成功时推进，
`inventoryAttemptedAt` 表示最近尝试时间。首次刷新前时间、计数和 `Truncated` 为 `null`，
`inventoryStale=true`。正常刷新时清单最多滞后一小时，运行标志与本轮剩余数量仍实时返回。

health 和清单日志报告 `dryRun`、`expiredActiveJobIds`、`orphanJobIds`、`pendingDeletionJobIds`
及 `unmanagedJobIds`。每类清单最多 20 条，分别附有 `expiredActiveCount/Truncated`、
`orphanCount/Truncated`、`pendingDeletionCount/Truncated` 和 `unmanagedCount/Truncated`。
`unmanagedJobs` 给出前 20 个任务的具体原因，包括创建时间异常、元数据缺失或 `job.json`
JSON/字段校验失败；health 的错误原因不包含原始作品内容或绝对路径，I/O 错误仅报告
异常类型、errno 和标准说明。完整元数据加载错误及路径见应用日志。
JSON 语法错误包含解析说明与行列，字段错误用点分隔字段名（如 `result.fullTrack: missing`），
目录与 `jobId` 不一致明确报告 `jobId does not match directory`。
关闭策略时 `.expired` 残留不会继续删除，但仍报告其摘要；重新正式启用后才重试。
三类扫描（待删除目录、孤立/异常元数据任务、活跃任务）独立进行。失败时 `inventoryError`
给出原因、`inventoryStale=true`，失败类别保留上次成功数据并标记对应的 `*Stale=true`；
没有成功数据的类别，其 `Count` 和 `Truncated` 保持 `null`。其它类别仍刷新。
计数、空列表或非空 `inventoryUpdatedAt` 均不能单独作为“确认没有残留”的依据。

后台循环遇到意外异常会记录 `retention_monitor_failed` 和应用堆栈，在下次清单刷新间隔重试，
不会因单轮异常永久停止调度；正常取消仍向上传递。此功能不更改全局日志或 httpx 日志级别；
文件日志不可写时，仅 retention 自身增加 stderr handler，关停时释放。

音频接口先非阻塞打开并确认普通文件，再恢复阻塞读取；FIFO、目录和 Unix socket 返回 404。
普通文件不存在返回 404，权限不足返回 403，文件描述符耗尽返回 503（`Retry-After: 5`），
其余读取前 I/O 故障返回 500 并记录堆栈。服务端故障不能作为“歌曲已过期”的信号；
响应头已发送后发生的磁盘读取故障只能中断流，客户端应检查下载是否完整。

成功响应继续包含：

```text
jobId, prompt, durationMinutes, structuredPrompt, fullTrack,
stems, stemUrls, waveforms, splitEnabled, debug,
audioRevision, mixConfig, replacedVocal, mixedTrack, playback
```

双格式下载沿用结果字段，`result.alternatives` 中每首候选使用相同映射：

| 音频 | WAV | MP3 |
| --- | --- | --- |
| 原曲 / 完整混音 | `fullTrack` | `playback.fullTrack` |
| 合轨成品 | `mixedTrack` | `playback.mixedTrack` |
| 单轨 / 原人声 | `stems[stemId]` | `playback.stems[stemId]` |
| 替换人声 | `replacedVocal` | `playback.replacedVocal` |

任务列表、详情、SSE、替换完成（含缓存命中）与重启恢复共用 URL 渲染逻辑。
WAV 对应的 MP3 位于同歌曲的 `playtrack/` 下，文件名包含 WAV 的修改时间和大小；
读取结果会复用磁盘上的当前版本，即使旧元数据遗漏了 `playback`，也不会重新编码。
没有有效 MP3 时不返回对应地址。音频从现有 `/output/` 出口下载，支持 Range 和查询参数；
WAV 被软删除或版本改变后，其旧 MP3 链接返回 404，撤回删除会恢复有效地址。
试听 MP3 固定用 CBR192 编码，文件名带 `cbr192` 版本标记；旧版 VBR 试听不再发布（链接
返回 404），旧任务在重新分轨/替换/合轨生成新试听前回落到 WAV。

兼容例外：旧任务的 `fullTrack`（或旧单轨字段）可能本来就是 MP3。
后端保留该地址，并把它同步投影到对应的 `playback` 字段，不改写旧任务的原始引用。
前端应按 URL pathname 的真实扩展名判断格式，仅在原始字段以 `.wav` 结尾时启用 WAV 下载；
替换页的当前状态 `output` / `playback` 分别取当前歌曲的 `replacedVocal` /
`playback.replacedVocal`。后端没有单独的 `output` 响应字段。
只有 MP3 的旧任务没有可下载的 WAV，本次不做批量转码；如需补齐，必须实际解码生成 WAV，
并通过新增 WAV 地址字段提供给前端（需要同步适配），不能只改扩展名，也不能恢复压缩前的音质。

`waveforms` 是每条音轨的归一化 RMS 包络，固定 640 个 bin
（`app/services/waveforms.py` 的 `WAVEFORM_BIN_COUNT`）。每轨按自身峰值归一化到
`0..1`，保留轨内强弱变化；波形高度表示轨内相对幅度，不用于比较不同轨的实际音量。
归一化前将 RMS 低于 −60 dBFS 的采样段按静音显示，避免分离残留被放大成假波形。
完整音频、所有分轨、替换人声、合轨和上传录音共用此规则；只影响波形数据，不修改音频。
按原始采样率计算双声道能量，避免高频丢失或左右声道反相抵消。
多轨编辑器按车道满宽绘制时 640 个点才像波形，且同一个 `waveforms` 字典内的所有音轨共享同一长度，客户端可以用同一
根 x 轴对齐。因此：

- `GET /api/jobs/{jobId}` 返回真实包络；`GET /api/jobs` 是历史列表投影，其中的
  `waveforms` 一律为空对象 `{}`（列表不绘制编辑器车道，避免按任务数放大响应体积）。
- `GET /api/jobs/{jobId}/events` 只在终态 `done` 帧里带真实波形，中间帧的 `waveforms`
  同样是 `{}`：任务结束前 `result` 不会变化（分轨也在收尾那一刻才写入波形），而分轨
  期间的中间帧每 15 秒就会被保活超时重推一次，带上波形等于把同一份 640-bin 包络重复
  发送二十多次。需要随时拿真实包络时走 `GET /api/jobs/{jobId}`。
- 拆轨时会连同 `full` 一起重新提取波形：早期版本把 `full` 存成 64 个 bin，重新提取后
  同一首歌的所有车道都统一为 640 个 bin，不会出现长短不一的车道。

## 按需六轨分离

后端使用 [BS-RoFormer-SW](https://github.com/openmirlab/bs-roformer-infer)，固定输出
`vocal / drums / bass / guitar / piano / other` 六轨，guitar 和 piano 排在 other 前面。
详情及 SSE 响应也会按此顺序排列已有歌曲的分轨。运行库固定为 `bs-roformer-infer==0.1.5`，
与现有 Python 3.10、NumPy 1.23.5、CUDA Torch 2.7.1 兼容。只保留这三个分轨配置：

```env
BS_ROFORMER_DEVICE=cuda
BS_ROFORMER_MODELS_DIR=./output/.bs-roformer-models
BS_ROFORMER_TIMEOUT_SECONDS=1800
```

代码默认设备为 `auto`（有 CUDA 用 CUDA，否则 CPU）；可显式设置 `cpu`、`cuda` 或
`cuda:N`，指定的 CUDA 设备不可用时直接报错。首次分轨自动下载并校验六轨权重，约 700 MB；
缓存目录必须可写，Docker 使用 `output` 挂载持久化，不对外提供下载。也可部署前预下载：

```bash
uv run bs-roformer-download --model roformer-model-bs-roformer-sw-by-jarredou \
  --output-dir ./output/.bs-roformer-models
```

分轨在独立子进程运行：先把母带转为 44.1 kHz 双声道 WAV，再执行六轨模型；转换、
首次权重下载和推理共用一个超时预算。六轨全部生成并验证后才发布，每次生成独立文件名，
失败或取消不会覆盖当前音频，临时目录会清理。模型额外产生的 instrumental 不加入编辑器，
避免重复播放伴奏；模型的 `vocals` 映射成既有的 `vocal` 键。前端应遍历实际 `stems` 键。

已有工程默认继续按缓存返回；要重建当前母带的六轨，调用
`POST /api/jobs/{jobId}/split?song=0&variant=original&force=true` 并带
`X-Audio-Revision: <当前 audioRevision>`。返回 `202` 后继续通过 GET/SSE 等待；成功会将
`audioRevision` 加 1，`debug.splitterEngine=bs_roformer`、`debug.splitterModel` 记录实际模型。
`variant` 可选 `original` / `mixed`，分别重建原曲或替换后的工程。
`force=true` 会重置所选工程的草稿音轨及其命名、删除撤回和人声替换结果；需要保留的录音与音量调整应
先合轨提交到母带。母带和其它候选歌曲不受影响，失败或在结果发布前取消会回滚旧工程。
未带 `force` 时仍保持原有的缓存、删除车道不自动补回的行为。

音乐生成只产出完整混音，不会自动拆轨。从历史记录进入编辑时，调用分轨接口执行
BS-RoFormer 六轨分离并通过任务 SSE 推送真实进度；`splitEnabled=true`（或 `stems` 非空）时
未请求 `force=true` 时直接返回**当前保留的音轨**，不会重新运行 BS-RoFormer，也不会补回被移除的轨道 —— 再次编辑
只编辑保留下来的文件。尚未建立分轨时首次分离会推进 `audioRevision`，让前端丢弃旧单轨
页面的控制与撤回记录；分离出的原始人声与伴奏就是此后编辑与混音的输入。

`GET /api/health` 的 `splitting` 字段返回 `engine`、`model`、`device`、`stems`、
`timeoutSeconds`；Demucs 时代的 `profile`、`jobs`、`segment` 已移除。

拆轨的 `operationStage` 依次经过 `starting_split`（读取输入，0）、`splitting`（BS-RoFormer，25）、
`waveform`（真实波形，50–75）、`preview`（每轨试听 MP3，75–99）和 `completed`。
`operationProgress` 按实际处理波形/试听轨道数更新；只有结果可用且操作成功时才报告 100、
`splitStatus=succeeded` 和 `operationStatus=succeeded`。POST、GET 与 SSE 共用独立操作字段，
SSE 按 `operationStatus` 判断音频操作是否结束，主生成任务成功不会提前关闭操作流。
SSE 保存阶段快照，队列最多 16 帧；慢客户端积压时丢弃最旧更新，失败/取消优先于缓冲帧。
生成进度通知可合并，队列非空表示消费者已有待处理通知；所有发布点使用同一 helper。
后端不人为延时。`preview` 阶段的 `result.stems`、`splitEnabled`、`waveforms` 已经写入，
`splitStatus` 仍为 `running`：

- **`preview` 阶段取消只取消编码器**：`PATCH /api/jobs/{jobId}` 仍会取消该阶段的
  ffmpeg 编码任务，因为 WAV 分轨与波形都已落盘，`splitStatus` 不被改写，任务照常以
  `succeeded` 收尾，只是 `result.playback.stems` 缺少试听 MP3（客户端回落到 WAV）。
  这样 PATCH 的响应与最终状态不会互相矛盾。输入读取、`splitting`/`waveform` 阶段取消才是终态：
  `splitStatus=cancelled`。
- **分轨进行中（`splitStatus` 为 `pending`/`running`，含 `preview` 窗口）拒绝删除/恢复
  分轨**，返回 `409`：此时编码器正在读写 `playtrack/`，删除会丢失播放引用，恢复会永久
  降级为 WAV。`/split`、`/replace` 的同类冲突防护不变。
- 单轨试听编码失败或被取消只影响那一轨（该 key 不出现在 `result.playback.stems`），
  不会把整个分轨判为失败。

## Track 页面：用户录音上传

`POST /api/jobs/{jobId}/recordings?song=0&variant=original`，请求使用
`multipart/form-data`，只有两个业务字段：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `name` | 文本，必填 | 录音名称，去掉首尾空白后不能为空；最多 100 字，禁止控制字符和不可见格式字符（双向覆盖、零宽空格等；ZWNJ/ZWJ 允许） |
| `file` | 音频文件，必填 | 浏览器录音 Blob 或本地文件；支持 WebM/Opus、MP4/M4A、WAV、MP3、OGG、FLAC、AAC |

`song` 从 0 开始，选择候选歌曲；`variant` 为 `original` 或 `mixed`，与现有分轨、合轨接口
一致。上传前歌曲必须生成成功且完成分轨（`splitEnabled=true`）；单轨歌曲先调用 `/split`。
请求头 `X-Audio-Revision` 填当前工程的 `audioRevision`，未提供按 0 验证。
默认文件上限 50 MB，由 `RECORDING_MAX_UPLOAD_MB` 设置；时长上限跟随
`MAX_DURATION_MINUTES`，默认 360 秒。按音频内容解码，不依赖扩展名和 MIME；
服务端只读取本地音频容器，不接受播放列表或远程音频地址。

成功同步返回 `201`，录音已保存，无需等待任务轮询：

```json
{
  "id": "recording_0123456789abcdef0123456789abcdef",
  "name": "我的和声",
  "audioUrl": "http://localhost:8000/output/jobs/job-id/song_1/recording_0123456789abcdef0123456789abcdef.wav",
  "playbackUrl": "http://localhost:8000/output/jobs/job-id/song_1/playtrack/recording_0123456789abcdef0123456789abcdef.playback-cbr192-<mtime_ns>-<size>.mp3",
  "durationSeconds": 12.5,
  "waveform": [0.2, 0.7, 0.4],
  "audioRevision": 2
}
```

`waveform` 通常为 640 个归一化 RMS 幅值，提取失败时为 `[]`；`playbackUrl` 是试听 MP3，
编码失败时为 `null`，前端回落到 `audioUrl`。原始录音转成 44.1 kHz、双声道 PCM WAV
作为合轨输入。示例中的波形与试听文件名已缩写。

录音就是普通音轨：返回的 `id` 同时是 `stems[id]`、`waveforms[id]` 和
`playback.stems[id]` 的键。新字段 `stemNames[id]` 保存用户命名，原有模型分轨没有命名
时仍使用前端内置的标签。`GET /api/jobs/{jobId}`、SSE、历史记录和重启后均保留命名及音频。
前端可直接使用上传响应添加车道，或重新获取任务详情；务必更新当前工程版本号，上传使
`audioRevision` 加 1。每首工程最多 32 条音轨。

```ts
const form = new FormData();
form.append("name", recordingName.trim());
form.append("file", recordedBlob, "recording");
const response = await fetch(
  `${apiBase}/api/jobs/${jobId}/recordings?song=${song}&variant=${variant}`,
  {
    method: "POST",
    headers: { "X-Audio-Revision": String(audioRevision) },
    body: form,
  },
);
// 不要手动设置 Content-Type，浏览器自动生成 multipart boundary。
if (!response.ok) throw new Error((await response.json()).message ?? "录音上传失败");
const recording = await response.json();
// 用 recording.audioRevision 更新版本；默认音量 0 dB，从歌曲起点播放。
```

合轨继续调用 `/mix`，录音车道提交为
`{"id":"recording_<id>","source":"stem","stemId":"recording_<id>","gainDb":0}`。
选择录音后，必须在 `tracks` 中包含所有要保留的音轨；未列入的车道会按现有覆盖规则移除。
合轨后保留所选录音的名称。删除与撤回复用
`DELETE /api/jobs/{jobId}/stems/{id}` 和 `PUT /api/jobs/{jobId}/stems/{id}`，查询参数和
版本头与上传相同。录音默认从 0 秒开始，合轨时按母带长度补静音或截断。

错误沿用 `{ "success": false, "message": "..." }`：`400` 表示空白名称、空文件、
无效音频或超长录音；`404` 表示任务/候选/替换后工程不存在；`409` 表示未完成生成/分轨、
版本过期、音频操作冲突或音轨数已满；`413` 表示上传过大；`429` 表示处理容量已满；
`500` 表示处理工具或存储故障。缺失字段、超出原始名称长度或查询参数无效时，
FastAPI 返回标准 `422` 校验错误。解码失败、版本冲突、取消或存储失败均不增加车道，
并清理本次临时文件。

强制重新分轨（`/split?force=true`）只替换模型分轨：已上传的录音连同名称、波形与试听原样
保留。重建成功后，不再被任何工程或撤回记录引用的旧分轨与旧试听文件会被删除。

## 本地基础设施模式

当前版本不需要 Redis 或消息队列：

```env
TASK_BACKEND=inline
CACHE_BACKEND=none
EVENT_BACKEND=none
```

对应接口位于 `app/infrastructure/`。后续可以新增 Redis/队列实现而不改变
`GenerationOrchestrator` 的音乐生成业务流程。不要在当前版本中把
`TASK_BACKEND` 或 `CACHE_BACKEND` 设置为未实现的值；配置层会在启动时拒绝它们。

## 测试

运行全部本地测试：

```bash
uv run pytest
```

测试分为：

- `tests/unit`：配置、Prompt、Provider、BS-RoFormer 命令、音频预处理、完整发布、超时取消和基础设施接口
- `tests/integration`：FastAPI API 契约、并发、音频下载
- `tests/functional`：启动真实 Uvicorn TCP 服务完成生成与六轨下载

`tests/functional/test_frontend_track_mix.py` 还会在临时副本里起同目录 `SHUJI-BAND`
（可用 `SHUJI_FRONTEND_DIR` 指定）的真实 Next 路由，转发到真实 Uvicorn 与真实 FFmpeg，
核对「完成创作」的覆盖提交链路。缺少前端检出、Node 或 ffmpeg 时该项跳过。

真实 BS-RoFormer 冒烟测试默认跳过，因为首次运行需要下载模型且耗时较长：

```bash
RUN_REAL_BS_ROFORMER_TEST=1 uv run pytest -m slow \
  tests/functional/test_real_bs_roformer_optional.py
```

质量检查：

```bash
uv run ruff check .
uv run pytest --cov=app --cov-report=term-missing
```

## 运行约束

本地内联执行使用进程内并发限制。BS-RoFormer 模式建议保持单 Uvicorn worker：

```text
MAX_CONCURRENT_GENERATIONS=1
```

如果未来使用多个 API worker，应先实现共享任务队列或分布式锁，不能依赖每个
进程独立的内存计数器。
