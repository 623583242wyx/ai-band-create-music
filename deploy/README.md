# 音乐生成独立部署

本目录的 `component.yaml` 声明音乐服务构建目标、8010 端口、输出数据卷和 live/ready 探针；`compose.yaml` 用 Compose 项目名隔离网络与命名卷。部署台（3199）的音乐适配器按此声明校验并据此构建、探活与登记数据卷，`spec` 须与 `contract-music.json` 完全一致。

## 首次部署

在仓库根目录准备环境配置和 RVC 资产，再启动独立环境：

```bash
cp .env.example .env
chmod 600 .env
# 按需修改 LLM、音乐 Provider、APP_PORT、APP_UID/APP_GID
docker compose -f deploy/compose.yaml config --quiet
docker compose -f deploy/compose.yaml build
docker compose -f deploy/compose.yaml up -d --wait --wait-timeout 120
```

每个环境使用不同的 `COMPOSE_PROJECT_NAME` 和 `APP_PORT`。项目名变化会创建新的网络与命名卷，不会读取其他环境的数据：

```bash
COMPOSE_PROJECT_NAME=shuji-music-test APP_PORT=18010 \
  docker compose -f deploy/compose.yaml up -d --build --wait --wait-timeout 120
```

## 数据隔离

- 服务数据统一写入容器内 `/app/output`，对应 Compose 命名卷 `music-output`；卷名带项目前缀，不同 `COMPOSE_PROJECT_NAME` 天然隔离。
- `MUSIC_ASSETS_DIR` 只挂载只读 RVC 资产；不要把它指向另一个环境的输出目录。
- 日常停止使用 `docker compose -f deploy/compose.yaml down`，保留命名卷；`down -v` 会删除该环境输出，不用于日常停止或回滚。
- `OUTPUT_DIR`、`BS_ROFORMER_MODELS_DIR` 由 Compose 固定到数据卷，不要另设到容器临时层。

## 探针与版本

```bash
curl --fail http://127.0.0.1:${APP_PORT:-8010}/health/live
curl --fail http://127.0.0.1:${APP_PORT:-8010}/health/ready
curl --fail http://127.0.0.1:${APP_PORT:-8010}/version
```

- `live`：进程可响应。
- `ready`：输出卷可写，且当前音乐 Provider 的必要配置存在；不探测外部生成服务。
- `version`：组件、源码版本、镜像配方、环境名、运行模式与 Provider，不返回密钥。

## 更新

`SOURCE_REVISION` 由构建参数传入，镜像标签默认 `latest`；发布时使用唯一标签并记录镜像 ID：

```bash
export SOURCE_REVISION="$(git rev-parse HEAD)"
export IMAGE_TAG="$(git rev-parse --short HEAD)"
docker compose -f deploy/compose.yaml build
docker compose -f deploy/compose.yaml up -d --no-build --wait --wait-timeout 120
```

构建和启动必须在同一个 shell 中执行，或通过 `--env-file` 明确传入同一份发布变量。`up` 不继承上一条命令的临时环境，否则会退回到 `.env` 或默认的 `latest` 镜像。

服务更新会重启单个进程；正在运行的生成任务可能中断。需要保留旧输出时先停止服务，按数据备份流程处理后再切换。
