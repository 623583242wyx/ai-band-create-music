from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import SecretStr

from app.core.config import Settings
from app.core.errors import GenerationError
from app.services.job_files import update_job_diagnostics

STYLE_TAG_MIN = 6
STYLE_TAG_MAX = 8
STYLE_TAG_KEY_MAX_CHARS = 80
STYLE_TAG_VALUE_MAX_CHARS = 800
GENERATED_LYRICS_MAX_CHARS = 1000
GENERATED_TITLE_MIN_CHARS = 2
GENERATED_TITLE_MAX_CHARS = 10
GENERATED_TITLE_PATTERN = re.compile(
    rf"[\u4e00-\u9fff]{{{GENERATED_TITLE_MIN_CHARS},{GENERATED_TITLE_MAX_CHARS}}}"
)
TITLE_REQUIREMENT = (
    f"根据歌词的主题意境拟一个文雅、贴切的简体中文歌名，"
    f"限 {GENERATED_TITLE_MIN_CHARS}-{GENERATED_TITLE_MAX_CHARS} 个汉字，"
    "不要书名号、引号、标点或风格标签；在 JSON 中增加字符串 title。"
)
REQUIRED_STYLE_CATEGORY_ALIASES = (
    ("genre", "style"),
    ("tempo", "rhythm", "meter"),
    ("mood", "emotion"),
    ("instrument", "guitar", "drum", "bass"),
    ("vocal", "voice"),
    ("arrangement", "structure", "section"),
    ("production", "mix", "recording"),
    ("negative", "exclusion", "avoid"),
)
STYLE_TAG_REQUIREMENTS = "\n".join(
    [
        f"styleTags 必须包含 {STYLE_TAG_MIN}-{STYLE_TAG_MAX} 个详细英文音乐制作标签，"
        "每个标签严格使用 [Category: value] 格式。",
        "整体覆盖 Genre and Era、Tempo and Meter、Mood、Instrumentation、Vocal、"
        "Arrangement、Production and Mix、Negative Constraints；允许在同一标签中合并相邻类别。",
        "标签值应包含可听见、可执行的细节，例如 BPM、拍号、律动/切分、调式/音阶、"
        "音程走向、音色、中文人声唱法、段落推进、空间效果、动态变化和需要避免的声音。",
        "Tempo and Meter（节奏与拍号）标签除 BPM 和拍号外，必须写明律动/切分与"
        "调式/音阶、音程走向等旋律节奏特征。",
        "每个原始标签值不超过 200 个字符；需要合并时仍须保持信息完整。",
        "严格遵守用户提供的风格要求，不得替换或违背；"
        "为未指定项补充协调一致的专业选择；不得引用具体艺人或受版权保护作品。",
        "Vocal 标签必须描述可演唱的主唱（音色、唱法、语言）；"
        "禁止出现 no lyrics、no vocals、instrumental、wordless vocal，以及「无歌词」"
        "「纯音乐」「无人声」「仅哼唱」等排除歌词的描述。"
        "Negative Constraints 也不得包含这些排除人声的词。",
    ]
)

LYRICS_AND_STYLE_SYSTEM_PROMPT = "\n".join(
    [
        "你是专业音乐制作人与歌词结构编辑，同时完成歌词分段和音乐风格扩写。",
        "只能插入 [intro]、[verse]、[pre-chorus]、[chorus]、[hook]、[post-chorus]、"
        "[bridge]、[instrumental]、[solo]、[outro]，标签必须单独占一行。",
        "不得增加、删除、改写、重排或重复任何歌词。",
        STYLE_TAG_REQUIREMENTS,
        "只返回 JSON 对象，包含字符串 taggedLyrics 和字符串 styleTags；不要解释或输出代码块。",
    ]
)

GENERATE_LYRICS_AND_STYLE_SYSTEM_PROMPT = "\n".join(
    [
        "你是专业音乐制作人与填词人，根据用户的创作要求同时生成原创歌词和音乐风格说明。",
        "默认创作可直接演唱的原创简体中文歌词，使用 [intro]、[verse]、[pre-chorus]、"
        "[chorus]、[hook]、[bridge]、[outro] 等小写结构标签，每个标签独占一行。",
        "taggedLyrics 只能包含支持的英文结构标签和真正需要唱出的歌词；禁止 [guitar solo]、"
        "[final chorus] 等自造标签，禁止用括号写演奏、制作或时长说明。独奏只能写 [solo]，"
        "最后副歌仍写 [chorus]。",
        f"除结构标签外，taggedLyrics 的歌词正文合计不得超过 {GENERATED_LYRICS_MAX_CHARS} 个字符。",
        STYLE_TAG_REQUIREMENTS,
        "只返回 JSON 对象，包含字符串 taggedLyrics 和字符串 styleTags；不要解释或输出代码块。",
    ]
)

STRUCTURED_TAG_PATTERN = re.compile(
    rf"\[\s*[A-Za-z][A-Za-z0-9 /_-]{{0,{STYLE_TAG_KEY_MAX_CHARS - 1}}}\s*:"
    rf"\s*[^\[\]\r\n]{{1,{STYLE_TAG_VALUE_MAX_CHARS}}}\s*\]"
)
LYRICS_SECTION_SPECS = {
    "intro": ("intro", False),
    "verse": ("verse", True),
    "prechorus": ("pre-chorus", True),
    "chorus": ("chorus", True),
    "hook": ("hook", False),
    "postchorus": ("post-chorus", True),
    "bridge": ("bridge", True),
    "instrumental": ("instrumental", True),
    "instrumentalbreak": ("instrumental break", True),
    "solo": ("solo", False),
    "outro": ("outro", False),
    "间奏": ("instrumental", True),
}
_CANONICAL_LYRICS_SECTIONS = dict(LYRICS_SECTION_SPECS.values())
LYRICS_SECTION_PATTERN = re.compile(
    r"\[(?:"
    + "|".join(
        rf"{re.escape(canonical)}(?: \d+)?" if numbered else re.escape(canonical)
        for canonical, numbered in _CANONICAL_LYRICS_SECTIONS.items()
    )
    + r")\]",
    re.IGNORECASE,
)
FORBIDDEN_LYRICS_CONSTRAINT_PATTERN = re.compile(
    r"(?:[,;]\s*)?\bno (?:lyrics(?:\s+or\s+melody(?:\s+generation)?)?|"
    r"vocals?|melody(?:\s+generation)?)\b",
    re.IGNORECASE,
)
NO_VOCAL_STYLE_PATTERN = re.compile(
    r"无歌词|没有歌词|纯音乐|无人声|无演唱|仅(?:使用[^，。；\n]{0,12})?哼唱|"
    r"不要人声|vocal\s*chops|instrumental|wordless|no\s+(?:lyrics|vocals?)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    structured_prompt: str
    lyrics: str
    duration_seconds: int | None
    title: str | None = None


class PromptExpander(Protocol):
    async def prepare(
        self,
        user_prompt: str,
        duration_minutes: float | None = None,
        *,
        job_id: str | None = None,
        title: str | None = None,
    ) -> PreparedPrompt: ...

    async def expand_style(self, prompt: str) -> str: ...


def split_generation_prompt(user_prompt: str) -> tuple[str, str]:
    """Split the exact section format sent by the Create page."""
    prompt = user_prompt.strip()
    lyrics_marker = "[歌词与创作内容]"
    style_marker = "[风格要求]"
    if prompt.startswith(lyrics_marker):
        parts = re.split(
            r"(?:\r\n|\n|\r)\[风格要求\](?:\r\n|\n|\r|$)",
            prompt[len(lyrics_marker) :],
            maxsplit=1,
        )
        return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
    if prompt.startswith(style_marker):
        return "", prompt[len(style_marker) :].strip()
    return "", prompt


def normalize_llm_output(raw_content: object) -> str:
    content = str(raw_content or "").strip()
    content = re.sub(r"^```[a-z]*\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```$", "", content)
    return content.strip().strip("\"'").strip()


def normalize_lyrics_section_tags(lyrics: str) -> str:
    normalized_lines: list[str] = []
    for line in lyrics.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        ending = line[len(content) :]
        stripped = content.strip()
        bracketed = re.fullmatch(r"\[[^\[\]\r\n]+\]", stripped)
        if not bracketed:
            normalized_lines.append(line)
            continue
        token = stripped[1:-1].strip()
        compact = re.sub(r"[\s_-]+", "", token).casefold()
        match = re.fullmatch(r"([^\d]+)(\d*)", compact)
        spec = LYRICS_SECTION_SPECS.get(match.group(1)) if match else None
        number = match.group(2) if match else ""
        if spec and (not number or spec[1]):
            normalized_lines.append(f"[{spec[0]}{f' {number}' if number else ''}]{ending}")
        else:
            normalized_lines.append(line)
    return "".join(normalized_lines)


def extract_json_object(raw_content: object) -> dict[str, Any]:
    """Extract the first JSON object from an otherwise chatty LLM response."""
    content = normalize_llm_output(raw_content)
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, ValueError):
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            parsed, _ = decoder.raw_decode(content[match.start() :])
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("模型未返回 JSON 对象")


def extract_tagged_lyrics(raw_content: object, original_lyrics: str) -> str | None:
    lines = [
        line.strip() for line in normalize_llm_output(raw_content).splitlines() if line.strip()
    ]
    lyric_lines = [line for line in lines if not LYRICS_SECTION_PATTERN.fullmatch(line)]
    original_lines = [line.strip() for line in original_lyrics.splitlines() if line.strip()]
    if lyric_lines != original_lines or len(lyric_lines) == len(lines):
        return None
    return "\n".join(lines)


def validate_tagged_lyrics(lyrics: str, *, strict: bool = True) -> str:
    text = normalize_lyrics_section_tags(lyrics.strip())
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    stage_direction_pattern = r"[（(].*[）)]"
    if not any(LYRICS_SECTION_PATTERN.fullmatch(line) for line in lines):
        raise ValueError("歌词缺少受支持的段落标签")
    if strict and any(
        line.startswith("[") and not LYRICS_SECTION_PATTERN.fullmatch(line) for line in lines
    ):
        raise ValueError("歌词包含不支持的段落标签")
    if strict and any(re.fullmatch(stage_direction_pattern, line) for line in lines):
        raise ValueError("歌词包含演奏或制作说明")
    if not any(
        not LYRICS_SECTION_PATTERN.fullmatch(line)
        and not (line.startswith("[") and line.endswith("]"))
        and not re.fullmatch(stage_direction_pattern, line)
        for line in lines
    ):
        raise ValueError("歌词没有可演唱内容")
    return text


def extract_structured_music_tags(raw_content: object) -> str | None:
    content = normalize_llm_output(raw_content)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
    candidates = [content]
    for marker in ("Final Answer:", "Final:", "最终答案：", "最终答案:", "Output:"):
        if marker in content:
            candidates.insert(0, content.rsplit(marker, 1)[-1].strip())

    json_candidate = _extract_json_tags(content)
    if json_candidate:
        candidates.insert(0, json_candidate)

    for candidate in candidates:
        normalized = _normalize_tags_only(candidate)
        if normalized:
            return normalized

    # Reasoning models sometimes expose their analysis in `content`, followed by
    # a final tag-only block without a stable "Final Answer" marker. Walk from the
    # bottom and accept only a contiguous block whose lines contain tags and no prose.
    tag_blocks: list[list[str]] = []
    current_block: list[str] = []
    for line in content.splitlines():
        if _normalize_tags_only(line):
            current_block.append(line)
        elif current_block:
            tag_blocks.append(current_block)
            current_block = []
    if current_block:
        tag_blocks.append(current_block)
    for block in reversed(tag_blocks):
        normalized = _normalize_tags_only("\n".join(block))
        if normalized and normalized.count("[") >= 2:
            return normalized

    trailing_tags = _extract_trailing_tag_sequence(content)
    if trailing_tags:
        return trailing_tags
    return None


def _normalize_tags_only(candidate: str) -> str | None:
    tags = STRUCTURED_TAG_PATTERN.findall(candidate)
    if not tags:
        return None
    remainder = STRUCTURED_TAG_PATTERN.sub("", candidate)
    if remainder.strip(" \t\r\n,;*_`-'\""):
        return None
    normalized_tags: list[str] = []
    for tag in tags:
        cleaned = FORBIDDEN_LYRICS_CONSTRAINT_PATTERN.sub("", re.sub(r"\s+", " ", tag)).strip()
        key, value = cleaned[1:-1].split(":", 1)
        value = value.strip(" \t,;")
        if value:
            normalized_tags.append(f"[{key.strip()}: {value}]")
    if len(normalized_tags) > STYLE_TAG_MAX:
        normalized_tags = _merge_style_tags(normalized_tags)
        if normalized_tags is None:
            return None
    return ", ".join(normalized_tags) or None


def _merge_style_tags(tags: list[str]) -> list[str] | None:
    keys = [tag[1 : tag.index(":")].strip().lower() for tag in tags]
    selected: set[int] = set()
    for aliases in REQUIRED_STYLE_CATEGORY_ALIASES:
        match = next(
            (index for index, key in enumerate(keys) if any(alias in key for alias in aliases)),
            None,
        )
        if match is not None:
            selected.add(match)
    for index in range(len(tags)):
        if len(selected) >= STYLE_TAG_MAX:
            break
        selected.add(index)

    selected_indexes = sorted(selected)[:STYLE_TAG_MAX]
    selected_tags = [tags[index] for index in selected_indexes]
    overflow = [tag[1:-1] for index, tag in enumerate(tags) if index not in selected]
    left, right = selected_tags[-2:]
    left_key = left[1 : left.index(":")].strip()
    right_key = right[1 : right.index(":")].strip()
    combined_key = f"{left_key} and {right_key}"
    combined_value = f"{left[1:-1]}; {right[1:-1]}"
    additional_value = "; ".join(overflow)
    if (
        len(combined_key) > STYLE_TAG_KEY_MAX_CHARS
        or len(combined_value) > STYLE_TAG_VALUE_MAX_CHARS
        or len(additional_value) > STYLE_TAG_VALUE_MAX_CHARS
    ):
        return None
    combined = f"[{combined_key}: {combined_value}]"
    additional = f"[Additional Directions: {additional_value}]"
    return [*selected_tags[:-2], combined, additional]


def _extract_json_tags(content: str) -> str | None:
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    tags = data.get("tags")
    if isinstance(tags, list):
        normalized_tags = []
        for tag in tags:
            text = str(tag).strip()
            if not text:
                continue
            normalized_tags.append(text if text.startswith("[") else f"[{text}]")
        return ", ".join(normalized_tags) or None
    pairs = [
        f"[{key}: {value}]"
        for key, value in data.items()
        if isinstance(key, str) and isinstance(value, (str, int, float))
    ]
    return ", ".join(pairs) if pairs else None


def _extract_trailing_tag_sequence(content: str) -> str | None:
    """Extract at least three adjacent tags when they are the final payload."""
    matches = list(STRUCTURED_TAG_PATTERN.finditer(content))
    if len(matches) < 3:
        return None
    trailing = matches[-1]
    if content[trailing.end() :].strip(" \t\r\n,;.*_`-'\""):
        return None

    selected = [trailing.group()]
    cursor = trailing.start()
    for match in reversed(matches[:-1]):
        separator = content[match.end() : cursor]
        if separator.strip(" \t\r\n,;*_`-'\""):
            break
        selected.append(match.group())
        cursor = match.start()
    if len(selected) < 3:
        return None
    selected.reverse()
    return _normalize_tags_only(", ".join(selected))


def music_prompt_validation_error(structured_prompt: str) -> str | None:
    normalized = _normalize_tags_only(structured_prompt)
    if normalized is None:
        return "模型未返回完整格式的风格标签"
    tags = STRUCTURED_TAG_PATTERN.findall(normalized)
    if not STYLE_TAG_MIN <= len(tags) <= STYLE_TAG_MAX:
        return (
            f"模型应返回 {STYLE_TAG_MIN}-{STYLE_TAG_MAX} 个完整格式的风格标签；"
            f"当前标签数 {len(tags)}"
        )

    keys = [tag[1 : tag.index(":")].strip().lower() for tag in tags]
    missing = [
        aliases[0]
        for aliases in REQUIRED_STYLE_CATEGORY_ALIASES
        if not any(alias in key for key in keys for alias in aliases)
    ]
    if missing:
        return f"模型未覆盖以下风格类别：{', '.join(missing)}；当前标签数 {len(tags)}"
    return None


def is_expanded_music_prompt(structured_prompt: str) -> bool:
    return music_prompt_validation_error(structured_prompt) is None


class OpenAICompatiblePromptExpander:
    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        style_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._style_client = style_client or client

    @staticmethod
    def _auth_headers(api_key: SecretStr | None) -> dict[str, str]:
        """空 API Key 不发送 Authorization；空 Bearer 会被 httpx 拒绝。"""
        headers = {"Content-Type": "application/json"}
        key = api_key.get_secret_value().strip() if api_key else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def expand_style(self, prompt: str) -> str:
        return await self._style_completion(STYLE_ONLY_SYSTEM_PROMPT, prompt)

    async def _style_completion(
        self,
        system: str,
        prompt: str,
    ) -> str:
        settings = self._settings
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        schema = {
            "type": "object",
            "properties": {"styleTags": {"type": "string"}},
            "required": ["styleTags"],
            "additionalProperties": False,
        }
        client = self._style_client
        url = settings.style_llm_url
        model = settings.style_llm_model
        temperature = settings.style_llm_temperature
        max_tokens = settings.style_llm_max_tokens
        timeout_seconds = settings.style_llm_timeout_seconds
        api_key = settings.style_llm_api_key
        max_attempts = settings.style_llm_max_attempts
        response_format: dict[str, Any] | None = None
        body_extra: dict[str, Any] = {}
        if settings.style_llm_inherits_main:
            # 与主 prepare 的条件保持一致：只有 Doubao/OpenAI 才发 json_object。
            if settings.llm_disable_thinking:
                if "qwen3" in model.lower():
                    body_extra["chat_template_kwargs"] = {"enable_thinking": False}
                elif "doubao" in model.lower():
                    body_extra["thinking"] = {"type": "disabled"}
            if "doubao" in model.lower() or settings.llm_url.startswith(
                "https://api.openai.com/"
            ):
                response_format = {"type": "json_object"}
        else:
            response_format = (
                {
                    "type": "json_schema",
                    "json_schema": {"name": "music_style", "strict": True, "schema": schema},
                }
                if settings.style_llm_response_format == "json_schema"
                else {"type": "json_object"}
            )
            if "qwen3" in model.lower() or settings.style_llm_enable_thinking:
                body_extra["chat_template_kwargs"] = {
                    "enable_thinking": settings.style_llm_enable_thinking
                }
        body: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
            **body_extra,
        }
        if response_format is not None:
            body["response_format"] = response_format
        headers = self._auth_headers(api_key)
        reason = ""
        for attempt in range(max_attempts):
            if attempt:
                body["messages"] = [
                    *messages,
                    {
                        "role": "user",
                        "content": f"上次响应未满足要求：{reason}。请修正，只返回 JSON。",
                    },
                ]
            try:
                response = await client.post(
                    url,
                    json=body,
                    headers=headers,
                    timeout=httpx.Timeout(
                        timeout_seconds,
                        connect=min(10, timeout_seconds),
                    ),
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                style = extract_json_object(content).get("styleTags")
                if not isinstance(style, str) or not style.strip():
                    raise ValueError("风格说明必须为非空字符串")
                # 中文模型常用全角冒号，先统一标签分隔符再校验。
                style = re.sub(
                    r"\[([^\[\]:：\r\n]+)[:：]\s*",
                    lambda match: f"[{match[1].strip()}: ",
                    style,
                )
                validate_chinese_style(style)
                style = "\n".join(
                    f"{category}：{value.strip()}"
                    for category, value in re.findall(
                        r"\[([^:\[\]\n]+):\s*([^\[\]\n]+)\]", style
                    )
                )
                if len(style) > 2000:
                    raise ValueError("风格说明不能超过 2000 个字符")
                return style
            except httpx.HTTPError as exc:
                raise GenerationError(f"风格处理失败：{_http_failure_message(exc)}") from exc
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                reason = str(exc)
        raise GenerationError(f"风格处理失败：{reason}")

    async def prepare(
        self,
        user_prompt: str,
        duration_minutes: float | None = None,
        *,
        job_id: str | None = None,
        title: str | None = None,
    ) -> PreparedPrompt:
        if self._settings.llm_api_key is None:
            raise GenerationError("歌词与风格处理失败：缺少 LLM_API_KEY 环境变量。")
        lyrics, style = split_generation_prompt(user_prompt)
        lyrics = normalize_lyrics_section_tags(lyrics)
        generate_lyrics = not lyrics
        if generate_lyrics:
            request_content = f"创作要求：\n{style or user_prompt}"
        elif not style:
            request_content = (
                f"歌词：\n{lyrics}\n\n"
                "风格要求为空，请根据歌词的主题、情绪、意象和语言节奏推断适合的音乐风格，"
                "并生成完整的 styleTags；必须保留原歌词，只能添加段落标签。"
            )
        else:
            request_content = f"歌词：\n{lyrics}\n\n风格要求：\n{style}"
        request_body: dict[str, object] = {
            "model": self._settings.llm_model,
            "temperature": 0,
            "max_tokens": self._settings.llm_max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        GENERATE_LYRICS_AND_STYLE_SYSTEM_PROMPT
                        if generate_lyrics
                        else LYRICS_AND_STYLE_SYSTEM_PROMPT
                    )
                    + ("\n" + TITLE_REQUIREMENT if title is None else ""),
                },
                {"role": "user", "content": request_content},
            ],
        }
        if self._settings.llm_disable_thinking:
            if "qwen3" in self._settings.llm_model.lower():
                request_body["chat_template_kwargs"] = {"enable_thinking": False}
            elif "doubao" in self._settings.llm_model.lower():
                request_body["thinking"] = {"type": "disabled"}
        if "doubao" in self._settings.llm_model.lower() or self._settings.llm_url.startswith(
            "https://api.openai.com/"
        ):
            request_body["response_format"] = {"type": "json_object"}
        try:
            original_messages = list(request_body["messages"])
            rejected_content = ""
            rejection_reason = ""
            diagnostics: list[dict[str, object]] = []
            for attempt in range(2):
                if attempt:
                    request_body["messages"] = [
                        *original_messages,
                        *(
                            [{"role": "assistant", "content": rejected_content[:2000]}]
                            if rejected_content
                            else []
                        ),
                        {
                            "role": "user",
                            "content": (
                                f"上次响应未满足要求：{rejection_reason}。"
                                "请修正后只输出 JSON 对象。"
                            ),
                        },
                    ]
                try:
                    diagnostics.append(
                        {
                            "attempt": attempt + 1,
                            "request": {
                                "method": "POST",
                                "url": self._settings.llm_url,
                                "timeoutSeconds": self._settings.llm_timeout_seconds,
                                "body": json.loads(json.dumps(request_body)),
                            },
                        }
                    )
                    if job_id:
                        update_job_diagnostics(
                            self._settings.output_dir, job_id, llmAttempts=diagnostics
                        )
                    response = await self._client.post(
                        self._settings.llm_url,
                        json=request_body,
                        headers=self._auth_headers(self._settings.llm_api_key),
                        timeout=httpx.Timeout(
                            self._settings.llm_timeout_seconds,
                            connect=min(10, self._settings.llm_timeout_seconds),
                        ),
                    )
                    try:
                        response_body: object = response.json()
                    except ValueError:
                        response_body = response.text
                    diagnostics[-1]["response"] = {
                        "statusCode": response.status_code,
                        "headers": dict(response.headers),
                        "body": response_body,
                    }
                    if job_id:
                        update_job_diagnostics(
                            self._settings.output_dir, job_id, llmAttempts=diagnostics
                        )
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if isinstance(content, list):
                        content = "\n".join(
                            str(part.get("text") or part.get("content") or "")
                            for part in content
                            if isinstance(part, dict)
                        )
                    rejected_content = str(content or "")
                    prepared = extract_json_object(content)
                    tagged_content = (
                        prepared.get("taggedLyrics")
                        or prepared.get("tagged_lyrics")
                        or prepared.get("lyrics")
                    )
                    if isinstance(tagged_content, list):
                        tagged_content = "\n".join(str(line) for line in tagged_content)
                    tagged_content = normalize_lyrics_section_tags(
                        normalize_llm_output(tagged_content)
                    )
                    if generate_lyrics:
                        tagged = tagged_content
                        if tagged and not LYRICS_SECTION_PATTERN.search(tagged):
                            tagged = f"[verse]\n{tagged}"
                    elif LYRICS_SECTION_PATTERN.search(lyrics):
                        tagged = lyrics
                    else:
                        tagged = extract_tagged_lyrics(tagged_content, lyrics) or (
                            f"[verse]\n{lyrics}"
                        )
                    tagged = validate_tagged_lyrics(tagged, strict=generate_lyrics)
                    if generate_lyrics:
                        lyric_chars = sum(
                            len(line.strip())
                            for line in tagged.splitlines()
                            if not LYRICS_SECTION_PATTERN.fullmatch(line.strip())
                        )
                        if lyric_chars > GENERATED_LYRICS_MAX_CHARS:
                            raise ValueError(
                                "模型生成的歌词正文超过 "
                                f"{GENERATED_LYRICS_MAX_CHARS} 个字符；当前 {lyric_chars} 个字符"
                            )
                    style_tags = (
                        prepared.get("styleTags")
                        or prepared.get("style_tags")
                        or prepared.get("structuredPrompt")
                        or prepared.get("style")
                    )
                    if isinstance(style_tags, list):
                        # Qwen often returns bare "Category: value" items without brackets.
                        style_tags = ", ".join(
                            tag if tag.startswith("[") else f"[{tag}]"
                            for tag in (str(item).strip() for item in style_tags)
                        )
                    elif isinstance(style_tags, dict):
                        style_tags = json.dumps(style_tags, ensure_ascii=False)
                    structured = extract_structured_music_tags(style_tags) or normalize_llm_output(
                        style_tags
                    )
                    validation_error = music_prompt_validation_error(structured)
                    if validation_error:
                        raise ValueError(validation_error)
                    generated_title = title
                    if generated_title is None:
                        candidate = prepared.get("title")
                        candidate = candidate.strip() if isinstance(candidate, str) else ""
                        generated_title = (
                            candidate if GENERATED_TITLE_PATTERN.fullmatch(candidate) else None
                        )
                    return PreparedPrompt(
                        structured,
                        tagged,
                        round(duration_minutes * 60) if duration_minutes is not None else None,
                        generated_title,
                    )
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    rejection_reason = str(exc)
                    diagnostics[-1]["validationError"] = rejection_reason
                    if job_id:
                        update_job_diagnostics(
                            self._settings.output_dir, job_id, llmAttempts=diagnostics
                        )
                    if attempt:
                        raise
            raise GenerationError("歌词与风格处理失败：重试后仍未获得有效结果。")
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise GenerationError(f"歌词与风格处理失败：{_http_failure_message(exc)}") from exc


def _http_failure_message(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        message = ""
        try:
            data = response.json()
            detail = data.get("detail") if isinstance(data, dict) else None
            if isinstance(detail, dict):
                message = str(detail.get("message", ""))
            elif detail:
                message = str(detail)
            if not message and isinstance(data, dict):
                nested_error = data.get("error")
                if isinstance(nested_error, dict):
                    message = str(nested_error.get("message", ""))
                elif nested_error:
                    message = str(nested_error)
                message = message or str(data.get("message") or data.get("msg") or "")
        except ValueError:
            message = response.text.strip()
        return f"HTTP {response.status_code}{f': {message}' if message else ''}"
    return str(error) or error.__class__.__name__


CHINESE_STYLE_CATEGORIES = {
    "曲风与年代": "Genre and Era",
    "节奏与拍号": "Tempo and Meter",
    "情绪": "Mood",
    "配器": "Instrumentation",
    "人声": "Vocal",
    "编排": "Arrangement",
    "制作与混音": "Production and Mix",
    "避免事项": "Negative Constraints",
}
STYLE_ONLY_SYSTEM_PROMPT = "\n".join(
    [
        "你是专业音乐制作人，只扩写用户的音乐风格，不生成歌词正文、歌名或任何演唱内容。",
        STYLE_TAG_REQUIREMENTS.replace("详细英文音乐制作标签", "详细中文音乐制作标签").replace(
            "Genre and Era、Tempo and Meter、Mood、Instrumentation、Vocal、"
            "Arrangement、Production and Mix、Negative Constraints",
            "曲风与年代、节奏与拍号、情绪、配器、人声、编排、制作与混音、避免事项",
        ),
        "用简体中文输出八个风格标签，类别依次为：" + "、".join(CHINESE_STYLE_CATEGORIES) + "。",
        "Category 和 value 均使用中文，格式为 [类别: 内容]；保留必要的 BPM、拍号、"
        "乐器与音阶/调式缩写。",
        "「节奏与拍号」标签要写清 BPM、拍号、律动/切分，并补充调式/音阶与音程走向。",
        "人声标签必须描述可演唱的普通话主唱；禁止写"
        "「无歌词演唱」「纯音乐」「无人声」「仅哼唱」「Vocal Chops」等排除歌词的描述。",
        "风格标签合计不得超过 2000 个字符。只返回 JSON 对象，唯一字段 styleTags 为字符串。",
    ]
)


def validate_chinese_style(style: str) -> None:
    translated = style
    for chinese, english in CHINESE_STYLE_CATEGORIES.items():
        translated = translated.replace(f"[{chinese}:", f"[{english}:")
    error = music_prompt_validation_error(translated)
    if error:
        raise ValueError(error)
    for category, value in re.findall(r"\[([^:\[\]\n]+):\s*([^\[\]\n]+)\]", style):
        if category not in CHINESE_STYLE_CATEGORIES or not re.search(r"[\u4e00-\u9fff]", value):
            raise ValueError("风格类别和说明必须使用中文")
        if category == "人声" and NO_VOCAL_STYLE_PATTERN.search(value):
            raise ValueError(
                "人声标签必须描述可演唱的主唱，不能写无歌词、纯音乐、无人声或仅哼唱"
            )
