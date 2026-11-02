"""写入侧抽取：把原始交互变成可检索、可冲突消解的记忆单元。

默认 RuleExtractor 是**离线确定性抽取器**，用于演示与测试；生产中可直接换成
LLMExtractor（同一 Extractor 协议），其余管线不变。规则覆盖：
  - 身份 / 职业 / 地点 / 迁移历史
  - 偏好 / 饮食禁忌 / 过敏（多值）
  - 项目事实 / 数据库栈 / 计数
  - 可复用流程（first/then/finally）
  - 中英文常见表达
"""

from __future__ import annotations

import calendar
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from .types import ExtractedMemory, MemoryKind

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

STOP_SUBJECTS = {
    "i", "my", "the", "this", "that", "it", "he", "she", "they", "we", "you",
    "actually", "also", "however", "so", "then", "now", "last", "next", "in", "on",
    "at", "a", "an", "and", "but", "or", "if", "when", "while", "there", "here",
}

STOP_ENTITIES = STOP_SUBJECTS | {
    "where", "what", "who", "whom", "whose", "when", "why", "how", "which",
    "does", "do", "did", "is", "are", "was", "were", "can", "could", "should",
    "would", "will", "tell", "give", "find", "show", "list", "please", "user",
}

PROCEDURE_MARKERS = ["first", "then", "next", "after that", "afterwards", "finally", "lastly"]

# 检索时的同义词扩展：让 "where/live" 能命中 "moved to"，"work" 能命中 occupation/employer。
PREDICATE_SYNONYMS = [
    ("location", "live lives living based city moved move address location"),
    ("occupation", "work works job role occupation position"),
    ("employer", "work employer company organization"),
    ("allergy", "allergy allergic"),
    ("preference", "preference prefer like love favorite cuisine"),
    ("favorite", "favorite prefer like love"),
    ("completed_count", "completed count total number how many"),
    ("member_count", "member count number team"),
    ("stack", "stack database uses technology built with"),
    ("uses", "uses supports technology"),
    ("supports", "supports uses technology"),
    ("name", "name called"),
    ("headquarters", "headquarters headquartered company city"),
    ("procedure", "procedure steps process how to"),
]


def predicate_synonyms(predicate: str | None) -> str:
    if not predicate:
        return ""
    key = predicate.lower()
    parts = [expansion for prefix, expansion in PREDICATE_SYNONYMS if key.startswith(prefix)]
    return " ".join(parts)

_MONTH_RE = "|".join(sorted(MONTHS.keys(), key=len, reverse=True))
_DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "ymd"),
    (re.compile(rf"\b({_MONTH_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I), "mdy"),
    (re.compile(rf"\b({_MONTH_RE})\s+(\d{{4}})\b", re.I), "my"),
]

_PROPER_RE = re.compile(r"\b([A-Z][a-zA-Z0-9'\-]+(?:\s+[A-Z][a-zA-Z0-9'\-]+)*)\b")
_QUOTED_RE = re.compile(r'"([^"\n]{2,60})"')
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _key(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^(?:a|an|the)\s+", "", value)
    value = re.sub(r"[^\w\u3400-\u9fff]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _predicate_key(value: str) -> str:
    """保留 predicate 里的层级分隔符，例如 preference:food / project:atlas.stack。"""
    value = value.strip().lower()
    value = re.sub(r"[^\w\u3400-\u9fff:.\-]+", "_", value)
    return value.strip("_")


def _clean_value(value: str) -> str:
    value = value.strip().strip(" \t\r\n.,;:!?\"'()[]")
    # 截断并列从句，避免 "Lisbon and I work as ..." 把后半句也当成地点
    value = re.split(
        r"\s+and\s+(?:(?:i|we|they|he|she)\s+)?"
        r"(?:work|works|live|lives|am|is|are|have|has|prefer|like|love|moved|completed|need|want)\b",
        value, maxsplit=1, flags=re.I,
    )[0]
    value = re.sub(r"\s+(?:now|too|as well)$", "", value, flags=re.I)
    return value.strip()


def _strip_temporal_tail(value: str) -> str:
    return re.sub(
        r"\s+(?:last|this|next|in|on|during|since|about)\s+(?:week|month|year|night|"
        r"spring|summer|autumn|fall|winter|\d{4}|january|february|march|april|may|june|"
        r"july|august|september|october|november|december)\b.*$",
        "",
        value,
        flags=re.I,
    ).strip()


_WEEKDAY_INDEX = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a few": 3, "a couple of": 2, "couple of": 2, "several": 3,
}
_WEEKDAY_RE = "|".join(_WEEKDAY_INDEX)
_AGO_RE = re.compile(
    rf"\b(?:(\d{{1,2}})|({'|'.join(sorted(_NUMBER_WORDS, key=len, reverse=True))}))\s+"
    r"(day|week|month|year)s?\s+ago\b",
    re.I,
)
_WEEKDAY_REL_RE = re.compile(rf"\b(last|this|next)\s+({_WEEKDAY_RE})(?:s)?\b", re.I)
_PERIOD_REL_RE = re.compile(r"\b(last|this|next)\s+(night|weekend|week|month|year)\b", re.I)
_SIMPLE_REL_RE = re.compile(r"\b(yesterday|today|tomorrow|tonight)\b", re.I)


_MONTH_LABELS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def human_date(stamp: float) -> str:
    """epoch → '7 May 2023'（与 LoCoMo / 人读答案一致的日期写法）。"""
    dt = datetime.fromtimestamp(stamp, timezone.utc)
    return f"{dt.day} {_MONTH_LABELS[dt.month]} {dt.year}"


def _week_start(ref: datetime, offset_days: int = 0) -> datetime:
    day = ref + timedelta(days=offset_days)
    return datetime(day.year, day.month, day.day) - timedelta(days=day.weekday())


def normalize_relative_time(text: str, event_time: float) -> str:
    """把相对时间改写为绝对日期，保留原措辞：'yesterday (7 May 2023)'。

    记忆系统能否回答 "When did X happen?" 取决于上下文里有没有可比的绝对日期；
    原始对话只写 "yesterday / last week / 3 years ago"，读取侧必须自己换算，
    极易出错。这里在写入时一次性换算，并在括号内保留原词以免丢失说话人语气。
    """
    if not text:
        return text
    try:
        ref = datetime.fromtimestamp(event_time, timezone.utc)
    except (ValueError, OSError, OverflowError):
        return text
    ref_day = datetime(ref.year, ref.month, ref.day)

    def repl_simple(m: re.Match[str]) -> str:
        word = m.group(1).lower()
        delta = {"yesterday": -1, "today": 0, "tonight": 0, "tomorrow": 1}[word]
        return f"{m.group(0)} ({human_date(calendar.timegm((ref_day + timedelta(days=delta)).timetuple()))})"

    def repl_ago(m: re.Match[str]) -> str:
        raw = m.group(0)
        num_text = (m.group(1) or m.group(2) or "1").lower()
        count = int(num_text) if num_text.isdigit() else _NUMBER_WORDS.get(num_text, 1)
        unit = m.group(3).lower()
        days = count * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        return f"{raw} ({human_date(calendar.timegm((ref_day - timedelta(days=days)).timetuple()))})"

    def repl_weekday(m: re.Match[str]) -> str:
        raw, direction, weekday = m.group(0), m.group(1).lower(), m.group(2).lower()
        target = _WEEKDAY_INDEX[weekday]
        gap = (ref_day.weekday() - target) % 7
        if direction == "last":
            delta = -(gap or 7)
        elif direction == "this":
            delta = -gap
        else:
            delta = 7 - gap if gap else 7
        return f"{raw} ({human_date(calendar.timegm((ref_day + timedelta(days=delta)).timetuple()))})"

    def repl_period(m: re.Match[str]) -> str:
        raw, direction, unit = m.group(0), m.group(1).lower(), m.group(2).lower()
        if unit == "night":
            delta = -1 if direction == "last" else (0 if direction == "this" else 1)
            return f"{raw} ({human_date(calendar.timegm((ref_day + timedelta(days=delta)).timetuple()))})"
        if unit == "weekend":
            gap = (ref_day.weekday() - 5) % 7  # 最近的周六
            base = ref_day + timedelta(days=-gap)
            base += timedelta(days={"last": -7, "this": 0, "next": 7}[direction])
            return f"{raw} (the weekend of {human_date(calendar.timegm(base.timetuple()))})"
        if unit == "week":
            offset = {"last": -7, "this": 0, "next": 7}[direction]
            start = _week_start(ref, offset)
            return f"{raw} (the week of {human_date(calendar.timegm(start.timetuple()))})"
        offset = {"last": -1, "this": 0, "next": 1}[direction]
        if unit == "month":
            month = ref.month + offset
            year = ref.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            return f"{raw} ({_MONTH_LABELS[month]} {year})"
        year = ref.year + offset
        return f"{raw} ({year})"

    out = _AGO_RE.sub(repl_ago, text)
    out = _WEEKDAY_REL_RE.sub(repl_weekday, out)
    out = _PERIOD_REL_RE.sub(repl_period, out)
    out = _SIMPLE_REL_RE.sub(repl_simple, out)
    return out


def parse_date(text: str, reference: float) -> float | None:
    """从文本解析绝对/相对日期，返回 UTC epoch；解析失败返回 None。"""
    for pattern, kind in _DATE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        try:
            if kind == "ymd":
                dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            elif kind == "mdy":
                dt = datetime(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
            else:
                dt = datetime(int(m.group(2)), MONTHS[m.group(1).lower()], 1)
            return float(calendar.timegm(dt.timetuple()))
        except (ValueError, KeyError):
            continue
    lower = text.lower()
    ref = datetime.fromtimestamp(reference, timezone.utc)
    if "yesterday" in lower:
        return float(calendar.timegm((ref - timedelta(days=1)).timetuple()))
    if "today" in lower:
        return float(calendar.timegm(ref.timetuple()))
    if "last week" in lower:
        return float(calendar.timegm((ref - timedelta(days=7)).timetuple()))
    if "last month" in lower:
        return float(calendar.timegm((ref - timedelta(days=30)).timetuple()))
    if "last year" in lower:
        return float(calendar.timegm((ref - timedelta(days=365)).timetuple()))
    m = re.search(r"(\d+)\s+(day|week|month|year)s?\s+ago", lower)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        return float(calendar.timegm((ref - timedelta(days=days)).timetuple()))
    return None


def extract_entities(text: str) -> list[str]:
    """抽取命名实体：大写词组、引号短语；中文槽位值由调用方补充。"""
    found: list[str] = []
    for m in _PROPER_RE.finditer(text):
        span = m.group(1).strip()
        if not span or span.lower() in STOP_ENTITIES:
            continue
        if len(span) == 1:
            continue
        found.append(span)
    for m in _QUOTED_RE.finditer(text):
        found.append(m.group(1).strip())
    # 保序去重（忽略大小写）
    seen: set[str] = set()
    out: list[str] = []
    for item in found:
        key = item.lower()
        if key not in seen and len(item) >= 2:
            seen.add(key)
            out.append(item)
    return out


def _head_topic(phrase: str) -> str:
    """取偏好短语的中心词，作为 predicate 的一部分（避免不同偏好互相覆盖）。"""
    lower = _key(phrase)
    if not lower:
        return "general"
    if re.search(r"\b(food|cuisine|meal|dish)\b", lower):
        return "food"
    if re.search(r"\b(seat|seating)\b", lower):
        return "seat"
    if re.search(r"\b(coffee|tea|drink|beverage)\b", lower):
        return "drink"
    if re.search(r"\b(book|novel|author)\b", lower):
        return "book"
    if re.search(r"\b(music|band|song|artist)\b", lower):
        return "music"
    if re.search(r"\b(movie|film|show|series)\b", lower):
        return "movie"
    if re.search(r"\b(sport|team|game)\b", lower):
        return "sport"
    words = [w for w in lower.split() if len(w) > 2]
    if words:
        return words[-1]
    cjk = re.findall(r"[\u4e00-\u9fff]{2,}", lower)
    return cjk[-1] if cjk else "general"


class RuleExtractor:
    """规则化抽取器。输出统一 ExtractedMemory，供写入管线做冲突消解。"""

    name = "rule"

    def extract(
        self,
        text: str,
        event_time: float,
        context: dict[str, Any] | None = None,
    ) -> list[ExtractedMemory]:
        out: list[ExtractedMemory] = []
        seen: set[tuple[str, str, str, str]] = set()
        self._extract_procedures(text, event_time, out, seen)
        for sentence in split_sentences(text):
            self._extract_sentence(sentence, text, event_time, out, seen)
        return out

    # ----------------------------------------------------------------------------------
    def _add(
        self,
        out: list[ExtractedMemory],
        seen: set[tuple[str, str, str, str]],
        kind: MemoryKind,
        text: str,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        entities: Sequence[str] | None = None,
        importance: float = 0.6,
        confidence: float = 0.85,
        event_time: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        subject_key = ""
        predicate_key = ""
        object_key = ""
        if subject and predicate:
            subject_key = "user" if _key(subject) == "user" else _key(subject).replace(" ", "-")
            if subject_key.startswith("project-"):
                subject_key = "project:" + subject_key[len("project-"):]
            predicate_key = _predicate_key(predicate)
        if obj is not None:
            object_key = _key(obj)
        marker = (kind.value, predicate_key, object_key, str(data.get("polarity", "")) if data else "")
        if marker in seen:
            return
        seen.add(marker)

        merged_entities = list(entities or [])
        for candidate in [subject, obj]:
            if not candidate:
                continue
            cleaned = _clean_value(str(candidate))
            if 1 < len(cleaned) <= 40 and not re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
                if _key(cleaned) not in {"user", "i", "me", "my"}:
                    merged_entities.append(cleaned)
        seen_entities: set[str] = set()
        final_entities = []
        for ent in merged_entities:
            key = ent.lower()
            if key not in seen_entities:
                seen_entities.add(key)
                final_entities.append(ent)

        payload = dict(data or {})
        payload.update(
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "subject_key": subject_key,
                "predicate_key": predicate_key,
                "object_key": object_key,
                "entities": final_entities,
            }
        )
        out.append(
            ExtractedMemory(
                kind=kind,
                text=text.strip(),
                subject=subject,
                predicate=predicate,
                object=obj,
                entities=final_entities,
                importance=importance,
                confidence=confidence,
                event_time=event_time,
                data=payload,
            )
        )

    # ----------------------------------------------------------------------------------
    def _extract_procedures(
        self,
        text: str,
        event_time: float,
        out: list[ExtractedMemory],
        seen: set[tuple[str, str, str, str]],
    ) -> None:
        # 1) 单句/段落内的 first/then/finally
        for sentence in split_sentences(text):
            lower = sentence.lower()
            hits = [m for m in PROCEDURE_MARKERS if re.search(rf"\b{re.escape(m)}\b", lower)]
            if len(hits) < 2:
                continue
            marker_re = r"(?:first|then|next|after that|afterwards|finally|lastly)"
            pieces = re.split(rf"\b{marker_re}\b[,:]?\s*", sentence, flags=re.I)
            # 去掉第一个 marker 之前的目的状语，只保留真正的步骤
            steps = [_clean_value(p) for p in pieces[1:] if len(_clean_value(p)) > 3]
            steps = [re.sub(r"[,\s]+and$", "", step).strip() for step in steps]
            steps = [step for step in steps if step]
            if len(steps) < 2:
                continue
            prefix = re.split(rf"\b{marker_re}\b", sentence, maxsplit=1, flags=re.I)[0]
            intent = re.sub(r"^(?:to|how to|when you want to|if you want to)\s+", "", prefix.strip(), flags=re.I)
            intent = _clean_value(intent)[:120] or "procedure"
            self._add(
                out, seen, MemoryKind.PROCEDURE, sentence,
                subject="user", predicate=f"procedure:{_key(intent)[:40]}", obj=intent,
                entities=extract_entities(sentence),
                importance=0.75, confidence=0.85, event_time=event_time,
                data={"intent": intent, "steps": steps},
            )

        # 2) 编号/项目符号列表
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        list_items = [
            re.sub(r"^(?:\d+[.)]|[-*])\s*", "", ln)
            for ln in lines
            if re.match(r"^(?:\d+[.)]|[-*])\s+", ln)
        ]
        if len(list_items) >= 2:
            intent = ""
            for ln in lines:
                if re.search(r"(steps?|how to|procedure|流程|步骤)", ln, re.I) and ln not in list_items:
                    intent = _clean_value(re.sub(r"[:：].*$", "", ln))
                    break
            intent = intent or "procedure"
            self._add(
                out, seen, MemoryKind.PROCEDURE, " ".join(list_items),
                subject="user", predicate=f"procedure:{_key(intent)[:40]}", obj=intent,
                entities=extract_entities(text),
                importance=0.75, confidence=0.8, event_time=event_time,
                data={"intent": intent, "steps": list_items},
            )

    # ----------------------------------------------------------------------------------
    def _extract_sentence(
        self,
        sentence: str,
        full_text: str,
        event_time: float,
        out: list[ExtractedMemory],
        seen: set[tuple[str, str, str, str]],
    ) -> None:
        lower = sentence.lower()
        sentence_time = parse_date(sentence, event_time) or event_time
        entities = extract_entities(sentence)

        # ---------------- 中文规则 ----------------
        if _CJK_RE.search(sentence):
            self._extract_chinese(sentence, sentence_time, out, seen)

        # ---------------- 姓名 ----------------
        m = re.search(r"\bmy name is\s+([A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)?)", sentence, re.I)
        if not m:
            m = re.search(
                r"\bi(?:'m| am)\s+([A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)?)(?=[,.]|$)",
                sentence, re.I,
            )
        if m:
            value = _clean_value(m.group(1))
            if value.lower() not in {"allergic", "based", "looking", "not", "currently"}:
                self._add(out, seen, MemoryKind.FACT, sentence, "user", "name", value,
                          entities, 0.9, 0.9, sentence_time)

        # ---------------- 职业 / 雇主 ----------------
        for pattern, predicate, importance in [
            (r"\bi(?: now)? work as (?:a|an)\s+([^.,;!?]+)", "occupation", 0.88),
            (r"\bmy (?:new )?(?:job|role|title) is\s+([^.,;!?]+)", "occupation", 0.9),
            (r"\bi(?: now)? work (?:at|for)\s+([^.,;!?]+)", "employer", 0.85),
        ]:
            m = re.search(pattern, sentence, re.I)
            if m:
                value = _clean_value(_strip_temporal_tail(m.group(1)))
                if value and predicate == "occupation":
                    value = _clean_value(re.split(r"\s+(?:at|for)\s+", value, maxsplit=1)[0])
                if value:
                    is_change = bool(re.search(r"\b(now|new)\b", sentence, re.I))
                    self._add(out, seen, MemoryKind.FACT, sentence, "user", predicate, value,
                              entities, importance, 0.9, sentence_time,
                              {"is_change": True} if is_change else None)
        m = re.search(
            r"\bi(?: now)? work as (?:a|an)\s+[^.,;!?]+?\s+(?:at|for)\s+([^.,;!?]+)",
            sentence, re.I,
        )
        if m:
            value = _clean_value(_strip_temporal_tail(m.group(1)))
            if value:
                is_change = bool(re.search(r"\b(now|new)\b", sentence, re.I))
                self._add(out, seen, MemoryKind.FACT, sentence, "user", "employer", value,
                          entities, 0.85, 0.9, sentence_time,
                          {"is_change": True} if is_change else None)

        # ---------------- 地点 / 迁移 ----------------
        m = re.search(r"\bi used to live in\s+([^.,;!?]+)", sentence, re.I)
        if m:
            value = _clean_value(_strip_temporal_tail(m.group(1)))
            self._add(out, seen, MemoryKind.FACT, sentence, "user", "previous_location", value,
                      entities, 0.8, 0.85, sentence_time, {"historical": True})
        else:
            patterns = [
                r"\bi(?: now)? live in\s+([^.,;!?]+)",
                r"\bi(?:'m| am) (?:now )?(?:living|based) in\s+([^.,;!?]+)",
                r"\bi(?:'ve| have)? (?:moved|relocated) to\s+([^.,;!?]+)",
                r"\bi now live in\s+([^.,;!?]+)",
            ]
            for pattern in patterns:
                m = re.search(pattern, sentence, re.I)
                if m:
                    value = _clean_value(_strip_temporal_tail(m.group(1)))
                    if value:
                        is_change = bool(re.search(r"\b(moved|relocated|now live|have moved)\b", sentence, re.I))
                        self._add(out, seen, MemoryKind.FACT, sentence, "user", "location", value,
                                  entities, 0.9, 0.92, sentence_time,
                                  {"is_change": True} if is_change else None)
                    break

        # ---------------- 过敏 / 饮食 ----------------
        m = re.search(r"\bi(?:'m| am) allergic to\s+([^.,;!?]+)", sentence, re.I)
        if m:
            raw = _clean_value(m.group(1))
            parts = re.split(r",|\band\b", raw, flags=re.I)
            for part in parts:
                value = _clean_value(part)
                if value:
                    self._add(out, seen, MemoryKind.FACT, sentence, "user", "allergy", value,
                              entities, 0.95, 0.92, sentence_time, {"multi": True})
        m = re.search(r"\bi(?:'m| am) (vegetarian|vegan|pescatarian|gluten[- ]free|dairy[- ]free)", sentence, re.I)
        if m:
            self._add(out, seen, MemoryKind.FACT, sentence, "user", "diet", _clean_value(m.group(1)),
                      entities, 0.9, 0.9, sentence_time)
        m = re.search(r"\bi (?:don't|do not) eat\s+([^.,;!?]+)", sentence, re.I)
        if m:
            value = _clean_value(m.group(1))
            self._add(out, seen, MemoryKind.FACT, sentence, "user", "diet_restriction", value,
                      entities, 0.9, 0.9, sentence_time, {"multi": True})

        # ---------------- 偏好 ----------------
        m = re.search(r"\bmy favorite\s+([a-z ]{2,20}?)\s+is\s+([^.,;!?]+)", sentence, re.I)
        if m:
            topic = _head_topic(m.group(1))
            value = _clean_value(m.group(2))
            self._add(out, seen, MemoryKind.PREFERENCE, sentence, "user", f"favorite:{topic}", value,
                      entities, 0.85, 0.9, sentence_time, {"polarity": "positive", "topic": topic})
        m = re.search(
            r"\bi\s+(?:now\s+)?(?:really\s+)?(?:like|love|enjoy|prefer|hate|dislike|avoid)\s+([^.,;!?]+)",
            sentence, re.I,
        )
        if m:
            verb = re.search(r"\b(like|love|enjoy|prefer|hate|dislike|avoid)\b", lower)
            polarity = "negative" if verb and verb.group(1) in {"hate", "dislike", "avoid"} else "positive"
            value = _clean_value(re.split(r"\s+to\s+", m.group(1), maxsplit=1)[0])
            topic = _head_topic(value)
            is_change = bool(re.search(r"\bnow\b", sentence, re.I))
            self._add(out, seen, MemoryKind.PREFERENCE, sentence, "user", f"preference:{topic}", value,
                      entities, 0.82, 0.88, sentence_time,
                      {"polarity": polarity, "topic": topic, "is_change": is_change})
        # 条件式 / 推荐类偏好：I'd like、I would prefer、I'd rather、I want ...
        m = re.search(
            r"\bi(?:'d| would)?\s+(?:really\s+)?(?:prefer|like|love|rather|want)\s+"
            r"(?:it if|to|when|seeing|having)?\s*([^.,;!?]{2,80})",
            sentence, re.I,
        )
        if m and not re.match(r"\bi\s+(?:don't|do not)\b", sentence, re.I):
            value = _clean_value(m.group(1))
            if value and len(value.split()) >= 2:
                topic = _head_topic(value)
                self._add(out, seen, MemoryKind.PREFERENCE, sentence, "user", f"preference:{topic}", value,
                          entities, 0.78, 0.85, sentence_time,
                          {"polarity": "positive", "topic": topic})
        # 习惯类偏好：I usually / always / typically ...
        m = re.search(
            r"\bi\s+(?:usually|always|typically|generally|normally)\s+([^.,;!?]{2,80})",
            sentence, re.I,
        )
        if m:
            value = _clean_value(m.group(1))
            if value and len(value.split()) >= 2:
                topic = _head_topic(value)
                self._add(out, seen, MemoryKind.PREFERENCE, sentence, "user", f"habit:{topic}", value,
                          entities, 0.72, 0.82, sentence_time,
                          {"polarity": "positive", "topic": topic})
        # 身份式偏好：I'm a big fan of X / I'm into X / I'm not a fan of X
        m = re.search(
            r"\bi(?:'m| am)\s+(?:(a (?:big )?fan of)|into|(?:not a fan of))\s+([^.,;!?]{2,80})",
            sentence, re.I,
        )
        if m:
            value = _clean_value(m.group(2))
            if value:
                polarity = "negative" if "not a fan" in m.group(0).lower() else "positive"
                topic = _head_topic(value)
                self._add(out, seen, MemoryKind.PREFERENCE, sentence, "user", f"preference:{topic}", value,
                          entities, 0.8, 0.85, sentence_time,
                          {"polarity": polarity, "topic": topic})

        # ---------------- 计数（多值） ----------------
        m = re.search(
            r"\bi (?:have )?(?:completed|finished|took)\s+(\d+)\s+([a-z ]+?)"
            r"(?:\s+on\s+([A-Za-z][\w ]+?))?(?=[.,;!?]|$)",
            sentence, re.I,
        )
        if m:
            number = m.group(1)
            what = _key(m.group(2)) or "items"
            platform = _key(m.group(3)) if m.group(3) else "general"
            self._add(out, seen, MemoryKind.FACT, sentence, "user",
                      f"completed_count:{what}:{platform}", number, entities,
                      0.75, 0.9, sentence_time,
                      {"number": int(number), "unit": what, "platform": platform})

        # ---------------- 项目 / 组织事实 ----------------
        m = re.search(
            r"\b([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*)\s+(?:is|was)\s+headquartered in\s+([^.,;!?]+)",
            sentence,
        )
        if m:
            subject = _clean_value(m.group(1))
            if subject.lower() not in STOP_SUBJECTS:
                self._add(out, seen, MemoryKind.FACT, sentence, subject, "headquarters",
                          _clean_value(m.group(2)), entities, 0.78, 0.88, sentence_time)

        project_patterns = [
            (r"\b(Project\s+[A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)?)\s+(?:now\s+)?(?:uses|runs on|is built with|migrated to)\s+([^.,;!?]+)",
             "stack", 0.82),
            (r"\b(Project\s+[A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)?)\s+(?:is|was)\s+(?:led by|managed by)\s+([^.,;!?]+)",
             "lead", 0.8),
            (r"\b(Project\s+[A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)?)\s+(?:has|includes)\s+(\d+)\s+([^.,;!?]+)",
             "member_count", 0.7),
        ]
        for pattern, predicate, importance in project_patterns:
            m = re.search(pattern, sentence)
            if not m:
                continue
            subject = _clean_value(m.group(1))
            if predicate == "member_count":
                value = f"{m.group(2)} {_clean_value(m.group(3))}"
            else:
                value = _clean_value(_strip_temporal_tail(m.group(2)))
            if value:
                self._add(out, seen, MemoryKind.FACT, sentence, subject, predicate, value,
                          entities, importance, 0.88, sentence_time)

        # ---------------- 通用英文事实（受限） ----------------
        m = re.search(
            r"\b([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*)\s+"
            r"(uses|requires|supports|costs|takes|contains|includes|won|completed|started|finished)\s+"
            r"([^.,;!?]+)",
            sentence,
        )
        if m and m.group(1).lower() not in STOP_SUBJECTS:
            subject = _clean_value(m.group(1))
            if subject.lower() not in {"project", "the"}:
                self._add(out, seen, MemoryKind.FACT, sentence, subject, m.group(2).lower(),
                          _clean_value(m.group(3)), entities, 0.62, 0.72, sentence_time)

        # ---------------- “X is Y” 通用事实（受限） ----------------
        m = re.search(
            r"\b([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*)\s+(?:is|was)\s+(?:a|an|the)?\s*([^.,;!?]{3,80})",
            sentence,
        )
        if m and m.group(1).lower() not in STOP_SUBJECTS:
            subject = _clean_value(m.group(1))
            value = _clean_value(m.group(2))
            if subject.lower() not in {"project", "the", "it", "this", "that"} and not value.lower().startswith("not "):
                self._add(out, seen, MemoryKind.FACT, sentence, subject, "is", value,
                          entities, 0.55, 0.65, sentence_time)

    # ----------------------------------------------------------------------------------
    def _extract_chinese(
        self,
        sentence: str,
        event_time: float,
        out: list[ExtractedMemory],
        seen: set[tuple[str, str, str, str]],
    ) -> None:
        entities = extract_entities(sentence)
        patterns = [
            (r"我叫\s*([\u4e00-\u9fffA-Za-z]{2,10})", "name", 0.9),
            (r"我(?:现在)?(?:住在|居住在|搬到了|搬家到)\s*([^。，；！？]+)", "location", 0.9),
            (r"我(?:的)?(?:工作|职业|岗位)是\s*([^。，；！？]+)", "occupation", 0.85),
            (r"我对\s*([^。，；！？]+?)\s*过敏", "allergy", 0.95),
        ]
        for pattern, predicate, importance in patterns:
            m = re.search(pattern, sentence)
            if m:
                value = _clean_value(m.group(1))
                if value:
                    data = {"multi": True} if predicate == "allergy" else None
                    self._add(out, seen, MemoryKind.FACT, sentence, "user", predicate, value,
                              entities, importance, 0.9, event_time, data)

        m = re.search(r"我的\s*([^。，；！？]{1,10}?)\s*是\s*([^。，；！？]+)", sentence)
        if m:
            topic = _head_topic(m.group(1))
            self._add(out, seen, MemoryKind.PREFERENCE, sentence, "user", f"favorite:{topic}",
                      _clean_value(m.group(2)), entities, 0.85, 0.88, event_time,
                      {"polarity": "positive", "topic": topic})
        m = re.search(r"我(?:现在)?(?:喜欢|偏爱|爱好|讨厌|不喜欢)\s*([^。，；！？]+)", sentence)
        if m:
            polarity = "negative" if re.search(r"讨厌|不喜欢", m.group(0)) else "positive"
            value = _clean_value(m.group(1))
            topic = _head_topic(value)
            self._add(out, seen, MemoryKind.PREFERENCE, sentence, "user", f"preference:{topic}", value,
                      entities, 0.82, 0.88, event_time, {"polarity": polarity, "topic": topic})

        m = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{2,20})(?:项目)?(?:现在)?(?:使用|采用|支持|需要|包含)\s*([^。，；！？]+)", sentence)
        if m and m.group(1) not in {"我", "我们"}:
            subject = _clean_value(m.group(1))
            value = _clean_value(m.group(2))
            predicate = "uses" if re.search(r"使用|采用", m.group(0)) else "supports"
            self._add(out, seen, MemoryKind.FACT, sentence, subject, predicate, value,
                      entities, 0.65, 0.78, event_time)


class LLMExtractor:
    """用 LLM 做抽取的可插拔实现；解析失败自动回退到 RuleExtractor。"""

    name = "llm"

    PROMPT = """你是记忆抽取器。把输入对话抽成 JSON 数组，每项包含：
kind (fact|preference|procedure), text (一句完整原话), subject, predicate, object,
entities (数组), importance (0-1), confidence (0-1), event_time (ISO8601 或 null),
data (可选对象)。要求：
1) 只抽取对未来对话有用、可验证的信息；不要臆造。
2) 用户偏好用 preference，流程用 procedure，其余事实用 fact。
3) 同一事实不要重复。只输出 JSON，不要解释。
对话：
{text}
"""

    def __init__(self, client: Any, fallback: RuleExtractor | None = None) -> None:
        self.client = client
        self.fallback = fallback or RuleExtractor()

    def extract(self, text: str, event_time: float, context: dict[str, Any] | None = None) -> list[ExtractedMemory]:
        try:
            raw = self.client.complete(self.PROMPT.format(text=text))
            raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
            rows = json.loads(raw)
            out: list[ExtractedMemory] = []
            for row in rows:
                kind = MemoryKind(str(row.get("kind", "fact")))
                stamp = parse_date(str(row.get("event_time") or ""), event_time) or event_time
                out.append(
                    ExtractedMemory(
                        kind=kind,
                        text=str(row.get("text") or ""),
                        subject=row.get("subject"),
                        predicate=row.get("predicate"),
                        object=row.get("object"),
                        entities=[str(e) for e in (row.get("entities") or [])],
                        importance=float(row.get("importance", 0.6)),
                        confidence=float(row.get("confidence", 0.8)),
                        event_time=stamp,
                        data=dict(row.get("data") or {}),
                    )
                )
            return [item for item in out if item.text]
        except Exception:
            return self.fallback.extract(text, event_time, context)


class RuleSummarizer:
    """抽取式摘要：优先保留含数字、实体、偏好的句子。"""

    def summarize(self, texts: Sequence[str], max_chars: int = 600) -> str:
        sentences: list[str] = []
        for text in texts:
            sentences.extend(split_sentences(text))
        if not sentences:
            return ""
        scored: list[tuple[float, int, str]] = []
        for idx, sent in enumerate(sentences):
            score = 1.0
            if re.search(r"\d", sent):
                score += 0.8
            if extract_entities(sent):
                score += 0.6
            if re.search(r"\b(prefer|like|love|allergic|moved|live|work|project)\b", sent, re.I):
                score += 0.5
            if idx == 0:
                score += 0.4
            score -= idx * 0.01
            scored.append((score, idx, sent.strip()))
        scored.sort(reverse=True)
        chosen: list[tuple[int, str]] = []
        total = 0
        for _, idx, sent in scored:
            if total + len(sent) > max_chars and chosen:
                continue
            chosen.append((idx, sent))
            total += len(sent)
            if total >= max_chars:
                break
        chosen.sort()
        return " ".join(s for _, s in chosen)[:max_chars]


class LLMSummarizer:
    def __init__(self, client: Any, fallback: RuleSummarizer | None = None) -> None:
        self.client = client
        self.fallback = fallback or RuleSummarizer()

    def summarize(self, texts: Sequence[str], max_chars: int = 600) -> str:
        prompt = (
            "用一段话总结下面的对话，保留人物、地点、时间、数字、偏好、承诺和未完成事项。"
            f"不超过 {max_chars} 字。\n\n" + "\n".join(texts)
        )
        try:
            return self.client.complete(prompt).strip()[:max_chars]
        except Exception:
            return self.fallback.summarize(texts, max_chars)
