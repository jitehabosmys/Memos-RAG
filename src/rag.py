import os
import sys
import math
import re
import calendar
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# 加载环境变量 (.env)
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from sentence_transformers import CrossEncoder

from etl import fetch_all_memos, process_documents

# 配置路径
PERSIST_DIRECTORY = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL_NAME = os.getenv("RAG_RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
DENSE_TOP_K = int(os.getenv("RAG_DENSE_TOP_K", "12"))
BM25_TOP_K = int(os.getenv("RAG_BM25_TOP_K", "12"))
FUSION_TOP_K = int(os.getenv("RAG_FUSION_TOP_K", "15"))
FINAL_TOP_K = int(os.getenv("RAG_FINAL_TOP_K", "5"))
QUERY_REWRITE_COUNT = int(os.getenv("RAG_QUERY_REWRITE_COUNT", "2"))
MULTI_QUERY_DENSE_TOP_K = int(os.getenv("RAG_MULTI_QUERY_DENSE_TOP_K", "8"))
MULTI_QUERY_BM25_TOP_K = int(os.getenv("RAG_MULTI_QUERY_BM25_TOP_K", "8"))
MULTI_QUERY_FUSION_TOP_K = int(os.getenv("RAG_MULTI_QUERY_FUSION_TOP_K", "8"))
MULTI_QUERY_GLOBAL_TOP_K = int(os.getenv("RAG_MULTI_QUERY_GLOBAL_TOP_K", "24"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
RERANK_BATCH_SIZE = int(os.getenv("RAG_RERANK_BATCH_SIZE", "8"))
TIMEZONE_NAME = os.getenv("RAG_TIMEZONE", "Asia/Shanghai")
TIME_RECENCY_BOOST_WEIGHT = float(os.getenv("RAG_TIME_RECENCY_BOOST_WEIGHT", "0.008"))
TIME_RECENCY_HALF_LIFE_DAYS = float(os.getenv("RAG_TIME_RECENCY_HALF_LIFE_DAYS", "45"))
TIME_SORT_CANDIDATE_POOL = int(os.getenv("RAG_TIME_SORT_CANDIDATE_POOL", "10"))
TIME_INTENT_MODEL_NAME = os.getenv("RAG_TIME_INTENT_MODEL_NAME", os.getenv("OPENAI_MODEL_NAME", "glm-4.6"))


def env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


USE_RERANK = env_flag("RAG_USE_RERANK", True)
USE_QUERY_REWRITE = env_flag("RAG_USE_QUERY_REWRITE", False)
USE_TIME_AWARE_RETRIEVAL = env_flag("RAG_USE_TIME_AWARE_RETRIEVAL", True)
USE_LLM_TIME_PARSER = env_flag("RAG_USE_LLM_TIME_PARSER", True)

try:
    LOCAL_TIMEZONE = ZoneInfo(TIMEZONE_NAME)
except Exception:
    LOCAL_TIMEZONE = None

_TIME_INTENT_CHAIN = None


@dataclass
class TimeIntent:
    start_ts: int | None = None
    end_ts: int | None = None
    boost_recent: bool = False
    sort_direction: str | None = None
    semantic_query: str | None = None
    matched_phrases: list[str] = field(default_factory=list)
    reason: str = "none"
    parser_source: str = "none"
    retrieval_strategy: str = "semantic_first"

    @property
    def has_hard_filter(self) -> bool:
        return self.start_ts is not None or self.end_ts is not None

    @property
    def is_active(self) -> bool:
        return self.has_hard_filter or self.boost_recent or self.sort_direction is not None


def tokenize_for_bm25(text: str) -> list[str]:
    """将中英文混合文本切分为适合 BM25 的 token。"""
    return re.findall(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]", text.lower())


def get_chunk_id(doc) -> str:
    """统一获取检索文档的稳定 ID。"""
    return doc.id or doc.metadata.get("chunk_id", "")


def get_created_ts(doc) -> int | None:
    created_ts = doc.metadata.get("created_ts")
    if created_ts is None:
        return None
    try:
        return int(created_ts)
    except (TypeError, ValueError):
        return None


def get_now() -> datetime:
    if LOCAL_TIMEZONE is not None:
        return datetime.now(LOCAL_TIMEZONE)
    return datetime.now()


def to_unix_ts(dt: datetime) -> int:
    return int(dt.timestamp())


def make_day_bounds(reference: datetime) -> tuple[int, int]:
    start = datetime.combine(reference.date(), time.min, tzinfo=reference.tzinfo)
    end = datetime.combine(reference.date(), time.max, tzinfo=reference.tzinfo)
    return to_unix_ts(start), to_unix_ts(end)


def make_month_bounds(year: int, month: int, tzinfo) -> tuple[int, int]:
    last_day = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1, tzinfo=tzinfo)
    end = datetime(year, month, last_day, 23, 59, 59, 999999, tzinfo=tzinfo)
    return to_unix_ts(start), to_unix_ts(end)


def make_year_bounds(year: int, tzinfo) -> tuple[int, int]:
    start = datetime(year, 1, 1, tzinfo=tzinfo)
    end = datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=tzinfo)
    return to_unix_ts(start), to_unix_ts(end)


def make_week_bounds(reference: datetime) -> tuple[int, int]:
    start_of_week = reference - timedelta(days=reference.weekday())
    start = datetime.combine(start_of_week.date(), time.min, tzinfo=reference.tzinfo)
    end = start + timedelta(days=7) - timedelta(microseconds=1)
    return to_unix_ts(start), to_unix_ts(end)


def format_ts(ts: int | None) -> str | None:
    if ts is None:
        return None
    if LOCAL_TIMEZONE is not None:
        return datetime.fromtimestamp(ts, LOCAL_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def get_chunk_index(doc) -> int:
    chunk_id = get_chunk_id(doc)
    match = re.search(r"_(\d+)$", chunk_id)
    if not match:
        return 0
    return int(match.group(1))


def add_matched_phrase(intent: TimeIntent, phrase: str | None):
    if phrase and phrase not in intent.matched_phrases:
        intent.matched_phrases.append(phrase)


def normalize_semantic_query(query: str | None) -> str | None:
    if not query or not isinstance(query, str):
        return None
    normalized = re.sub(r"\s+", " ", query).strip()
    normalized = re.sub(r"[，。！？?、,]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    meaningful_tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", normalized)
    if len(meaningful_tokens) < 2:
        return None
    return normalized


def parse_small_number(raw: str) -> int | None:
    mapping = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    return mapping.get(raw)


def set_time_range(intent: TimeIntent, start_ts: int, end_ts: int, phrase: str, reason: str):
    intent.start_ts = start_ts
    intent.end_ts = end_ts
    intent.reason = reason
    add_matched_phrase(intent, phrase)


def build_time_intent_from_payload(payload: dict, now: datetime | None = None) -> TimeIntent:
    now = now or get_now()
    intent = TimeIntent(parser_source="llm")

    if not isinstance(payload, dict):
        return intent

    matched_phrases = payload.get("matched_phrases") or []
    if isinstance(matched_phrases, str):
        matched_phrases = [matched_phrases]
    for phrase in matched_phrases:
        if isinstance(phrase, str):
            add_matched_phrase(intent, phrase.strip())

    sort_direction = str(payload.get("sort_direction") or "none").strip().lower()
    if sort_direction in {"oldest", "recent"}:
        intent.sort_direction = sort_direction

    retrieval_strategy = str(payload.get("retrieval_strategy") or "semantic_first").strip().lower()
    if retrieval_strategy in {"semantic_first", "metadata_first"}:
        intent.retrieval_strategy = retrieval_strategy

    semantic_query = normalize_semantic_query(payload.get("semantic_query"))
    if semantic_query:
        intent.semantic_query = semantic_query

    mode = str(payload.get("mode") or "none").strip().lower()
    time_hint_type = str(payload.get("time_hint_type") or "none").strip().lower()
    calendar_period = str(payload.get("calendar_period") or "none").strip().lower()
    relative_direction = str(payload.get("relative_direction") or "none").strip().lower()
    relative_unit = str(payload.get("relative_unit") or "none").strip().lower()

    try:
        relative_value = int(payload.get("relative_value") or 0)
    except (TypeError, ValueError):
        relative_value = 0

    if mode in {"soft_recent"} or time_hint_type == "fuzzy_recent":
        intent.boost_recent = True

    if time_hint_type == "relative_day":
        day_map = {
            "today": 0,
            "yesterday": 1,
            "day_before_yesterday": 2,
        }
        delta_days = day_map.get(calendar_period)
        if delta_days is not None:
            reference = now - timedelta(days=delta_days)
            start_ts, end_ts = make_day_bounds(reference)
            set_time_range(intent, start_ts, end_ts, calendar_period, "relative_day")
    elif time_hint_type == "calendar_period":
        if calendar_period == "this_week":
            start_ts, end_ts = make_week_bounds(now)
            set_time_range(intent, start_ts, end_ts, "this_week", "calendar_period")
        elif calendar_period == "last_week":
            start_ts, end_ts = make_week_bounds(now - timedelta(days=7))
            set_time_range(intent, start_ts, end_ts, "last_week", "calendar_period")
        elif calendar_period == "this_month":
            start_ts, end_ts = make_month_bounds(now.year, now.month, now.tzinfo)
            set_time_range(intent, start_ts, end_ts, "this_month", "calendar_period")
        elif calendar_period == "last_month":
            if now.month == 1:
                year, month = now.year - 1, 12
            else:
                year, month = now.year, now.month - 1
            start_ts, end_ts = make_month_bounds(year, month, now.tzinfo)
            set_time_range(intent, start_ts, end_ts, "last_month", "calendar_period")
        elif calendar_period == "this_year":
            start_ts, end_ts = make_year_bounds(now.year, now.tzinfo)
            set_time_range(intent, start_ts, end_ts, "this_year", "calendar_period")
        elif calendar_period == "last_year":
            start_ts, end_ts = make_year_bounds(now.year - 1, now.tzinfo)
            set_time_range(intent, start_ts, end_ts, "last_year", "calendar_period")
    elif time_hint_type == "relative_range":
        if relative_direction in {"past", "previous"} and relative_unit in {"day", "week", "month", "year"} and relative_value > 0:
            end_ts = to_unix_ts(now)
            if relative_unit == "day":
                start = now - timedelta(days=max(relative_value - 1, 0))
            elif relative_unit == "week":
                start = now - timedelta(days=7 * relative_value - 1)
            elif relative_unit == "month":
                start = now - timedelta(days=30 * relative_value - 1)
            else:
                start = now - timedelta(days=365 * relative_value - 1)
            start_ts, _ = make_day_bounds(start)
            set_time_range(intent, start_ts, end_ts, "relative_range", "relative_range")
    elif time_hint_type == "absolute_day":
        absolute_date = payload.get("absolute_date")
        if isinstance(absolute_date, str):
            try:
                reference = datetime.strptime(absolute_date, "%Y-%m-%d").replace(tzinfo=now.tzinfo)
                start_ts, end_ts = make_day_bounds(reference)
                set_time_range(intent, start_ts, end_ts, absolute_date, "absolute_day")
            except ValueError:
                pass
    elif time_hint_type == "absolute_month":
        try:
            absolute_year = int(payload.get("absolute_year"))
            absolute_month = int(payload.get("absolute_month"))
            start_ts, end_ts = make_month_bounds(absolute_year, absolute_month, now.tzinfo)
            set_time_range(intent, start_ts, end_ts, f"{absolute_year}-{absolute_month:02d}", "absolute_month")
        except (TypeError, ValueError):
            pass
    elif time_hint_type == "absolute_year":
        try:
            absolute_year = int(payload.get("absolute_year"))
            start_ts, end_ts = make_year_bounds(absolute_year, now.tzinfo)
            set_time_range(intent, start_ts, end_ts, str(absolute_year), "absolute_year")
        except (TypeError, ValueError):
            pass
    elif time_hint_type == "absolute_range":
        absolute_start = payload.get("absolute_start")
        absolute_end = payload.get("absolute_end")
        if isinstance(absolute_start, str) and isinstance(absolute_end, str):
            try:
                start_dt = datetime.strptime(absolute_start, "%Y-%m-%d").replace(tzinfo=now.tzinfo)
                end_dt = datetime.strptime(absolute_end, "%Y-%m-%d").replace(tzinfo=now.tzinfo)
                start_ts = to_unix_ts(datetime.combine(start_dt.date(), time.min, tzinfo=now.tzinfo))
                end_ts = to_unix_ts(datetime.combine(end_dt.date(), time.max, tzinfo=now.tzinfo))
                set_time_range(intent, start_ts, end_ts, f"{absolute_start}~{absolute_end}", "absolute_range")
            except ValueError:
                pass

    if intent.reason == "none":
        if intent.sort_direction is not None:
            intent.reason = f"sort_{intent.sort_direction}"
        elif intent.boost_recent:
            intent.reason = "soft_recent"

    return intent


def build_time_intent(question: str, now: datetime | None = None) -> TimeIntent:
    if not USE_TIME_AWARE_RETRIEVAL:
        return TimeIntent()

    now = now or get_now()
    try:
        intent = parse_time_intent_with_llm(question, now=now)
        if intent is not None:
            return intent
    except Exception as exc:
        print(f"⚠️ Time intent parsing fell back to rules: {exc}")

    return build_time_intent_rule_based(question, now=now)


def parse_time_intent_with_llm(question: str, now: datetime | None = None) -> TimeIntent | None:
    if not USE_TIME_AWARE_RETRIEVAL or not USE_LLM_TIME_PARSER:
        return None

    now = now or get_now()
    chain = get_time_intent_chain()
    if chain is None:
        return None

    raw_text = chain.invoke(
        {
            "question": question,
            "current_datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    payload = extract_json_object(raw_text)
    if not payload:
        return None

    intent = build_time_intent_from_payload(payload, now=now)
    mode = str(payload.get("mode") or "none").strip().lower()
    if mode != "none" and intent.retrieval_strategy == "semantic_first" and not intent.semantic_query:
        return None

    if intent.is_active:
        return intent
    if mode == "none":
        return intent
    return None


def build_time_intent_rule_based(question: str, now: datetime | None = None) -> TimeIntent:
    if not USE_TIME_AWARE_RETRIEVAL:
        return TimeIntent()

    now = now or get_now()
    intent = TimeIntent(parser_source="rule")

    explicit_patterns = [
        (r"(最近|近)(\d+|一|二|两|三|四|五|六|七|八|九|十)(天)", "day"),
        (r"(最近|近)(\d+|一|二|两|三|四|五|六|七|八|九|十)(周|星期)", "week"),
        (r"(最近|近)(\d+|一|二|两|三|四|五|六|七|八|九|十)(个?月)", "month"),
        (r"(最近|近)(\d+|一|二|两|三|四|五|六|七|八|九|十)(年)", "year"),
    ]
    for pattern, unit in explicit_patterns:
        match = re.search(pattern, question)
        if not match:
            continue
        value = parse_small_number(match.group(2))
        if value is None:
            continue

        phrase = match.group(0)
        end_ts = to_unix_ts(now)
        if unit == "day":
            start = now - timedelta(days=max(value - 1, 0))
        elif unit == "week":
            start = now - timedelta(days=7 * value - 1)
        elif unit == "month":
            start = now - timedelta(days=30 * value - 1)
        else:
            start = now - timedelta(days=365 * value - 1)
        start_ts, _ = make_day_bounds(start)
        set_time_range(intent, start_ts, end_ts, phrase, f"rolling_{unit}_range")
        break

    if not intent.has_hard_filter:
        for phrase, delta_days in (("今天", 0), ("昨日", 1), ("昨天", 1), ("前天", 2)):
            if phrase in question:
                reference = now - timedelta(days=delta_days)
                start_ts, end_ts = make_day_bounds(reference)
                set_time_range(intent, start_ts, end_ts, phrase, "relative_day")
                break

    if not intent.has_hard_filter:
        if "上周" in question:
            reference = now - timedelta(days=7)
            start_ts, end_ts = make_week_bounds(reference)
            set_time_range(intent, start_ts, end_ts, "上周", "last_week")
        elif "这周" in question or "本周" in question:
            phrase = "这周" if "这周" in question else "本周"
            start_ts, end_ts = make_week_bounds(now)
            set_time_range(intent, start_ts, end_ts, phrase, "this_week")

    if not intent.has_hard_filter:
        if "上个月" in question:
            if now.month == 1:
                year, month = now.year - 1, 12
            else:
                year, month = now.year, now.month - 1
            start_ts, end_ts = make_month_bounds(year, month, now.tzinfo)
            set_time_range(intent, start_ts, end_ts, "上个月", "last_month")
        elif "这个月" in question or "本月" in question:
            phrase = "这个月" if "这个月" in question else "本月"
            start_ts, end_ts = make_month_bounds(now.year, now.month, now.tzinfo)
            set_time_range(intent, start_ts, end_ts, phrase, "this_month")

    if not intent.has_hard_filter:
        if "去年" in question:
            start_ts, end_ts = make_year_bounds(now.year - 1, now.tzinfo)
            set_time_range(intent, start_ts, end_ts, "去年", "last_year")
        elif "今年" in question:
            start_ts, end_ts = make_year_bounds(now.year, now.tzinfo)
            set_time_range(intent, start_ts, end_ts, "今年", "this_year")

    if not intent.has_hard_filter:
        day_match = re.search(r"(\d{4})[年/\-.](\d{1,2})[月/\-.](\d{1,2})日?", question)
        if day_match:
            year = int(day_match.group(1))
            month = int(day_match.group(2))
            day = int(day_match.group(3))
            reference = datetime(year, month, day, tzinfo=now.tzinfo)
            start_ts, end_ts = make_day_bounds(reference)
            set_time_range(intent, start_ts, end_ts, day_match.group(0), "absolute_day")

    if not intent.has_hard_filter:
        month_match = re.search(r"(\d{4})[年/\-.](\d{1,2})月?", question)
        if month_match:
            year = int(month_match.group(1))
            month = int(month_match.group(2))
            start_ts, end_ts = make_month_bounds(year, month, now.tzinfo)
            set_time_range(intent, start_ts, end_ts, month_match.group(0), "absolute_month")

    if not intent.has_hard_filter:
        match = re.search(r"(\d{4})年", question)
        if match:
            year = int(match.group(1))
            start_ts, end_ts = make_year_bounds(year, now.tzinfo)
            set_time_range(intent, start_ts, end_ts, match.group(0), "absolute_year")

    if not intent.has_hard_filter:
        match = re.search(r"(\d{1,2})月(\d{1,2})日", question)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            reference = datetime(now.year, month, day, tzinfo=now.tzinfo)
            start_ts, end_ts = make_day_bounds(reference)
            set_time_range(intent, start_ts, end_ts, match.group(0), "month_day")

    recent_sort_patterns = ("最新", "最近一次", "最后一次", "上一次", "最新一条", "最后一条")
    oldest_sort_patterns = ("最早", "第一次", "最开始", "最先")
    oldest_sort_regexes = (
        r"第[一二两三四五六七八九十\d]+条",
        r"第[一二两三四五六七八九十\d]+篇",
        r"第[一二两三四五六七八九十\d]+个",
    )

    for phrase in recent_sort_patterns:
        if phrase in question:
            intent.sort_direction = "recent"
            add_matched_phrase(intent, phrase)
            if intent.reason == "none":
                intent.reason = "sort_recent"
            break

    if intent.sort_direction is None:
        for phrase in oldest_sort_patterns:
            if phrase in question:
                intent.sort_direction = "oldest"
                add_matched_phrase(intent, phrase)
                if intent.reason == "none":
                    intent.reason = "sort_oldest"
                break

    if intent.sort_direction is None:
        for pattern in oldest_sort_regexes:
            match = re.search(pattern, question)
            if match:
                intent.sort_direction = "oldest"
                add_matched_phrase(intent, match.group(0))
                if intent.reason == "none":
                    intent.reason = "sort_oldest"
                break

    if (
        not intent.has_hard_filter
        and intent.sort_direction is None
        and re.search(r"(最近|近期|近来|这阵子|这段时间)", question)
    ):
        intent.boost_recent = True
        if intent.reason == "none":
            intent.reason = "soft_recent"
        soft_match = re.search(r"(最近|近期|近来|这阵子|这段时间)", question)
        add_matched_phrase(intent, soft_match.group(0) if soft_match else None)

    return intent


def build_rule_based_retrieval_query(question: str, time_intent: TimeIntent | None) -> str:
    cleaned = question
    for phrase in sorted(set((time_intent or TimeIntent()).matched_phrases), key=len, reverse=True):
        cleaned = cleaned.replace(phrase, " ", 1)

    cleaned = normalize_semantic_query(cleaned)
    if cleaned is None:
        return question.strip()
    return cleaned


def build_retrieval_query(question: str, time_intent: TimeIntent | None = None) -> str:
    if not USE_TIME_AWARE_RETRIEVAL:
        return question.strip()

    time_intent = time_intent or build_time_intent(question)
    if time_intent.parser_source == "llm":
        return time_intent.semantic_query or question.strip()
    return build_rule_based_retrieval_query(question, time_intent)


def strip_generic_query_words(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[，。！？?、,.:：;；!！]", "", normalized)

    generic_phrases = [
        "是什么时候",
        "是什么",
        "有哪些",
        "有没有",
        "哪一条",
        "哪一篇",
        "哪一个",
        "哪条",
        "哪篇",
        "哪个",
        "什么",
        "哪些",
        "时候",
        "内容",
        "memo",
        "笔记",
        "记录",
        "我发的",
        "我写的",
        "我记的",
        "我发过的",
        "我写过的",
        "我记过的",
        "我发过",
        "我写过",
        "我记过",
        "我发",
        "我写",
        "我记",
        "发过",
        "写过",
        "记过",
        "发的",
        "写的",
        "记的",
        "发",
        "写",
        "记",
        "我的",
        "我",
        "是",
        "了",
        "的",
        "吗",
        "呢",
    ]

    for phrase in sorted(generic_phrases, key=len, reverse=True):
        normalized = normalized.replace(phrase, "")

    return normalized


def decide_time_retrieval_strategy(question: str, search_query: str, time_intent: TimeIntent | None) -> str:
    if (
        not USE_TIME_AWARE_RETRIEVAL
        or time_intent is None
        or time_intent.sort_direction not in {"oldest", "recent"}
    ):
        return "semantic_first"

    if time_intent.parser_source == "llm" and not time_intent.semantic_query:
        if time_intent.retrieval_strategy in {"semantic_first", "metadata_first"}:
            return time_intent.retrieval_strategy
        return "metadata_first"

    semantic_basis = time_intent.semantic_query or search_query
    remainder = strip_generic_query_words(semantic_basis)
    if remainder:
        return "semantic_first"

    if time_intent.retrieval_strategy == "metadata_first":
        return "metadata_first"

    return "metadata_first"


def get_metadata_sorted_candidates(documents: list, time_intent: TimeIntent | None, top_k: int) -> list:
    candidates = [doc for doc in documents if doc_matches_time_filter(doc, time_intent)]
    if not candidates:
        return []

    reverse = time_intent is not None and time_intent.sort_direction == "recent"
    candidates.sort(
        key=lambda doc: ((get_created_ts(doc) or 0), -get_chunk_index(doc) if reverse else get_chunk_index(doc)),
        reverse=reverse,
    )
    return candidates[:top_k]


def build_chroma_time_filter(time_intent: TimeIntent | None) -> dict | None:
    if not USE_TIME_AWARE_RETRIEVAL or time_intent is None or not time_intent.has_hard_filter:
        return None

    clauses = []
    if time_intent.start_ts is not None:
        clauses.append({"created_ts": {"$gte": int(time_intent.start_ts)}})
    if time_intent.end_ts is not None:
        clauses.append({"created_ts": {"$lte": int(time_intent.end_ts)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def doc_matches_time_filter(doc, time_intent: TimeIntent | None) -> bool:
    if not USE_TIME_AWARE_RETRIEVAL or time_intent is None or not time_intent.has_hard_filter:
        return True

    created_ts = get_created_ts(doc)
    if created_ts is None:
        return False
    if time_intent.start_ts is not None and created_ts < time_intent.start_ts:
        return False
    if time_intent.end_ts is not None and created_ts > time_intent.end_ts:
        return False
    return True


def compute_recency_score(created_ts: int | None, now_ts: int, half_life_days: float) -> float:
    if created_ts is None:
        return 0.0
    age_days = max((now_ts - created_ts) / 86400.0, 0.0)
    return math.exp(-age_days / max(half_life_days, 1.0))


def apply_recency_boost(scored_docs: list[tuple], time_intent: TimeIntent | None, top_k: int) -> list[tuple]:
    if not scored_docs:
        return []

    should_boost = (
        USE_TIME_AWARE_RETRIEVAL
        and time_intent is not None
        and (time_intent.boost_recent or time_intent.sort_direction == "recent")
    )
    if not should_boost:
        return scored_docs[:top_k]

    now_ts = to_unix_ts(get_now())
    weight = TIME_RECENCY_BOOST_WEIGHT
    if time_intent.sort_direction == "recent":
        weight *= 1.5

    boosted = []
    for doc, base_score in scored_docs:
        recency_score = compute_recency_score(get_created_ts(doc), now_ts, TIME_RECENCY_HALF_LIFE_DAYS)
        boosted.append((doc, float(base_score) + weight * recency_score))

    boosted.sort(key=lambda item: item[1], reverse=True)
    return boosted[:top_k]


def apply_final_time_ordering(results: list[tuple], time_intent: TimeIntent | None, top_k: int) -> list[tuple]:
    if (
        not USE_TIME_AWARE_RETRIEVAL
        or time_intent is None
        or time_intent.sort_direction not in {"recent", "oldest"}
    ):
        return results[:top_k]

    reverse = time_intent.sort_direction == "recent"
    ordered = sorted(
        results,
        key=lambda item: (get_created_ts(item[0]) or 0, float(item[1])),
        reverse=reverse,
    )
    return ordered[:top_k]


def serialize_time_intent(time_intent: TimeIntent | None) -> dict:
    if time_intent is None:
        return {
            "active": False,
            "has_hard_filter": False,
            "boost_recent": False,
            "sort_direction": None,
            "semantic_query": None,
            "matched_phrases": [],
            "reason": "none",
            "parser_source": "none",
            "retrieval_strategy": "semantic_first",
            "start_ts": None,
            "end_ts": None,
            "start_date": None,
            "end_date": None,
        }

    return {
        "active": time_intent.is_active,
        "has_hard_filter": time_intent.has_hard_filter,
        "boost_recent": time_intent.boost_recent,
        "sort_direction": time_intent.sort_direction,
        "semantic_query": time_intent.semantic_query,
        "matched_phrases": list(time_intent.matched_phrases),
        "reason": time_intent.reason,
        "parser_source": time_intent.parser_source,
        "retrieval_strategy": time_intent.retrieval_strategy,
        "start_ts": time_intent.start_ts,
        "end_ts": time_intent.end_ts,
        "start_date": format_ts(time_intent.start_ts),
        "end_date": format_ts(time_intent.end_ts),
    }


class SimpleBM25Retriever:
    """一个轻量 BM25 检索器，避免为混合检索额外引入新依赖。"""

    def __init__(self, documents):
        self.documents = documents
        self.doc_tokens = [tokenize_for_bm25(doc.page_content) for doc in documents]
        self.doc_term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.doc_freqs = defaultdict(int)
        self.k1 = 1.5
        self.b = 0.75

        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freqs[token] += 1

    def retrieve(self, query: str, top_k: int, time_intent: TimeIntent | None = None):
        if not self.documents:
            return []

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        candidate_indices = [
            idx for idx, doc in enumerate(self.documents)
            if doc_matches_time_filter(doc, time_intent)
        ]
        if not candidate_indices:
            return []

        scores = []
        corpus_size = len(candidate_indices)
        avg_doc_len = sum(self.doc_lens[idx] for idx in candidate_indices) / corpus_size
        doc_freqs = defaultdict(int)
        for idx in candidate_indices:
            for token in set(self.doc_tokens[idx]):
                doc_freqs[token] += 1

        for idx in candidate_indices:
            term_freqs = self.doc_term_freqs[idx]
            doc_len = self.doc_lens[idx]
            score = 0.0

            for token in query_tokens:
                term_freq = term_freqs.get(token, 0)
                if term_freq == 0:
                    continue

                doc_freq = doc_freqs.get(token, 0)
                idf = math.log(1 + (corpus_size - doc_freq + 0.5) / (doc_freq + 0.5))
                denom = term_freq + self.k1 * (1 - self.b + self.b * doc_len / max(avg_doc_len, 1))
                score += idf * (term_freq * (self.k1 + 1)) / denom

            if score > 0:
                scores.append((self.documents[idx], score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]


def reciprocal_rank_fusion_scores(result_lists: Iterable[list]) -> list[tuple]:
    """使用 RRF 融合多路检索结果，并返回带分数的结果。"""
    fused_scores = defaultdict(float)
    doc_by_id = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list, start=1):
            doc = item[0] if isinstance(item, tuple) else item
            chunk_id = get_chunk_id(doc)
            if not chunk_id:
                continue

            doc_by_id[chunk_id] = doc
            fused_scores[chunk_id] += 1.0 / (RRF_K + rank)

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [(doc_by_id[chunk_id], fused_scores[chunk_id]) for chunk_id in ranked_ids]


def reciprocal_rank_fusion(result_lists: Iterable[list], top_k: int):
    """使用 RRF 融合多路检索结果。"""
    return [doc for doc, _ in reciprocal_rank_fusion_scores(result_lists)[:top_k]]


def set_hf_offline_mode(required_models: list[str]):
    """根据本地缓存情况决定是否启用 HuggingFace 离线模式。"""
    # 智能离线模式控制
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        cached_repo_ids = {str(repo.repo_id) for repo in cache_info.repos}
        all_cached = all(
            any(model_name in repo_id for repo_id in cached_repo_ids)
            for model_name in required_models
        )
        if all_cached:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ["HF_HUB_OFFLINE"] = "0"
    except:
        os.environ["HF_HUB_OFFLINE"] = "0"


def rerank_documents(query: str, docs: list, reranker, top_k: int = FINAL_TOP_K):
    """使用 cross-encoder reranker 对候选文档精排。"""
    if not docs or reranker is None:
        return [(doc, 0.0) for doc in docs[:top_k]]

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs, batch_size=RERANK_BATCH_SIZE, show_progress_bar=False)
    ranked = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:top_k]]


def create_chat_model(streaming: bool, temperature: float = 0.0, model_name: str | None = None):
    return ChatOpenAI(
        model=model_name or os.getenv("OPENAI_MODEL_NAME", "glm-4.6"),
        temperature=temperature,
        streaming=streaming,
    )


def create_time_intent_chain():
    prompt = ChatPromptTemplate.from_template(
        """你是一个时间意图解析器。请把用户问题中的时间检索意图解析成严格 JSON。

当前日期时间（本地时区）：
{current_datetime}

输出要求：
1. 只输出一个 JSON 对象，不要输出任何解释
2. 所有字段都必须出现
3. 不确定时使用最保守的值，不要编造
4. 你负责“理解时间表达”，不要计算 Unix 时间戳
5. 对“第一条 / 第一篇 / 第一个 / 最开始 / 最先 / 第一次”这类表达，应视为 sort_direction=oldest
6. 对“最新 / 最后一次 / 最近一次 / 最新一条”这类表达，应视为 sort_direction=recent
7. 对“最近 / 近期 / 近来 / 这段时间 / 前阵子”这类模糊表达，如果没有明确时间窗口，使用 soft_recent
8. 如果问题中既有过滤又有排序，可以同时表达
9. matched_phrases 必须是用户原问题中的原始片段，保持原文，不要翻译成英文
10. 如果问题主要是在问“第一条/最后一条 memo/笔记/记录是什么”这类元数据排序问题，而不是在某个主题上检索，请使用 retrieval_strategy=metadata_first
11. semantic_query 应该是去掉时间约束后、保留主题内容的检索 query；如果问题本身几乎没有主题内容，可以给空字符串

JSON schema:
{{
  "mode": "none | hard_filter | soft_recent | sort_only | hard_filter_and_sort",
  "retrieval_strategy": "semantic_first | metadata_first",
  "time_hint_type": "none | relative_day | relative_range | calendar_period | absolute_day | absolute_month | absolute_year | absolute_range | fuzzy_recent",
  "sort_direction": "none | oldest | recent",
  "relative_direction": "none | past | current | previous",
  "relative_unit": "none | day | week | month | year",
  "relative_value": 0,
  "calendar_period": "none | today | yesterday | day_before_yesterday | this_week | last_week | this_month | last_month | this_year | last_year",
  "absolute_date": null,
  "absolute_year": null,
  "absolute_month": null,
  "absolute_start": null,
  "absolute_end": null,
  "semantic_query": "",
  "matched_phrases": [],
  "confidence": 0.0
}}

字段说明：
- absolute_date: YYYY-MM-DD
- absolute_start / absolute_end: YYYY-MM-DD
- absolute_year: 四位年份整数
- absolute_month: 1-12 的整数
- relative_value: 如果是“最近一个月”，输出 1；如果不是相对范围则输出 0
- semantic_query: 去掉时间表达后，真正用于主题检索的纯内容 query

用户问题：
{question}
"""
    )
    return prompt | create_chat_model(streaming=False, temperature=0.0, model_name=TIME_INTENT_MODEL_NAME) | StrOutputParser()


def get_time_intent_chain():
    global _TIME_INTENT_CHAIN
    if _TIME_INTENT_CHAIN is None and USE_LLM_TIME_PARSER:
        _TIME_INTENT_CHAIN = create_time_intent_chain()
    return _TIME_INTENT_CHAIN


def extract_json_object(raw_text: str) -> dict | None:
    raw_text = raw_text.strip()
    if not raw_text:
        return None

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def create_query_rewrite_chain():
    prompt = ChatPromptTemplate.from_template(
        """你是一个检索改写助手。

请把用户问题改写为 {rewrite_count} 条适合知识库检索的查询语句。

要求：
1. 保持原意，不要添加原问题没有的新信息
2. 尽量使用不同表达方式
3. 一条可以偏自然语言，一条可以偏关键词
4. 每行输出一条，不要编号，不要解释

用户问题：
{question}
"""
    )
    return prompt | create_chat_model(streaming=False, temperature=0.0) | StrOutputParser()


def normalize_rewrite_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)
    return line.strip("` ").strip()


def parse_query_rewrites(raw_text: str, original_question: str, rewrite_count: int):
    rewrites = []
    seen = {original_question.strip()}

    for raw_line in raw_text.splitlines():
        line = normalize_rewrite_line(raw_line)
        if not line:
            continue
        if line.lower().startswith("用户问题"):
            continue
        if line in seen:
            continue
        rewrites.append(line)
        seen.add(line)
        if len(rewrites) >= rewrite_count:
            break

    return rewrites


def generate_query_rewrites(question: str, rewrite_chain=None, rewrite_count: int = QUERY_REWRITE_COUNT):
    if rewrite_chain is None or rewrite_count <= 0:
        return []

    try:
        raw_text = rewrite_chain.invoke({"question": question, "rewrite_count": rewrite_count})
        return parse_query_rewrites(raw_text, question, rewrite_count)
    except Exception as exc:
        print(f"⚠️ Query rewrite failed: {exc}")
        return []


def initialize_retrieval_components():
    """初始化混合检索所需的 Embedding、向量库、BM25 检索器、reranker 和 rewrite chain。"""
    required_models = [EMBEDDING_MODEL_NAME]
    if USE_RERANK:
        required_models.append(RERANKER_MODEL_NAME)
    set_hf_offline_mode(required_models)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="memos_rag"
    )

    bm25_documents = process_documents(fetch_all_memos())
    bm25_retriever = SimpleBM25Retriever(bm25_documents)
    reranker = None
    if USE_RERANK:
        reranker = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")
    rewrite_chain = None
    if USE_QUERY_REWRITE:
        rewrite_chain = create_query_rewrite_chain()
    return vector_db, bm25_retriever, reranker, rewrite_chain


def run_hybrid_retrieval(
    query: str,
    vector_db,
    bm25_retriever,
    reranker=None,
    dense_top_k: int = DENSE_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    fusion_top_k: int = FUSION_TOP_K,
    final_top_k: int = FINAL_TOP_K,
    time_intent: TimeIntent | None = None,
    search_query: str | None = None,
):
    """执行 dense + BM25 混合召回，并在需要时执行 rerank。"""
    time_intent = time_intent or build_time_intent(query)
    search_query = search_query or build_retrieval_query(query, time_intent)
    retrieval_strategy = decide_time_retrieval_strategy(query, search_query, time_intent)
    time_intent.retrieval_strategy = retrieval_strategy

    if retrieval_strategy == "metadata_first":
        metadata_candidate_k = max(final_top_k, fusion_top_k, TIME_SORT_CANDIDATE_POOL)
        metadata_docs = get_metadata_sorted_candidates(
            bm25_retriever.documents,
            time_intent,
            top_k=metadata_candidate_k,
        )
        metadata_results = [(doc, 0.0) for doc in metadata_docs]
        return {
            "query": query,
            "search_query": search_query,
            "time_intent": serialize_time_intent(time_intent),
            "dense": [],
            "bm25": [],
            "fused": metadata_docs[:fusion_top_k],
            "reranked": metadata_results[:final_top_k],
        }

    chroma_filter = build_chroma_time_filter(time_intent)

    dense_results = vector_db.similarity_search_with_score(
        search_query,
        k=dense_top_k,
        filter=chroma_filter,
    )
    bm25_results = bm25_retriever.retrieve(search_query, top_k=bm25_top_k, time_intent=time_intent)
    fused_scores = reciprocal_rank_fusion_scores([dense_results, bm25_results])
    fused_scores = apply_recency_boost(fused_scores, time_intent, fusion_top_k)
    fused_docs = [doc for doc, _ in fused_scores]

    rerank_top_k = final_top_k
    if USE_TIME_AWARE_RETRIEVAL and time_intent.sort_direction in {"recent", "oldest"}:
        rerank_top_k = max(final_top_k, TIME_SORT_CANDIDATE_POOL)

    reranked_results = rerank_documents(search_query, fused_docs, reranker, top_k=rerank_top_k)
    reranked_results = apply_final_time_ordering(reranked_results, time_intent, top_k=final_top_k)
    return {
        "query": query,
        "search_query": search_query,
        "time_intent": serialize_time_intent(time_intent),
        "dense": dense_results,
        "bm25": bm25_results,
        "fused": fused_docs,
        "reranked": reranked_results,
    }


def run_multi_query_retrieval(
    question: str,
    vector_db,
    bm25_retriever,
    reranker=None,
    rewrite_chain=None,
    rewrite_count: int = QUERY_REWRITE_COUNT,
    dense_top_k: int = MULTI_QUERY_DENSE_TOP_K,
    bm25_top_k: int = MULTI_QUERY_BM25_TOP_K,
    per_query_fusion_top_k: int = MULTI_QUERY_FUSION_TOP_K,
    global_fusion_top_k: int = MULTI_QUERY_GLOBAL_TOP_K,
    final_top_k: int = FINAL_TOP_K,
):
    original_time_intent = build_time_intent(question)
    original_search_query = build_retrieval_query(question, original_time_intent)
    rewrites = generate_query_rewrites(question, rewrite_chain, rewrite_count)
    queries = [question] + rewrites

    by_query = []
    per_query_fused_lists = []
    for query_text in queries:
        query_search_query = build_retrieval_query(query_text)
        query_result = run_hybrid_retrieval(
            query_text,
            vector_db,
            bm25_retriever,
            reranker=None,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            fusion_top_k=per_query_fusion_top_k,
            final_top_k=per_query_fusion_top_k,
            time_intent=original_time_intent,
            search_query=query_search_query,
        )
        by_query.append({"query": query_text, **query_result})
        per_query_fused_lists.append(query_result["fused"])

    global_fused_scores = reciprocal_rank_fusion_scores(per_query_fused_lists)
    global_fused_scores = apply_recency_boost(global_fused_scores, original_time_intent, global_fusion_top_k)
    global_fused_docs = [doc for doc, _ in global_fused_scores]

    rerank_top_k = final_top_k
    if USE_TIME_AWARE_RETRIEVAL and original_time_intent.sort_direction in {"recent", "oldest"}:
        rerank_top_k = max(final_top_k, TIME_SORT_CANDIDATE_POOL)

    reranked_results = rerank_documents(
        original_search_query,
        global_fused_docs,
        reranker,
        top_k=rerank_top_k,
    )
    reranked_results = apply_final_time_ordering(reranked_results, original_time_intent, top_k=final_top_k)
    return {
        "question": question,
        "search_query": original_search_query,
        "time_intent": serialize_time_intent(original_time_intent),
        "rewrites": rewrites,
        "queries": queries,
        "by_query": by_query,
        "global_fused": global_fused_docs,
        "reranked": reranked_results,
    }

def format_docs(docs):
    """将检索到的文档格式化为字符串，包含日期信息"""
    formatted = []
    for doc in docs:
        date = doc.metadata.get("date", "Unknown")
        content = doc.page_content
        formatted.append(f"[日期: {date}]\n{content}")
    return "\n\n---\n\n".join(formatted)


def build_retrieval_notes(retrieval_result: dict) -> str:
    time_intent = retrieval_result.get("time_intent", {}) or {}
    notes = []

    retrieval_strategy = time_intent.get("retrieval_strategy")
    sort_direction = time_intent.get("sort_direction")
    start_date = time_intent.get("start_date")
    end_date = time_intent.get("end_date")
    search_query = retrieval_result.get("search_query")

    if search_query:
        notes.append(f"- 主题检索 query: {search_query}")

    if start_date and end_date:
        notes.append(f"- 已先按时间范围过滤候选: {start_date} ~ {end_date}")

    if retrieval_strategy == "metadata_first":
        notes.append("- 这是 metadata-first 查询，候选主要依据时间元数据筛选与排序，不依赖正文显式写出“第一条/最后一次”等字样。")
    elif retrieval_strategy == "semantic_first":
        notes.append("- 这是 semantic-first 查询，候选先按主题相关性召回。")

    if sort_direction == "oldest":
        notes.append("- 当前结果已按时间从早到晚排序，第一条就是满足“最早/第一条/第一次”条件的候选。")
    elif sort_direction == "recent":
        notes.append("- 当前结果已按时间从晚到早排序，第一条就是满足“最近一次/最后一次/最新”条件的候选。")

    if time_intent.get("boost_recent"):
        notes.append("- 检索阶段已对较新的记录做 recent boost，但并不是硬性时间截断。")

    if not notes:
        return "无额外检索说明。"
    return "\n".join(notes)


def format_retrieval_context(retrieval_result: dict) -> str:
    docs = [doc for doc, _ in retrieval_result.get("reranked", [])]
    notes = build_retrieval_notes(retrieval_result)
    docs_text = format_docs(docs)
    return f"【检索说明】\n{notes}\n\n【相关的笔记片段】\n{docs_text}"


def build_empty_retrieval_response(retrieval_result: dict) -> str:
    time_intent = retrieval_result.get("time_intent", {}) or {}
    start_date = time_intent.get("start_date")
    end_date = time_intent.get("end_date")
    if start_date and end_date:
        return f"我的记忆库里没有相关记录。当前时间过滤范围是 {start_date} ~ {end_date}。"
    return "我的记忆库里没有相关记录。"

def get_rag_chain():
    """初始化并返回 RAG 处理链"""
    print("🧠 Initializing Second Brain Core...")
    vector_db, bm25_retriever, reranker, rewrite_chain = initialize_retrieval_components()

    def hybrid_retrieve(query: str):
        if USE_QUERY_REWRITE:
            return run_multi_query_retrieval(
                query,
                vector_db,
                bm25_retriever,
                reranker=reranker,
                rewrite_chain=rewrite_chain,
            )

        return run_hybrid_retrieval(query, vector_db, bm25_retriever, reranker=reranker)

    def prepare_answer_input(query: str):
        retrieval_result = hybrid_retrieve(query)
        docs = retrieval_result.get("reranked", [])
        if not docs:
            return {
                "question": query,
                "should_answer_directly": True,
                "direct_answer": build_empty_retrieval_response(retrieval_result),
            }

        return {
            "question": query,
            "context": format_retrieval_context(retrieval_result),
            "should_answer_directly": False,
        }

    # 4. 初始化 LLM
    llm = create_chat_model(streaming=True, temperature=0.3)

    # 5. 定义 Prompt
    template = """你是一个基于我的 Memos 笔记构建的【个人第二大脑】。
    
    请根据以下【检索说明】和【相关的笔记片段】来回答我的问题。
    如果笔记中没有相关内容，请诚实地告诉我“我的记忆库里没有相关记录”，不要编造。
    如果【检索说明】已经明确指出这些片段经过了时间过滤或排序，请把这个检索结论当作可靠前提，不要要求正文必须显式写出“第一条”“最早”“最后一次”等字样。
    
    回答时请引用笔记中的日期，以证明你的来源。
    
    {context}
    
    【我的问题】: {question}
    
    【你的回答】:
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    # 6. 构建 RAG 链 (LCEL)
    rag_chain = (
        RunnableLambda(prepare_answer_input)
        | RunnableBranch(
            (lambda x: x["should_answer_directly"], RunnableLambda(lambda x: x["direct_answer"])),
            RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]})
            | prompt
            | llm
            | StrOutputParser(),
        )
    )
    
    return rag_chain

def main():
    # 1. 检查 API Key
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "sk-...":
        print("⚠️  Warning: OPENAI_API_KEY not set in .env file.")
    
    # 2. 获取处理链
    rag_chain = get_rag_chain()

    print("✅ System Ready! (Type 'exit' to quit)")
    print("-" * 50)

    # 3. 聊天循环
    while True:
        try:
            user_input = input("\n🧑 You: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Bye!")
                break
            
            if not user_input.strip():
                continue

            print("🤖 Brain: ", end="", flush=True)
            
            # 流式输出
            for chunk in rag_chain.stream(user_input):
                print(chunk, end="", flush=True)
            print("\n")
            
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
