from __future__ import annotations

import hashlib
import html
import os
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from article_retriever import retrieve_relevant_articles
from data_loader import load_university_data
from document_reader import (
    DocumentReadError,
    extract_text_from_upload,
    get_ocr_status,
    is_image_filename,
)
from llm_service import LLMService
from person_ranker import rank_people


# -----------------------------------------------------------------------------
# Environment and page configuration
# -----------------------------------------------------------------------------
load_dotenv(override=False)

st.set_page_config(
    page_title="سامانه ارجاع هوشمند نامه‌ها",
    page_icon="📨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def escape(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [str(value)]


def format_number(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:,.{digits}f}"


def render_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&display=swap');

        :root {
            --bg: #07111f;
            --surface: rgba(15, 31, 50, 0.82);
            --surface-soft: rgba(255, 255, 255, 0.055);
            --border: rgba(148, 163, 184, 0.18);
            --text: #f8fafc;
            --muted: #a9b7c9;
            --primary: #7c3aed;
            --primary-2: #0891b2;
            --success: #10b981;
            --warning: #f59e0b;
        }

        /* Base RTL and typography */
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            font-family: 'Vazirmatn', Tahoma, Arial, sans-serif !important;
            direction: rtl !important;
            text-align: right !important;
        }

        .stApp {
            background:
                radial-gradient(circle at 88% 4%, rgba(124, 58, 237, 0.22), transparent 25%),
                radial-gradient(circle at 12% 18%, rgba(8, 145, 178, 0.17), transparent 23%),
                linear-gradient(145deg, #050b14 0%, #07111f 52%, #0a1727 100%);
            color: var(--text);
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        #MainMenu, footer, [data-testid="stToolbar"] {
            visibility: hidden;
        }

        .block-container {
            max-width: 1380px;
            padding-top: 1.2rem;
            padding-bottom: 4rem;
        }

        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"],
        [data-testid="stAlert"],
        [data-testid="stMetric"],
        label, p, span, h1, h2, h3, h4, h5, h6 {
            direction: rtl !important;
            text-align: right !important;
        }

        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input,
        [data-baseweb="input"] input {
            direction: rtl !important;
            text-align: right !important;
            font-family: 'Vazirmatn', Tahoma, Arial, sans-serif !important;
        }

        div[role="radiogroup"] {
            direction: rtl !important;
            justify-content: flex-start;
            gap: .45rem;
        }

        div[role="radiogroup"] label {
            direction: rtl !important;
            border: 1px solid var(--border);
            background: rgba(255,255,255,.04);
            border-radius: 14px;
            padding: .35rem .75rem;
        }

        [data-testid="stHorizontalBlock"] {
            direction: rtl;
        }

        section[data-testid="stSidebar"] {
            background: rgba(5, 13, 24, 0.96);
            border-left: 1px solid var(--border);
        }

        section[data-testid="stSidebar"] * {
            direction: rtl !important;
            text-align: right !important;
        }

        /* Hero */
        .hero-shell {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 30px;
            padding: 32px 34px;
            background:
                linear-gradient(120deg, rgba(124,58,237,.19), rgba(8,145,178,.08)),
                rgba(15,31,50,.70);
            box-shadow: 0 24px 70px rgba(0,0,0,.28);
            margin-bottom: 20px;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            width: 280px;
            height: 280px;
            left: -80px;
            top: -115px;
            border-radius: 50%;
            background: rgba(34,211,238,.13);
            filter: blur(8px);
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border: 1px solid rgba(167,139,250,.28);
            background: rgba(124,58,237,.10);
            color: #ddd6fe;
            border-radius: 999px;
            padding: 7px 12px;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 14px;
        }

        .hero-title {
            position: relative;
            z-index: 1;
            color: #ffffff;
            font-size: clamp(30px, 4vw, 52px);
            font-weight: 800;
            line-height: 1.35;
            margin: 0 0 10px 0;
        }

        .hero-title span {
            color: #67e8f9;
        }

        .hero-desc {
            position: relative;
            z-index: 1;
            max-width: 950px;
            color: #cbd5e1;
            line-height: 2;
            font-size: 16px;
            margin: 0;
        }

        /* Cards */
        .panel {
            border: 1px solid var(--border);
            background: var(--surface);
            border-radius: 24px;
            padding: 22px;
            box-shadow: 0 16px 48px rgba(0,0,0,.18);
            margin-bottom: 16px;
        }

        .panel-title {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #ffffff;
            font-size: 20px;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .panel-subtitle {
            color: var(--muted);
            line-height: 1.9;
            font-size: 14px;
            margin-bottom: 4px;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0,1fr));
            gap: 12px;
            margin: 0 0 20px 0;
        }

        .stat-card {
            border: 1px solid var(--border);
            background: var(--surface-soft);
            border-radius: 20px;
            padding: 17px 18px;
        }

        .stat-label {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 5px;
        }

        .stat-value {
            color: #ffffff;
            font-weight: 800;
            font-size: 24px;
        }

        .status-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 12px 14px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,.035);
            border-radius: 16px;
            margin-bottom: 10px;
        }

        .status-dot {
            width: 9px;
            height: 9px;
            display: inline-block;
            border-radius: 999px;
            margin-left: 7px;
        }

        .ok { background: #34d399; box-shadow: 0 0 0 5px rgba(52,211,153,.10); }
        .warn { background: #fbbf24; box-shadow: 0 0 0 5px rgba(251,191,36,.10); }

        .chip {
            display: inline-flex;
            align-items: center;
            padding: 6px 10px;
            margin: 3px;
            border-radius: 999px;
            border: 1px solid rgba(103,232,249,.22);
            background: rgba(8,145,178,.10);
            color: #cffafe;
            font-size: 13px;
        }

        .analysis-card, .person-card {
            border: 1px solid var(--border);
            border-radius: 22px;
            background:
                linear-gradient(135deg, rgba(124,58,237,.08), rgba(8,145,178,.06)),
                rgba(15,31,50,.76);
            padding: 20px;
            margin-bottom: 14px;
        }

        .person-name {
            color: #ffffff;
            font-size: 19px;
            font-weight: 800;
            margin-bottom: 10px;
        }

        .kv {
            display: grid;
            grid-template-columns: 150px 1fr;
            gap: 10px;
            padding: 8px 0;
            border-bottom: 1px dashed rgba(148,163,184,.16);
        }

        .kv:last-child { border-bottom: none; }
        .kv-label { color: #a5b4fc; font-weight: 700; }
        .kv-value { color: #e2e8f0; line-height: 1.9; }

        div[data-testid="stFileUploader"] {
            border: 1px dashed rgba(103,232,249,.34);
            border-radius: 20px;
            background: rgba(8,145,178,.055);
            padding: 10px;
        }

        div[data-testid="stTextArea"] textarea,
        div[data-testid="stNumberInput"] input {
            border-radius: 16px !important;
            border: 1px solid var(--border) !important;
            background: rgba(2, 12, 24, .62) !important;
            color: #f8fafc !important;
        }

        .stButton > button {
            min-height: 48px;
            border: none;
            border-radius: 16px;
            font-family: 'Vazirmatn', Tahoma, Arial, sans-serif !important;
            font-weight: 800;
            background: linear-gradient(95deg, #7c3aed, #0891b2);
            color: white;
            box-shadow: 0 14px 34px rgba(8,145,178,.18);
            transition: transform .15s ease, box-shadow .15s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 42px rgba(124,58,237,.25);
            color: white;
        }

        div[data-testid="stAlert"] {
            border-radius: 16px;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 18px;
            background: rgba(255,255,255,.035);
        }

        [data-testid="stDataFrame"] {
            direction: ltr !important;
            text-align: left !important;
            border-radius: 18px;
            overflow: hidden;
        }

        code, pre {
            direction: ltr !important;
            text-align: left !important;
        }

        @media (max-width: 900px) {
            .stat-grid { grid-template-columns: 1fr; }
            .hero-shell { padding: 24px 20px; border-radius: 24px; }
            .kv { grid-template-columns: 1fr; gap: 3px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_panel_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{escape(icon)} {escape(title)}</div>
            <div class="panel-subtitle">{escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chips(items: list[str]) -> None:
    if not items:
        st.markdown("<span class='chip'>موردی ثبت نشده است</span>", unsafe_allow_html=True)
        return
    markup = "".join(f"<span class='chip'>{escape(item)}</span>" for item in items)
    st.markdown(markup, unsafe_allow_html=True)


def render_analysis(analysis: Any) -> None:
    confidence = getattr(analysis, "confidence", None)
    if isinstance(confidence, (int, float)) and confidence <= 1:
        confidence_text = f"{confidence * 100:.0f}%"
    elif isinstance(confidence, (int, float)):
        confidence_text = f"{confidence:.0f}%"
    else:
        confidence_text = "—"

    st.markdown(
        f"""
        <div class="analysis-card">
            <div class="panel-title">🧠 تحلیل ساختاری نامه</div>
            <div class="kv"><div class="kv-label">خلاصه نامه</div><div class="kv-value">{escape(getattr(analysis, 'email_summary', '—'))}</div></div>
            <div class="kv"><div class="kv-label">نوع درخواست</div><div class="kv-value">{escape(getattr(analysis, 'intent', '—'))}</div></div>
            <div class="kv"><div class="kv-label">حوزه علمی</div><div class="kv-value">{escape(getattr(analysis, 'academic_field', '—'))}</div></div>
            <div class="kv"><div class="kv-label">اعتماد مدل</div><div class="kv-value">{escape(confidence_text)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### کلمات کلیدی")
    render_chips(as_list(getattr(analysis, "keywords", [])))
    st.markdown("#### موضوعات پژوهشی")
    render_chips(as_list(getattr(analysis, "research_topics", [])))


def render_person_card(rank: int, row: pd.Series) -> None:
    person_name = row.get("person_name", row.get("name", "نامشخص"))
    score = row.get("total_score", row.get("score", 0))
    article_count = row.get("related_articles_count", 0)
    avg_relevance = row.get("avg_relevance", 0)
    top_articles = as_list(row.get("top_articles", []))

    st.markdown(
        f"""
        <div class="person-card">
            <div class="person-name">{rank}. {escape(person_name)}</div>
            <div class="kv"><div class="kv-label">امتیاز نهایی</div><div class="kv-value">{escape(format_number(score))}</div></div>
            <div class="kv"><div class="kv-label">مقالات مرتبط</div><div class="kv-value">{escape(article_count)}</div></div>
            <div class="kv"><div class="kv-label">میانگین شباهت</div><div class="kv-value">{escape(format_number(avg_relevance))}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if top_articles:
        with st.expander("مشاهده مقاله‌های شاخص این فرد"):
            for title in top_articles:
                st.markdown(f"- {escape(title)}")


def upload_signature(uploaded_file: Any) -> str:
    data = uploaded_file.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    return f"{uploaded_file.name}:{len(data)}:{digest}"


def visible_article_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "id",
        "title",
        "published_year",
        "journal_name",
        "citation",
        "fwci",
        "relevance_score",
    ]
    return [column for column in preferred if column in df.columns]


# -----------------------------------------------------------------------------
# Application startup
# -----------------------------------------------------------------------------
render_styles()

st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-kicker">سامانه مبتنی بر LLM و داده‌های پژوهشی دانشگاه</div>
        <div class="hero-title">ارجاع هوشمند نامه‌ها به <span>افراد دانشگاهی مرتبط</span></div>
        <p class="hero-desc">
            نامه را به‌صورت متن، سند یا تصویر وارد کنید. سامانه متن را استخراج می‌کند، موضوع را با LLM تحلیل می‌کند،
            مقاله‌های مرتبط را می‌یابد و نویسندگان داخلی را بر اساس شواهد پژوهشی رتبه‌بندی می‌کند.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_load_data() -> dict[str, pd.DataFrame]:
    return load_university_data("data")


try:
    llm_service = LLMService()
except Exception as exc:  # noqa: BLE001
    st.error(f"خطا در راه‌اندازی سرویس LLM: {exc}")
    st.stop()

try:
    data = cached_load_data()
except Exception as exc:  # noqa: BLE001
    st.error(f"خطا در خواندن دیتاست‌ها: {exc}")
    st.stop()

persons_df = data["persons"]
articles_df = data["articles"]
article_authors_df = data["article_authors"]

if "letter_text" not in st.session_state:
    st.session_state.letter_text = ""
if "uploaded_signature" not in st.session_state:
    st.session_state.uploaded_signature = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "relevant_articles" not in st.session_state:
    st.session_state.relevant_articles = None
if "ranked_people" not in st.session_state:
    st.session_state.ranked_people = None

use_mock = os.getenv("USE_MOCK_LLM", "true").strip().lower() in {"1", "true", "yes", "on"}
ocr_available, ocr_message = get_ocr_status()

with st.sidebar:
    st.markdown("## وضعیت سامانه")
    st.markdown(
        f"""
        <div class="status-row">
            <span><span class="status-dot {'warn' if use_mock else 'ok'}"></span>سرویس تحلیل</span>
            <strong>{'Mock' if use_mock else 'OpenRouter'}</strong>
        </div>
        <div class="status-row">
            <span><span class="status-dot {'ok' if ocr_available else 'warn'}"></span>خواندن تصویر</span>
            <strong>{'فعال' if ocr_available else 'نیازمند تنظیم'}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(ocr_message)

    st.divider()
    st.markdown("### داده‌های بارگذاری‌شده")
    st.metric("افراد دانشگاهی", f"{len(persons_df):,}")
    st.metric("مقالات", f"{len(articles_df):,}")
    st.metric("روابط نویسندگی", f"{len(article_authors_df):,}")

    st.divider()
    st.caption("کلید API در رابط کاربری نمایش داده نمی‌شود و باید فقط در فایل .env یا Secrets سرور نگهداری شود.")

st.markdown(
    f"""
    <div class="stat-grid">
        <div class="stat-card"><div class="stat-label">افراد دانشگاهی</div><div class="stat-value">{len(persons_df):,}</div></div>
        <div class="stat-card"><div class="stat-label">مقالات قابل جست‌وجو</div><div class="stat-value">{len(articles_df):,}</div></div>
        <div class="stat-card"><div class="stat-label">روابط مقاله و نویسنده</div><div class="stat-value">{len(article_authors_df):,}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Input and settings
# -----------------------------------------------------------------------------
input_col, settings_col = st.columns([1.7, 1], gap="large")

with input_col:
    render_panel_header(
        "۱",
        "ورود نامه",
        "متن نامه را تایپ کنید یا فایل PDF، Word، TXT و تصویر را بارگذاری کنید. متن استخراج‌شده قبل از تحلیل قابل ویرایش است.",
    )

    letter_mode = st.radio(
        "روش ورود نامه",
        ["وارد کردن متن", "بارگذاری فایل یا تصویر"],
        horizontal=True,
        key="letter_mode",
    )

    if letter_mode == "وارد کردن متن":
        st.text_area(
            "متن نامه",
            height=330,
            placeholder="متن نامه را اینجا وارد کنید...",
            key="letter_text",
        )
    else:
        uploaded_letter = st.file_uploader(
            "فایل نامه را انتخاب کنید",
            type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "webp", "tif", "tiff", "bmp"],
            help="برای تصویر از OCR استفاده می‌شود. حداکثر حجم پیش‌فرض ۱۵ مگابایت است.",
            key="letter_upload",
        )

        if uploaded_letter is not None:
            if is_image_filename(uploaded_letter.name):
                st.image(uploaded_letter, caption="پیش‌نمایش تصویر نامه", use_container_width=True)

            signature = upload_signature(uploaded_letter)
            if signature != st.session_state.uploaded_signature:
                try:
                    with st.spinner("در حال استخراج متن فایل..."):
                        extracted_text = extract_text_from_upload(uploaded_letter)
                    st.session_state.letter_text = extracted_text
                    st.session_state.uploaded_signature = signature
                    if extracted_text:
                        st.success("متن نامه با موفقیت استخراج شد. لطفاً قبل از تحلیل آن را بررسی کنید.")
                    else:
                        st.warning("متنی از فایل استخراج نشد.")
                except DocumentReadError as exc:
                    st.error(str(exc))

            st.text_area(
                "متن استخراج‌شده و قابل ویرایش",
                height=300,
                key="letter_text",
            )

    clear_col, length_col = st.columns([1, 2])
    def clear_letter_state():
        st.session_state.letter_text = ""
        st.session_state.uploaded_signature = None
        st.session_state.analysis_result = None
        st.session_state.relevant_articles = None
        st.session_state.ranked_people = None
        if "letter_upload" in st.session_state:
            del st.session_state["letter_upload"]
    with clear_col:
        st.button(
            "پاک کردن متن",
            use_container_width=True,
            on_click=clear_letter_state,
        )
    with length_col:
        st.caption(f"تعداد نویسه‌های فعلی: {len(st.session_state.letter_text):,}")

with settings_col:
    render_panel_header(
        "۲",
        "تنظیمات تحلیل",
        "تعداد مقاله‌های بازیابی‌شده و افراد پیشنهادی را مشخص کنید. مقادیر بیشتر زمان پردازش را افزایش می‌دهد.",
    )

    top_articles = st.number_input(
        "تعداد مقاله‌های مرتبط",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        key="top_articles",
    )
    top_people = st.number_input(
        "تعداد افراد پیشنهادی",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        key="top_people",
    )

    st.markdown(
        """
        <div class="panel" style="margin-top:14px;">
            <div class="panel-title">مسیر پردازش</div>
            <div class="panel-subtitle">
                ۱. استخراج متن<br>
                ۲. تحلیل LLM<br>
                ۳. بازیابی مقاله‌ها<br>
                ۴. اتصال مقاله به نویسندگان<br>
                ۵. رتبه‌بندی افراد
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    analyze_clicked = st.button(
        "تحلیل نامه و یافتن افراد مرتبط",
        type="primary",
        use_container_width=True,
    )

# -----------------------------------------------------------------------------
# Analysis pipeline
# -----------------------------------------------------------------------------
if analyze_clicked:
    letter_text = st.session_state.letter_text.strip()
    if not letter_text:
        st.warning("لطفاً ابتدا متن نامه را وارد یا فایل نامه را بارگذاری کنید.")
    else:
        try:
            with st.status("پردازش نامه آغاز شد...", expanded=True) as status:
                st.write("تحلیل موضوع و هدف نامه با LLM")
                analysis = llm_service.analyze_letter(letter_text)

                st.write("جست‌وجوی مقاله‌های مرتبط")
                relevant_articles = retrieve_relevant_articles(
                    analysis=analysis,
                    articles_df=articles_df,
                    top_n=int(top_articles),
                )

                st.write("اتصال مقاله‌ها به نویسندگان و رتبه‌بندی افراد")
                ranked_people = rank_people(
                    relevant_articles=relevant_articles,
                    article_authors=article_authors_df,
                    persons=persons_df,
                ).head(int(top_people))

                st.session_state.analysis_result = analysis
                st.session_state.relevant_articles = relevant_articles
                st.session_state.ranked_people = ranked_people
                status.update(label="تحلیل با موفقیت کامل شد.", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001
            st.error(f"پردازش نامه با خطا روبه‌رو شد: {exc}")
            with st.expander("جزئیات فنی خطا"):
                st.exception(exc)

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
analysis = st.session_state.analysis_result
relevant_articles = st.session_state.relevant_articles
ranked_people = st.session_state.ranked_people

if analysis is not None:
    st.markdown("---")
    render_panel_header(
        "۳",
        "نتایج تحلیل و پیشنهاد",
        "خروجی زیر بر اساس تحلیل نامه، شباهت محتوایی مقاله‌ها و رابطه واقعی مقاله با نویسندگان داخلی تولید شده است.",
    )

    analysis_col, people_col = st.columns([1, 1.25], gap="large")

    with analysis_col:
        render_analysis(analysis)

    with people_col:
        st.markdown("### افراد پیشنهادی")
        if ranked_people is None or ranked_people.empty:
            st.warning("فرد مرتبطی پیدا نشد؛ نتیجه نیازمند بررسی انسانی است.")
        else:
            for position, (_, row) in enumerate(ranked_people.iterrows(), start=1):
                render_person_card(position, row)

    if relevant_articles is not None and not relevant_articles.empty:
        st.markdown("### شواهد پژوهشی")
        with st.expander("مشاهده مقاله‌های مرتبط", expanded=False):
            columns = visible_article_columns(relevant_articles)
            if columns:
                display_df = relevant_articles[columns].copy()
                if "relevance_score" in display_df.columns:
                    display_df["relevance_score"] = display_df["relevance_score"].round(2)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.dataframe(relevant_articles, use_container_width=True, hide_index=True)
else:
    st.markdown(
        """
        <div class="panel" style="text-align:center; padding:30px;">
            <div class="panel-title" style="justify-content:center;">هنوز تحلیلی انجام نشده است</div>
            <div class="panel-subtitle" style="text-align:center !important;">
                نامه را وارد کنید و روی دکمه «تحلیل نامه و یافتن افراد مرتبط» بزنید.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
