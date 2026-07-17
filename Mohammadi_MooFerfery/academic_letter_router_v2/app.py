from __future__ import annotations

import hashlib
import logging
import os
import html
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analysis_pipeline import AnalysisPipeline
from article_retriever import retrieve_relevant_articles
from config import PATHS, SETTINGS_PROFILE
from data_loader import load_university_data
from document_reader import DocumentReadError, extract_text_from_upload, get_ocr_status, is_image_filename
from embedding_service import EmbeddingService
from keyword_service import KeywordService
from llm_service import OllamaService
from logging_config import configure_logging
from models import AnalysisResult, AppSettings, LetterProposal
from person_ranker import rank_people
from proposal_service import ProposalService, proposal_to_docx_bytes
from settings_service import SettingsStore
from taxonomy_service import TaxonomyService


load_dotenv(override=False)
configure_logging(PATHS.runtime_dir / "logs")
logger = logging.getLogger(__name__)
st.set_page_config(
    page_title="سامانه ارجاع هوشمند نامه‌ها",
    page_icon="📨",
    layout="wide",
    initial_sidebar_state="expanded",
)


def escape(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def render_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg:#07111f; --surface:rgba(15,31,50,.84); --soft:rgba(255,255,255,.055);
            --border:rgba(148,163,184,.20); --text:#f8fafc; --muted:#a9b7c9;
            --primary:#7c3aed; --cyan:#0891b2; --success:#10b981; --warning:#f59e0b;
        }
        html,body,.stApp,[data-testid="stAppViewContainer"] {
            direction:rtl!important; text-align:right!important;
            font-family:Vazirmatn,Tahoma,Arial,sans-serif!important;
        }
        .stApp {
            background:radial-gradient(circle at 88% 4%,rgba(124,58,237,.22),transparent 25%),
                       radial-gradient(circle at 12% 18%,rgba(8,145,178,.17),transparent 23%),
                       linear-gradient(145deg,#050b14 0%,#07111f 52%,#0a1727 100%);
            color:var(--text);
        }
        header[data-testid="stHeader"]{background:transparent!important}
        #MainMenu,footer,[data-testid="stToolbar"]{visibility:hidden}
        .block-container{max-width:1400px;padding-top:1.1rem;padding-bottom:4rem}
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li, [data-testid="stCaptionContainer"],
        [data-testid="stAlert"],label,p,span,h1,h2,h3,h4,h5,h6 {
            direction:rtl!important;text-align:right!important
        }
        [data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input{direction:rtl!important;text-align:right!important}
        [data-testid="stHorizontalBlock"]{direction:rtl}
        section[data-testid="stSidebar"]{background:rgba(5,13,24,.97);border-left:1px solid var(--border)}
        section[data-testid="stSidebar"] *{direction:rtl!important;text-align:right!important}
        .hero{border:1px solid var(--border);border-radius:30px;padding:30px 34px;margin-bottom:20px;
              background:linear-gradient(120deg,rgba(124,58,237,.19),rgba(8,145,178,.08)),rgba(15,31,50,.72);
              box-shadow:0 24px 70px rgba(0,0,0,.28)}
        .hero-kicker{display:inline-flex;border:1px solid rgba(167,139,250,.28);background:rgba(124,58,237,.10);
                     color:#ddd6fe;border-radius:999px;padding:7px 12px;font-size:13px;font-weight:700;margin-bottom:12px}
        .hero-title{color:#fff;font-size:clamp(29px,4vw,50px);font-weight:800;line-height:1.35;margin:0 0 8px}
        .hero-title span{color:#67e8f9}.hero-desc{color:#cbd5e1;line-height:2;font-size:16px;max-width:980px}
        .panel,.card{border:1px solid var(--border);background:var(--surface);border-radius:22px;padding:20px;
                     box-shadow:0 16px 48px rgba(0,0,0,.16);margin-bottom:14px}
        .panel-title{color:#fff;font-size:20px;font-weight:800;margin-bottom:6px}.panel-sub{color:var(--muted);line-height:1.9}
        .stat-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:20px}
        .stat{border:1px solid var(--border);background:var(--soft);border-radius:18px;padding:16px 18px}
        .stat-label{color:var(--muted);font-size:13px}.stat-value{color:#fff;font-size:24px;font-weight:800}
        .chip{display:inline-flex;padding:6px 10px;margin:3px;border-radius:999px;border:1px solid rgba(103,232,249,.22);
              background:rgba(8,145,178,.10);color:#cffafe;font-size:13px}
        .kv{display:grid;grid-template-columns:155px 1fr;gap:10px;padding:8px 0;border-bottom:1px dashed rgba(148,163,184,.16)}
        .kv:last-child{border-bottom:none}.kv-label{color:#a5b4fc;font-weight:700}.kv-value{color:#e2e8f0;line-height:1.9}
        .score{display:inline-flex;border-radius:999px;padding:5px 10px;background:rgba(16,185,129,.12);color:#a7f3d0;
               border:1px solid rgba(16,185,129,.25);font-weight:700}
        .status{display:flex;justify-content:space-between;gap:10px;padding:11px 13px;border:1px solid var(--border);
                background:rgba(255,255,255,.035);border-radius:15px;margin-bottom:9px}
        .dot{width:9px;height:9px;display:inline-block;border-radius:50%;margin-left:7px}.ok{background:#34d399}.warn{background:#fbbf24}
        div[data-testid="stFileUploader"]{border:1px dashed rgba(103,232,249,.34);border-radius:18px;background:rgba(8,145,178,.055);padding:8px}
        .stButton>button{min-height:47px;border:none;border-radius:15px;font-weight:800;background:linear-gradient(95deg,#7c3aed,#0891b2);
                         color:#fff;box-shadow:0 12px 30px rgba(8,145,178,.18)}
        div[data-testid="stAlert"],div[data-testid="stExpander"]{border-radius:16px}
        [data-testid="stDataFrame"]{direction:ltr!important;text-align:left!important;border-radius:16px;overflow:hidden}
        code,pre{direction:ltr!important;text-align:left!important}
        @media(max-width:900px){.stat-grid{grid-template-columns:1fr}.hero{padding:23px 20px}.kv{grid-template-columns:1fr;gap:3px}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='panel'><div class='panel-title'>{escape(icon)} {escape(title)}</div>"
        f"<div class='panel-sub'>{escape(subtitle)}</div></div>",
        unsafe_allow_html=True,
    )


def render_chips(items: list[str]) -> None:
    if not items:
        st.markdown("<span class='chip'>مورد معتبری یافت نشد</span>", unsafe_allow_html=True)
        return
    st.markdown("".join(f"<span class='chip'>{escape(item)}</span>" for item in items), unsafe_allow_html=True)


def upload_signature(uploaded_file: Any) -> str:
    data = uploaded_file.getvalue()
    return f"{uploaded_file.name}:{len(data)}:{hashlib.sha256(data).hexdigest()}"


@st.cache_data(show_spinner=False)
def cached_load_data(data_dir: str) -> dict[str, pd.DataFrame]:
    return load_university_data(data_dir)


@st.cache_resource(show_spinner=False)
def cached_embedding_service(model_name: str) -> EmbeddingService:
    return EmbeddingService(model_name=model_name, cache_dir=PATHS.cache_dir)


@st.cache_resource(show_spinner=False)
def cached_keyword_service(model_name: str) -> KeywordService:
    return KeywordService(cached_embedding_service(model_name))


@st.cache_resource(show_spinner=False)
def cached_taxonomy_service(model_name: str) -> TaxonomyService:
    return TaxonomyService(
        field_names_path=PATHS.taxonomy_dir / "university_37_field_names_only.xlsx",
        medical_fields_path=PATHS.taxonomy_dir / "university_37_medical_fields_only.xlsx",
        embedding_service=cached_embedding_service(model_name),
    )


@st.cache_data(ttl=15, show_spinner=False)
def cached_ollama_status(model_name: str) -> tuple[bool, str, list[str]]:
    return OllamaService(model=model_name).status()


def initialize_state(settings: AppSettings) -> None:
    defaults = settings.model_dump()
    defaults.update(
        {
            "letter_text": "",
            "uploaded_signature": None,
            "analysis_result": None,
            "relevant_articles": None,
            "ranked_people": None,
            "proposal": None,
            "proposal_text": "",
            "proposal_subject": "",
            "proposal_recipient": "",
            "proposal_audience": "field",
        }
    )
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_results() -> None:
    st.session_state.analysis_result = None
    st.session_state.relevant_articles = None
    st.session_state.ranked_people = None
    st.session_state.proposal = None
    st.session_state.proposal_text = ""
    st.session_state.proposal_subject = ""
    st.session_state.proposal_recipient = ""


def clear_letter() -> None:
    st.session_state.letter_text = ""
    st.session_state.uploaded_signature = None
    clear_results()
    if "letter_upload" in st.session_state:
        del st.session_state["letter_upload"]


def render_analysis(result: AnalysisResult) -> None:
    analysis = result.analysis
    confidence = f"{analysis.confidence * 100:.0f}%"
    st.markdown(
        f"""
        <div class='card'>
          <div class='panel-title'>🧠 تحلیل ساختاری نامه</div>
          <div class='kv'><div class='kv-label'>خلاصه نامه</div><div class='kv-value'>{escape(analysis.email_summary or '—')}</div></div>
          <div class='kv'><div class='kv-label'>نوع درخواست</div><div class='kv-value'>{escape(analysis.intent)}</div></div>
          <div class='kv'><div class='kv-label'>حوزه علمی</div><div class='kv-value'>{escape(analysis.academic_field)}</div></div>
          <div class='kv'><div class='kv-label'>روش تحلیل</div><div class='kv-value'>{escape(analysis.analysis_method)}</div></div>
          <div class='kv'><div class='kv-label'>اعتماد مدل</div><div class='kv-value'><span class='score'>{confidence}</span></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### کلمات کلیدی نهایی")
    render_chips(analysis.keywords)
    st.markdown("#### موضوعات پژوهشی")
    render_chips(analysis.research_topics)

    if result.keyword_candidates:
        with st.expander("جزئیات خروجی KeyBERT"):
            keyword_df = pd.DataFrame([item.model_dump() for item in result.keyword_candidates])
            keyword_df["score"] = (keyword_df["score"] * 100).round(1)
            keyword_df.columns = ["عبارت", "امتیاز", "منبع"]
            st.dataframe(keyword_df, use_container_width=True, hide_index=True)


def render_fields(result: AnalysisResult) -> None:
    st.markdown("### حوزه‌های منطبق با ساختار دانشگاه")
    if not result.field_matches:
        st.warning("هیچ حوزه‌ای از taxonomy دانشگاه از حداقل امتیاز عبور نکرد. نتیجه باید انسانی بررسی شود.")
        return
    selected = set(result.analysis.selected_fields)
    for index, item in enumerate(result.field_matches, start=1):
        faculty = "، ".join(item.faculties[:3]) or "—"
        department = "، ".join(item.departments[:3]) or "—"
        selected_badge = " <span class='score'>انتخاب نهایی</span>" if item.canonical_name in selected else ""
        st.markdown(
            f"""
            <div class='card'>
              <div class='panel-title'>{index}. {escape(item.canonical_name)} <span class='score'>{item.score*100:.1f}</span>{selected_badge}</div>
              <div class='kv'><div class='kv-label'>دانشکده</div><div class='kv-value'>{escape(faculty)}</div></div>
              <div class='kv'><div class='kv-label'>گروه آموزشی</div><div class='kv-value'>{escape(department)}</div></div>
              <div class='kv'><div class='kv-label'>افراد ثبت‌شده</div><div class='kv-value'>{item.people_count}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_person_card(position: int, row: pd.Series) -> None:
    st.markdown(
        f"""
        <div class='card'>
          <div class='panel-title'>{position}. {escape(row.get('person_name','نامشخص'))} <span class='score'>{float(row.get('routing_score',0)):.1f}</span></div>
          <div class='kv'><div class='kv-label'>مقالات مرتبط</div><div class='kv-value'>{int(row.get('related_articles_count',0))}</div></div>
          <div class='kv'><div class='kv-label'>میانگین ارتباط</div><div class='kv-value'>{float(row.get('avg_relevance',0)):.1f}</div></div>
          <div class='kv'><div class='kv-label'>نویسنده مسئول</div><div class='kv-value'>{int(row.get('corresponding_author_count',0))} مقاله</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    reasons = as_list(row.get("explanations", []))
    articles = as_list(row.get("top_articles", []))
    with st.expander("چرا این فرد پیشنهاد شده است؟"):
        for reason in reasons:
            st.markdown(f"- {escape(reason)}")
        if articles:
            st.markdown("**مقاله‌های شاخص:**")
            for title in articles:
                st.markdown(f"- {escape(title)}")


def current_settings() -> AppSettings:
    return AppSettings(
        top_articles=int(st.session_state.top_articles),
        top_people=int(st.session_state.top_people),
        keyword_count=int(st.session_state.keyword_count),
        keyword_min_score=float(st.session_state.keyword_min_score),
        keyword_diversity=float(st.session_state.keyword_diversity),
        use_keybert=bool(st.session_state.use_keybert),
        use_ollama=bool(st.session_state.use_ollama),
        ollama_model=str(st.session_state.ollama_model).strip() or "qwen3:8b",
        embedding_model=str(st.session_state.embedding_model).strip() or "BAAI/bge-m3",
        field_candidate_count=int(st.session_state.field_candidate_count),
        field_min_score=float(st.session_state.field_min_score),
        proposal_tone=str(st.session_state.proposal_tone),
        proposal_mode=str(st.session_state.proposal_mode),
    )


render_styles()
settings_store = SettingsStore(PATHS.settings_db)
persisted_settings = settings_store.load(SETTINGS_PROFILE)
initialize_state(persisted_settings)

st.markdown(
    """
    <div class='hero'>
      <div class='hero-kicker'>سامانه محلی مبتنی بر KeyBERT، Ollama و شواهد پژوهشی دانشگاه</div>
      <div class='hero-title'>ارجاع هوشمند نامه‌ها به <span>افراد دانشگاهی مرتبط</span></div>
      <div class='hero-desc'>نامه را وارد کنید. سامانه کلمات کلیدی را کنترل‌شده استخراج می‌کند، حوزه را با taxonomy واقعی دانشگاه تطبیق می‌دهد، مقاله‌ها را معنایی بازیابی می‌کند، افراد را با دلیل رتبه‌بندی می‌کند و یک پیش‌نویس قابل ویرایش پیشنهاد می‌دهد.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    data = cached_load_data(str(PATHS.data_dir))
except Exception as exc:  # noqa: BLE001
    st.error(f"داده‌های اصلی دانشگاه بارگذاری نشدند: {exc}")
    st.info("فایل‌های persons.csv، articles.csv و article_authors.csv را در پوشه data قرار دهید. نمونه ساختار در data/examples موجود است.")
    st.stop()

persons_df = data["persons"]
articles_df = data["articles"]
article_authors_df = data["article_authors"]
ocr_available, ocr_message = get_ocr_status()
ollama_ok, ollama_message, _ = cached_ollama_status(str(st.session_state.ollama_model))

with st.sidebar:
    st.markdown("## وضعیت سامانه")
    st.markdown(
        f"<div class='status'><span><span class='dot {'ok' if ollama_ok else 'warn'}'></span>Ollama</span><strong>{'آماده' if ollama_ok else 'بررسی شود'}</strong></div>",
        unsafe_allow_html=True,
    )
    st.caption(ollama_message)
    st.markdown(
        f"<div class='status'><span><span class='dot {'ok' if ocr_available else 'warn'}'></span>OCR</span><strong>{'آماده' if ocr_available else 'اختیاری'}</strong></div>",
        unsafe_allow_html=True,
    )
    st.caption(ocr_message)
    st.divider()
    st.metric("افراد دانشگاهی", f"{len(persons_df):,}")
    st.metric("مقالات", f"{len(articles_df):,}")
    st.metric("روابط نویسندگی", f"{len(article_authors_df):,}")
    st.divider()
    st.caption("پردازش LLM و embedding به صورت محلی انجام می‌شود. تنظیمات انتخابی در SQLite ذخیره می‌شوند.")
    if os.getenv("EMBEDDING_DISK_CACHE", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        article_limit = max(0, int(os.getenv("ARTICLE_CORPUS_LIMIT", "0")))
        st.warning(
            "حالت تست بدون cache فعال است"
            + (f"؛ حداکثر {article_limit:,} مقاله بررسی می‌شود." if article_limit else ".")
        )

st.markdown(
    f"""
    <div class='stat-grid'>
      <div class='stat'><div class='stat-label'>افراد دانشگاهی</div><div class='stat-value'>{len(persons_df):,}</div></div>
      <div class='stat'><div class='stat-label'>مقالات قابل جست‌وجو</div><div class='stat-value'>{len(articles_df):,}</div></div>
      <div class='stat'><div class='stat-label'>روابط مقاله و نویسنده</div><div class='stat-value'>{len(article_authors_df):,}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

input_col, settings_col = st.columns([1.65, 1], gap="large")
with input_col:
    render_header("۱", "ورود نامه", "متن را تایپ کنید یا PDF، Word، TXT و تصویر بارگذاری کنید. متن استخراج‌شده قابل ویرایش است.")
    mode = st.radio("روش ورود نامه", ["وارد کردن متن", "بارگذاری فایل یا تصویر"], horizontal=True)
    if mode == "وارد کردن متن":
        st.text_area("متن نامه", height=330, placeholder="متن نامه را اینجا وارد کنید...", key="letter_text")
    else:
        uploaded = st.file_uploader(
            "فایل نامه را انتخاب کنید",
            type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "webp", "tif", "tiff", "bmp"],
            key="letter_upload",
        )
        if uploaded is not None:
            if is_image_filename(uploaded.name):
                st.image(uploaded, caption="پیش‌نمایش تصویر", use_container_width=True)
            signature = upload_signature(uploaded)
            if signature != st.session_state.uploaded_signature:
                try:
                    with st.spinner("در حال استخراج متن فایل..."):
                        st.session_state.letter_text = extract_text_from_upload(uploaded)
                    st.session_state.uploaded_signature = signature
                    clear_results()
                    st.success("متن استخراج شد. پیش از تحلیل آن را بررسی کنید.")
                except DocumentReadError as exc:
                    st.error(str(exc))
            st.text_area("متن استخراج‌شده و قابل ویرایش", height=300, key="letter_text")
    clear_col, count_col = st.columns([1, 2])
    with clear_col:
        st.button("پاک کردن", use_container_width=True, on_click=clear_letter)
    with count_col:
        st.caption(f"تعداد نویسه‌ها: {len(st.session_state.letter_text):,}")

with settings_col:
    render_header("۲", "تنظیمات تحلیل", "KeyBERT و Ollama مستقل هستند و می‌توانند همزمان فعال شوند. آخرین انتخاب‌ها ذخیره می‌شوند.")
    st.checkbox("استفاده از KeyBERT", key="use_keybert")
    st.checkbox("استفاده از Ollama", key="use_ollama")
    st.number_input("تعداد حداکثری کلمات کلیدی", min_value=3, max_value=15, step=1, key="keyword_count")
    st.number_input("تعداد مقاله‌های مرتبط", min_value=5, max_value=100, step=5, key="top_articles")
    st.number_input("تعداد افراد پیشنهادی", min_value=1, max_value=20, step=1, key="top_people")
    with st.expander("تنظیمات پیشرفته"):
        st.text_input("مدل Ollama", key="ollama_model")
        st.text_input("مدل embedding", key="embedding_model")
        st.slider("حداقل امتیاز کلمه کلیدی", 0.0, 1.0, step=0.05, key="keyword_min_score")
        st.slider("تنوع MMR", 0.0, 1.0, step=0.05, key="keyword_diversity")
        st.number_input("تعداد حوزه‌های کاندید", min_value=1, max_value=10, step=1, key="field_candidate_count")
        st.slider("حداقل شباهت حوزه", 0.0, 1.0, step=0.05, key="field_min_score")
    analyze_clicked = st.button("تحلیل نامه و یافتن افراد مرتبط", type="primary", use_container_width=True)
    st.caption("اولین اجرا ممکن است مدل embedding را دانلود و نمایه مقاله‌ها را ایجاد کند.")

settings = current_settings()
try:
    if settings != persisted_settings:
        settings_store.save(settings, SETTINGS_PROFILE)
except Exception as exc:  # noqa: BLE001
    st.warning(f"تنظیمات ذخیره نشدند: {exc}")

if analyze_clicked:
    text = st.session_state.letter_text.strip()
    if not text:
        st.warning("ابتدا متن نامه را وارد کنید.")
    elif not settings.use_keybert and not settings.use_ollama:
        st.warning("حداقل یکی از گزینه‌های KeyBERT یا Ollama را فعال کنید.")
    else:
        try:
            with st.status("پردازش نامه آغاز شد...", expanded=True) as status:
                st.write("بارگذاری مدل embedding و taxonomy دانشگاه")
                embedding_service = cached_embedding_service(settings.embedding_model)
                keyword_service = cached_keyword_service(settings.embedding_model)
                taxonomy_service = cached_taxonomy_service(settings.embedding_model)
                ollama_service = OllamaService(model=settings.ollama_model) if settings.use_ollama else None

                st.write("استخراج کلمات کلیدی و تحلیل حوزه")
                pipeline = AnalysisPipeline(
                    keyword_service=keyword_service,
                    taxonomy_service=taxonomy_service,
                    ollama_service=ollama_service,
                )
                result = pipeline.analyze(
                    text,
                    use_keybert=settings.use_keybert,
                    use_ollama=settings.use_ollama,
                    keyword_count=settings.keyword_count,
                    keyword_min_score=settings.keyword_min_score,
                    keyword_diversity=settings.keyword_diversity,
                    field_candidate_count=settings.field_candidate_count,
                    field_min_score=settings.field_min_score,
                )

                st.write("بازیابی معنایی مقاله‌ها")
                relevant_articles = retrieve_relevant_articles(
                    letter_text=text,
                    analysis=result.analysis,
                    articles_df=articles_df,
                    embedding_service=embedding_service,
                    keyword_candidates=result.keyword_candidates,
                    field_matches=result.field_matches,
                    top_n=settings.top_articles,
                )

                st.write("رتبه‌بندی و تولید دلایل قابل توضیح")
                ranked_people = rank_people(
                    relevant_articles,
                    article_authors_df,
                    persons_df,
                ).head(settings.top_people)

                st.session_state.analysis_result = result
                st.session_state.relevant_articles = relevant_articles
                st.session_state.ranked_people = ranked_people
                st.session_state.proposal = None
                st.session_state.proposal_text = ""
                st.session_state.proposal_subject = ""
                st.session_state.proposal_recipient = ""
                status.update(label="تحلیل کامل شد.", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Letter processing failed")
            st.error(f"پردازش نامه با خطا روبه‌رو شد: {exc}")
            with st.expander("جزئیات فنی"):
                st.exception(exc)

result: AnalysisResult | None = st.session_state.analysis_result
relevant_articles: pd.DataFrame | None = st.session_state.relevant_articles
ranked_people: pd.DataFrame | None = st.session_state.ranked_people

if result is None:
    st.markdown("<div class='panel'><div class='panel-title'>هنوز تحلیلی انجام نشده است</div><div class='panel-sub'>نامه را وارد کنید و دکمه تحلیل را بزنید.</div></div>", unsafe_allow_html=True)
    st.stop()

st.markdown("---")
render_header("۳", "نتایج تحلیل و ارجاع", "نتیجه بر اساس متن نامه، taxonomy دانشگاه، مقاله‌ها و نقش واقعی نویسندگان تولید شده است.")
for warning in result.warnings:
    st.warning(warning)

analysis_col, field_col = st.columns([1.05, 1], gap="large")
with analysis_col:
    render_analysis(result)
with field_col:
    render_fields(result)

st.markdown("### افراد پیشنهادی و دلایل انتخاب")
if ranked_people is None or ranked_people.empty:
    st.warning("فرد مرتبطی پیدا نشد. ارجاع باید به صورت انسانی بررسی شود.")
else:
    minimum_routing_score = float(os.getenv("MIN_ROUTING_SCORE", "50"))
    if float(ranked_people.iloc[0].get("routing_score", 0)) < minimum_routing_score:
        st.warning("شواهد پژوهشی برای پیشنهاد افراد ضعیف است. نتیجه باید پیش از ارجاع توسط کارشناس بررسی شود.")
    people_cols = st.columns(2, gap="large")
    for position, (_, row) in enumerate(ranked_people.iterrows(), start=1):
        with people_cols[(position - 1) % 2]:
            render_person_card(position, row)

if relevant_articles is not None and not relevant_articles.empty:
    with st.expander("شواهد پژوهشی: مقاله‌های مرتبط"):
        columns = [
            column for column in (
                "id", "title", "published_year", "journal_name", "citation", "fwci",
                "semantic_score", "keyword_overlap", "relevance_score", "matched_keywords"
            ) if column in relevant_articles.columns
        ]
        display_df = relevant_articles[columns].copy()
        for column in ("semantic_score", "keyword_overlap", "relevance_score"):
            if column in display_df.columns:
                display_df[column] = display_df[column].round(2)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")
render_header("۴", "پیش‌نویس پیشنهادی نامه", "نامه اصلی تغییر نمی‌کند. پیش‌نویس پیشنهادی قابل ویرایش و دانلود است.")
proposal_col, audience_col = st.columns([1, 1], gap="large")
with proposal_col:
    st.radio(
        "نوع تولید",
        options=["rewrite", "new"],
        format_func=lambda value: "بازنویسی نامه فعلی" if value == "rewrite" else "ساخت نامه جدید",
        horizontal=True,
        key="proposal_mode",
    )
    tone_options = ["رسمی و علمی", "بسیار رسمی و اداری", "علمی و پژوهشی", "کوتاه و مستقیم"]
    if st.session_state.proposal_tone not in tone_options:
        st.session_state.proposal_tone = tone_options[0]
    st.selectbox("لحن", tone_options, key="proposal_tone")
with audience_col:
    st.radio(
        "مخاطب",
        options=["field", "person", "generic"],
        format_func=lambda value: {
            "field": "گروه آموزشی مرتبط",
            "person": "فرد اول پیشنهادی",
            "generic": "واحد ارجاع دانشگاه",
        }[value],
        key="proposal_audience",
    )
    generate_proposal = st.button("تولید پیش‌نویس پیشنهادی", use_container_width=True)

if generate_proposal:
    ollama_service = OllamaService(model=settings.ollama_model) if settings.use_ollama else None
    service = ProposalService(ollama_service)
    try:
        with st.spinner("در حال تولید پیش‌نویس..."):
            proposal = service.generate(
                original_text=st.session_state.letter_text,
                result=result,
                ranked_people=ranked_people,
                use_ollama=settings.use_ollama,
                tone=st.session_state.proposal_tone,
                mode=st.session_state.proposal_mode,
                audience=st.session_state.proposal_audience,
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ollama proposal generation failed; using template fallback")
        st.warning(f"Ollama نتوانست پیش‌نویس را تولید کند؛ قالب امن محلی استفاده شد: {exc}")
        proposal = service.generate(
            original_text=st.session_state.letter_text,
            result=result,
            ranked_people=ranked_people,
            use_ollama=False,
            tone=st.session_state.proposal_tone,
            mode=st.session_state.proposal_mode,
            audience=st.session_state.proposal_audience,
        )
    st.session_state.proposal = proposal
    st.session_state.proposal_text = proposal.suggested_letter
    st.session_state.proposal_subject = proposal.suggested_subject
    st.session_state.proposal_recipient = proposal.recipient_title

proposal: LetterProposal | None = st.session_state.proposal
if proposal is not None:
    edited_subject = st.text_input("عنوان پیشنهادی و قابل ویرایش", key="proposal_subject")
    edited_recipient = st.text_input("مخاطب پیشنهادی و قابل ویرایش", key="proposal_recipient")
    edited_text = st.text_area("متن قابل ویرایش", height=390, key="proposal_text")
    st.caption(f"روش تولید: {'Ollama' if proposal.generation_method == 'ollama' else 'قالب امن محلی'}")
    if proposal.improvement_notes:
        with st.expander("نکات بهبود و اطلاعات ناقص"):
            for note in proposal.improvement_notes:
                st.markdown(f"- {escape(note)}")
            if proposal.missing_information:
                st.markdown("**اطلاعاتی که باید تکمیل شوند:**")
                for item in proposal.missing_information:
                    st.markdown(f"- {escape(item)}")
    downloadable = proposal.model_copy(
        update={
            "suggested_subject": edited_subject,
            "recipient_title": edited_recipient,
            "suggested_letter": edited_text,
        }
    )
    txt_content = f"موضوع: {downloadable.suggested_subject}\nمخاطب: {downloadable.recipient_title}\n\n{edited_text}"
    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button("دانلود TXT", txt_content.encode("utf-8-sig"), "letter_proposal.txt", "text/plain", use_container_width=True)
    with download_col2:
        st.download_button(
            "دانلود Word",
            proposal_to_docx_bytes(downloadable),
            "letter_proposal.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
