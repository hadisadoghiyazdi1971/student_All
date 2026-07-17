# راهنمای مهاجرت از نسخه قبلی

1. از پروژه فعلی نسخه پشتیبان بگیرید.
2. فایل‌های این بسته را جایگزین فایل‌های هم‌نام کنید.
3. فایل‌های اصلی `persons.csv`، `articles.csv` و `article_authors.csv` خودتان را در پوشه `data` قرار دهید.
4. دو فایل taxonomy موجود در `data/taxonomy` را حفظ کنید.
5. محیط مجازی قدیمی را ارسال یا کپی نکنید. یک محیط جدید بسازید و `requirements.txt` را نصب کنید.
6. متغیرهای OpenRouter دیگر در مسیر اصلی استفاده نمی‌شوند. `.env.example` را به `.env` تبدیل و مسیر Ollama را تنظیم کنید.
7. مدل را دریافت کنید:

```bash
ollama pull qwen3:8b
```

8. قبل از اجرای رابط کاربری وضعیت را بررسی کنید:

```bash
python startup_check.py
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

9. برنامه را اجرا کنید:

```bash
streamlit run app.py
```

## فایل‌های قدیمی

- `professor_data_loader.py` و تطبیق مستقیم پروفایل استاد در جریان اصلی استفاده نمی‌شوند.
- `article_retriever.py` قبلی مبتنی بر TF-IDF بوده و باید کامل جایگزین شود.
- `llm_service.py` قبلی مبتنی بر OpenRouter بوده و باید کامل جایگزین شود.
- نسخه قبلی `app.py` برای مقایسه در `legacy/app_v1.py` قرار دارد.
