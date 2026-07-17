# اجرای اولیه

1. فایل‌های واقعی دانشگاه را در `data/` قرار دهید:
   - `persons.csv`
   - `articles.csv`
   - `article_authors.csv`

2. محیط Python 3.11 بسازید و وابستگی‌ها را نصب کنید:

```bash
python -m venv .venv
```

ویندوز:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

لینوکس:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

3. Ollama و مدل را آماده کنید:

```bash
ollama pull qwen3:8b
ollama serve
```

4. داده‌ها و سرویس‌ها را بررسی کنید:

```bash
python startup_check.py
```

5. برای جلوگیری از تاخیر اولین درخواست نمایه‌ها را از قبل بسازید:

```bash
python scripts/prebuild_embeddings.py
```

6. برنامه را اجرا کنید:

```bash
streamlit run app.py
```

برای اجرای تست‌ها ابتدا `requirements-dev.txt` را نصب کنید.
