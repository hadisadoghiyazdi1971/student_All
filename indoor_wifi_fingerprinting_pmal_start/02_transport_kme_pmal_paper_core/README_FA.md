# نسخه v3 رو به جلو مقاله Transport-Calibrated KME-PMAL

این نسخه بر اساس همان نسخه اصلاحی قبلی ساخته شده است، اما مطالب نظری مفید حذف نشده‌اند؛ بلکه در مسیر جدید مقاله ادغام شده‌اند. هدف این نیست که فصل‌های قدیمی تکرار شوند، بلکه این است که زنجیره زیر واضح و پیوسته شود:

Transport-PMAL master problem → Hilbert/KME relaxation → finite-bag/Bayesian radius → finite-dimensional query rule → executable benchmark suite.

## تغییرات اصلی نسبت به v2

1. افزودن بخش `Forward Integrated Theoretical Spine` برای اتصال مستقیم مسئله PMAL به فرمول transport-calibrated.
2. افزودن گزاره `Operator-preserving relaxation` برای نشان دادن اینکه ترتیب min/max/min/min/sup حفظ می‌شود و فقط مجموعه ابهام از Wasserstein به Hilbert/KME relax می‌شود.
3. افزودن جدول مسیر نظری که نشان می‌دهد اجزای PMAL قدیمی چگونه در مقاله جدید، بدون تکرار خام، نقش جدید می‌گیرند.
4. افزودن قضیه `Local finite-dimensional reduction` برای پیوند دادن نظریه robust envelope به score عملیاتی TC-BKME-PMAL.
5. نگه‌داشتن بسته آزمایش‌های Python و افزودن `run_full_suite.py` و `make_latex_tables.py` برای اجرای ژورنالی و تولید جدول LaTeX.
6. اصلاح فرمول‌های عریض؛ PDF دو بار کامپایل و رندر شد و overfull جدی باقی نمانده است.

## اجرای آزمایش کامل

```bash
cd code
python run_full_suite.py --repeats 50 --budget 50 --outdir ../results_full --include-20news
```

برای تولید جدول LaTeX از نتایج:

```bash
python make_latex_tables.py --results ../results_full --out ../results_full/tables_for_paper.tex
```

## فایل اصلی

- `Transport_Calibrated_KME_PMAL_Forward_v3.tex`
- `Transport_Calibrated_KME_PMAL_Forward_v3.pdf`
