# نسخه v5: ادغام واقعی Heart of Risk با OT/KME-PMAL

در این نسخه، Heart of Risk دیگر یک مقدمه یا لایه جدا نیست. ساختار اصلی مقاله تغییر کرده است:

- query به عنوان action تعریف شده است؛
- آینده شامل label نامعلوم، completion برچسب‌های باقی‌مانده، و توزیع‌های واقعی محتمل `Q_i` است؛
- ریسک روی توزیع آینده‌ی post-query loss اعمال می‌شود؛
- OT/Wasserstein ambiguity داخل فضای سناریوها قرار گرفته است؛
- KME/Bayesian KME به عنوان envelope حل‌پذیر همین مسئله معرفی شده است؛
- قضیه dual risk-transport اضافه شده تا نشان دهد CVaR/coherent risk و OT ambiguity در یک فرمول واحد ترکیب می‌شوند.

فایل جدید مهم:

- `HeartRisk_OT_KME_PMAL_v5.tex`
- `HeartRisk_OT_KME_PMAL_v5.pdf`
- `code/heart_risk_transport_core.py`

نکته: اجرای کامل ۵۰ تکرار هنوز انجام نشده است؛ اما کد و ساختار معیارهای Heart-of-Risk برای اجرای کامل آماده شده‌اند.
