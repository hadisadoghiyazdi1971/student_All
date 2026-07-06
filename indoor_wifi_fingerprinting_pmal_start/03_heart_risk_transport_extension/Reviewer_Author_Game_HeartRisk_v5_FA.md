# چرخه داور-نویسنده برای v5

## ایراد داور به v4
Heart of Risk در ابتدای مقاله می‌آید، اما قضایا و الگوریتم هنوز از نسخه قبلی KME/OT می‌آیند. ریسک و OT واقعاً در یک مسئله واحد ترکیب نشده‌اند.

## اصلاح v5
در v5، سناریوی آینده به صورت `(y_q, \tilde y, Q_{1:N})` تعریف شد. بنابراین OT ambiguity مستقیماً داخل آینده قرار گرفت. سپس query risk به صورت risk functional روی post-query loss law تعریف شد.

## ایراد داور دوم
CVaR و coherent risk فقط نام برده شده‌اند؛ معلوم نیست چگونه با supremum روی Wasserstein ball یکی می‌شوند.

## اصلاح v5
قضیه `Dual risk-transport representation` اضافه شد:

`risk = sup over future-scenario reweightings`

و همزمان:

`sup over Q_i in Wasserstein balls`

پس دو robust layer داریم: tail-risk layer و measure-ambiguity layer.

## ایراد داور سوم
KME فقط یک surrogate است؛ چرا معتبر است؟

## اصلاح v5
قضیه `Heart-of-Risk transport envelope` و corollary `Envelope preservation under KME calibration` نشان می‌دهند که KME envelope نه فقط mean loss، بلکه تمام tail reweightingهای مجاز در risk envelope را upper-bound می‌کند.

## وضعیت فعلی
v5 از نظر نظری بسیار پیوسته‌تر است. برای ارسال نهایی باید اجرای کامل ۵۰ تکرار و اضافه شدن baselineهای دقیق PMARLDB/PMARLDS انجام شود.
