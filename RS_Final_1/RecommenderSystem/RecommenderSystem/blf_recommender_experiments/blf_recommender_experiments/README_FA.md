# بسته آزمایش‌های Bayesian Latent Filtering Recommender

این بسته برای نسخه جدید مقاله طراحی شده است؛ یعنی مدل را به شکل زیر اجرا می‌کند:

\[
x_u \rightarrow \mathbf z_u=\phi(x_u)
\rightarrow P(c_i\mid x_u)
\rightarrow f(\theta_j\mid c_i)
\rightarrow R_j(x_u)
\rightarrow \widehat r_{ua}
\]

هدف کد این نیست که فقط FCM را اجرا کند؛ بلکه ایده‌ی مقاله را به‌صورت یک توصیه‌گر با فضای پنهان، فیلتر بیزی، مجموعه‌های توصیه و پیش‌بینی امتیاز پیاده‌سازی می‌کند.

## ساختار فایل‌ها

```text
blf_recommender_experiments/
├── blf_recommender.py      # مدل اصلی Bayesian Latent Filtering
├── fuzzy_cmeans.py         # baseline تشخیصی FCM
├── baselines.py            # baselineهای ساده برای مقایسه
├── data_utils.py           # خواندن MovieLens و Yelp
├── metrics.py              # MAE, RMSE, Precision, Recall, F1
├── experiments.py          # اجرای آزمایش‌ها و ساخت جدول‌ها/نمودارها
├── run_all.py              # نقطه شروع اصلی
├── requirements.txt
└── README_FA.md
```

## نصب

در VS Code یا ترمینال:

```bash
pip install -r requirements.txt
```

## اجرای سریع با داده مصنوعی

برای تست اینکه همه‌چیز کار می‌کند:

```bash
python run_all.py --dataset synthetic --out results_synthetic
```

## اجرای MovieLens-100K

ابتدا دیتاست MovieLens-100K را دانلود کنید و پوشه `ml-100k` را در `data/` بگذارید؛ به‌طوری که فایل زیر وجود داشته باشد:

```text
data/ml-100k/u.data
```

سپس:

```bash
python run_all.py --dataset ml100k --data-dir data/ml-100k --out results_ml100k --components 80
```

## اجرای MovieLens-1M

پوشه `ml-1m` را در `data/` بگذارید؛ به‌طوری که فایل زیر وجود داشته باشد:

```text
data/ml-1m/ratings.dat
```

سپس:

```bash
python run_all.py --dataset ml1m --data-dir data/ml-1m --out results_ml1m --components 100
```

## اجرای Yelp

کد انتظار دارد یک فایل CSV با ستون‌های زیر داشته باشید:

```text
user_id,item_id,rating
```

سپس:

```bash
python run_all.py --dataset csv --ratings-csv data/yelp_ratings.csv --out results_yelp --components 110
```

## خروجی‌ها

در پوشه خروجی، این فایل‌ها ساخته می‌شود:

```text
metrics.csv
parameter_sensitivity.csv
convergence.csv
convergence.png
```

## نکته علمی مهم

برای مقاله، خروجی اصلی باید حول این‌ها گزارش شود:

1. MAE و RMSE برای پیش‌بینی rating.
2. Precision, Recall, F1 برای Top-N.
3. نمودار همگرایی منفی log-likelihood مدل.
4. مقایسه تشخیصی با FCM، نه اینکه مقاله به‌عنوان FCM معرفی شود.
5. sensitivity نسبت به تعداد hidden components.

اگر نتایج دقیقاً مثل جدول مقاله نشد، دلیلش معمولاً تفاوت split، preprocessing، seed، و تعریف cold-start است. این کد برای یک protocol تمیز، قابل تکرار و قابل دفاع نوشته شده است؛ برای رسیدن به بهترین عدد باید `--components`، `--latent-dim`، `--beta` و `--init-ratio` را sweep کنید.
