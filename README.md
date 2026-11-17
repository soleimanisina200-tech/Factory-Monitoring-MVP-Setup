# سامانه‌ی پایش کارخانه: شناسایی هویت کارکنان و ایمنی راننده‌ی لیفتراک

راهنمای کامل نصب، اجرا، پیکربندی و ارزیابی. اگر اولین بار است پروژه را باز می‌کنید، بخش‌ها را به ترتیب جلو بروید.

## فهرست
1. این سامانه چه می‌کند
2. پیش‌نیازها
3. ساختار پروژه
4. نصب (یک بار)
5. آماده‌سازی ویدیو
6. کالیبراسیون کف زمین
7. منطقه‌ها و خطوط ورود/خروج (اختیاری)
8. اجرای اصلی
9. خواندن خروجی‌ها
10. گزارش تصمیم‌های هویتی
11. مقایسه‌ی مدل‌ها و ردیاب‌ها
12. مرجع پیکربندی
13. اجرای مستقل ماژول لیفتراک
14. تست‌ها
15. رفع اشکال
16. محدودیت‌ها، مجوزها و نکات ایمنی

---

## ۱. این سامانه چه می‌کند

ویدیوی یک دوربین (فایل یا RTSP) را فریم‌به‌فریم پردازش می‌کند و دو کار انجام می‌دهد:

**الف) شناسایی هویت کارکنان**
- هر فرد یک **ID** می‌گیرد و تا وقتی در تصویر است همان را نگه می‌دارد، حتی اگر پشت مانع پنهان شود یا از کنار دیگران عبور کند.
- وقتی فرد از کادر **خارج** می‌شود و بعداً **برمی‌گردد**، تلاش می‌شود ID قبلی‌اش را پس بگیرد.
- اصل طراحی: **دادن ID اشتباه بدتر از دادن ID جدید است.** در صورت تردید، سیستم ID جدید می‌دهد و اگر بعداً مطمئن شد، دو ID را ادغام می‌کند.

**ب) ایمنی راننده‌ی لیفتراک**
- لیفتراک تشخیص داده می‌شود و راننده‌اش (فردی که در کابین نشسته و با لیفتراک حرکت می‌کند) به آن متصل می‌شود.
- اگر راننده پیاده شود و بیش از فاصله‌ی مجاز (پیش‌فرض ۳ متر) از لیفتراک دور شود، هشدار صادر و ثبت می‌شود.

### جریان پردازش هر فریم
```
فریم ویدیو
   │
   ├─► تشخیص افراد + اسکلت (YOLO-Pose) + ردیاب (ByteTrack / OC-SORT / ...)      app/detection/detector.py
   │
   ├─► مدیر هویت: Re-ID ظاهری (OSNet)، چرخه‌ی حیات، ورود مجدد، ادغام           app/identity/manager.py
   │        └─ لاگ تصمیم‌ها (logs/identity_decisions.jsonl)
   │
   ├─► منطقه‌ها و خطوط ورود/خروج                                                app/zones, app/logic/directional_tracker.py
   │
   ├─► ماژول لیفتراک: تشخیص وسیله (YOLO) + اتصال راننده + فاصله‌ی متری + هشدار  app/vehicle_monitoring/
   │        └─ رویدادها (data/vehicle_safety_events.csv / .db) + اسنپ‌شات
   │
   └─► رسم روی تصویر ← نمایش و/یا ذخیره‌ی ویدیوی خروجی
```

---

## ۲. پیش‌نیازها

| مورد | توضیح |
|---|---|
| سیستم‌عامل | ویندوز ۱۰/۱۱ (روی لینوکس هم کار می‌کند؛ دستورهای این راهنما PowerShell هستند) |
| GPU | کارت NVIDIA با درایور به‌روز. **بدون GPU سیستم حدود ۱ فریم در ثانیه است** و فقط برای تست درستی کار مناسب است. |
| Python | 3.10 (محیط conda داخل پوشه‌ی `env`) |
| اینترنت | فقط برای اولین اجرا (دانلود خودکار وزن مدل‌ها) |
| ویدیو | فایل ویدیو از دوربین ثابت کارخانه، یا آدرس RTSP |

---

## ۳. ساختار پروژه

```
Factory-Monitoring-MVP-Setup/
├── main.py                         ← برنامه‌ی اصلی (افراد + لیفتراک)
├── run_vehicle_monitoring.py       ← اجرای مستقل ماژول لیفتراک (بدون Re-ID/اسکلت)
├── requirements.txt                ← وابستگی‌ها
├── requirements-dev.txt            ← وابستگی‌ها + pytest
├── configs/
│   ├── settings.yaml               ← تنظیمات اصلی (دوربین، مدل‌ها، Re-ID، منطقه‌ها)
│   ├── vehicle_monitoring.yaml     ← تنظیمات لیفتراک، راننده، فاصله، هشدار
│   ├── calibration.yaml            ← (بعد از کالیبراسیون ساخته می‌شود)
│   ├── custom_bytetrack.yaml       ← ردیاب پیش‌فرض
│   └── trackers/                   ← ردیاب‌های جایگزین تنظیم‌شده برای کارخانه
│       ├── factory_ocsort.yaml
│       ├── factory_deepocsort.yaml
│       └── factory_tracktrack.yaml
├── app/
│   ├── detection/detector.py       ← YOLO-Pose + ردیاب
│   ├── identity/manager.py         ← مدیر هویت (منطق اصلی ID)
│   ├── identity/reid_model.py      ← استخراج ویژگی ظاهری (OSNet / OSNet-AIN)
│   ├── logic/geometry.py           ← نقطه‌ی تماس با زمین، تبدیل به متر
│   ├── logic/calibration.py        ← خواندن/نوشتن کالیبراسیون
│   ├── logic/directional_tracker.py← خطوط و دروازه‌های ورود/خروج
│   ├── logic/state_machine.py      ← حضور در منطقه‌ها
│   ├── zones/manager.py
│   ├── streams/video_reader.py     ← خواندن فایل/RTSP
│   └── vehicle_monitoring/         ← لیفتراک، راننده، فاصله، هشدار، داشبورد
├── scripts/
│   ├── calibrate_ground_plane.py   ← کالیبراسیون کف زمین
│   ├── analyze_decisions.py        ← گزارش تصمیم‌های هویتی
│   ├── draw_line.py                ← رسم خط/دروازه‌ی ورود و خروج
│   ├── draw_zone.py                ← رسم منطقه‌ها
│   └── diagnostic.py               ← بررسی سریع نصب
├── tests/                          ← تست‌های خودکار (pytest)
├── weights/                        ← وزن Re-ID (خودکار ساخته می‌شود)
├── data/                           ← ویدیوی ورودی و رویدادهای لیفتراک
├── logs/                           ← لاگ تصمیم‌های هویتی و گزارش‌ها
└── runs/                           ← خروجی آزمایش‌های مقایسه‌ای (بخش ۱۱)
```

پوشه‌های `env/`، `weights/`، `data/`، `logs/`، `runs/`، فایل‌های `configs/exp_*.yaml` و فایل‌های `*.pt` و `*.mp4` در git ثبت نمی‌شوند.

---

## ۴. نصب (یک بار)

### ۴-۱. باز کردن ترمینال و فعال‌کردن محیط
این کار را **هر بار** که ترمینال جدید باز می‌کنید انجام دهید:
```powershell
cd C:\Users\ai\Desktop\factory_mcmot\Factory-Monitoring-MVP-Setup
conda activate C:\Users\ai\Desktop\factory_mcmot\Factory-Monitoring-MVP-Setup\env
```
(اگر پروژه در مسیر دیگری است، مسیرها را عوض کنید.)

### ۴-۲. نصب PyTorch نسخه‌ی CUDA
دستور ساده‌ی `pip install torch` روی ویندوز **نسخه‌ی CPU** نصب می‌کند که برای این سامانه خیلی کند است.
```powershell
pip uninstall -y torch torchvision
```
سپس به https://pytorch.org/get-started/locally/ بروید، گزینه‌های **Stable / Windows / Pip / Python / جدیدترین CUDA** را انتخاب کنید و دستوری که سایت نشان می‌دهد را اجرا کنید. شکل کلی آن:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
```

### ۴-۳. نصب بقیه‌ی وابستگی‌ها
```powershell
pip install -r requirements-dev.txt
```
اگر نصب `torchreid` خطا داد، آن را از مخزن GitHub پروژه‌ی torchreid نصب کنید.

### ۴-۴. بررسی نصب
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import torchreid; print('torchreid OK')"
pytest -q
```

| دستور | خروجی درست |
|---|---|
| اول | `True` و نام کارت گرافیک، مثلاً `True NVIDIA GeForce RTX 4090` |
| دوم | `torchreid OK` |
| سوم | همه‌ی تست‌ها `passed` (هیچ `failed`) |

بررسی سریع دیگر: `python scripts/diagnostic.py`. هشدار «Python below 3.11» در این اسکریپت را می‌توانید نادیده بگیرید؛ پروژه با 3.10 کار می‌کند.

### ۴-۵. وزن مدل‌ها
در **اولین اجرا خودکار** دانلود می‌شوند:

| فایل | کاربرد |
|---|---|
| `yolo11m-pose.pt` | تشخیص افراد و اسکلت |
| `yolo11m.pt` | تشخیص وسیله |
| `weights/osnet_x1_0_msmt17.pt` | ویژگی ظاهری (Re-ID) |

اگر دانلود OSNet از Google Drive شکست خورد، فایل `osnet_x1_0` آموزش‌دیده روی MSMT17 را از
[Torchreid Model Zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO) دستی دانلود کنید و در `weights\osnet_x1_0_msmt17.pt` بگذارید.

**اختیاری:** برای امتحان OSNet-AIN (بخش ۱۱)، فایل `osnet_ain_x1_0` آموزش‌دیده روی MSMT17 (جدول cross-domain همان صفحه) را در `weights\osnet_ain_x1_0_msmt17.pt` بگذارید.

---

## ۵. آماده‌سازی ویدیو

```powershell
New-Item -ItemType Directory -Force data\input
```
فایل ویدیو را در `data\input\` کپی کنید، مثلاً `data\input\clip1.mp4`.

در `configs\settings.yaml` منبع را تنظیم کنید:
```yaml
stream: {source: 'data/input/clip1.mp4', reconnect_delay_sec: 3.0, drop_frames_on_lag: true}
```
- برای دوربین زنده به‌جای مسیر فایل، آدرس RTSP بگذارید: `rtsp://user:pass@ip:554/...`
- `main.py` با `--source` هم منبع را می‌گیرد، ولی اسکریپت‌های رسم خط و منطقه منبع را فقط از همین فایل می‌خوانند.

---

## ۶. کالیبراسیون کف زمین

**چرا لازم است:** فاصله‌ی متری (راننده تا لیفتراک)، سرعت افراد و بخشی از منطق هویت به تبدیل «پیکسل به متر» وابسته‌اند. بدون کالیبراسیون، یک مقدار پیش‌فرض تقریبی استفاده می‌شود که برای دوربین شما درست نیست.

**مراحل:**
1. روی زمین یک **مستطیل** انتخاب کنید که چهار گوشه‌اش در تصویر دیده می‌شود (مثلاً خطوط کف یا کاشی‌ها).
2. **طول و عرض واقعی** آن را با متر اندازه بگیرید.
3. اجرا کنید (عددها را با اندازه‌ی واقعی جایگزین کنید):
   ```powershell
   python scripts/calibrate_ground_plane.py --source data/input/clip1.mp4 --width-m 6 --length-m 8
   ```
   - `--width-m`: فاصله‌ی گوشه‌ی ۱ تا ۲ (بالا-چپ تا بالا-راست)
   - `--length-m`: فاصله‌ی گوشه‌ی ۲ تا ۳ (بالا-راست تا پایین-راست)
4. چهار گوشه را **ساعت‌گرد** کلیک کنید: بالا-چپ ← بالا-راست ← پایین-راست ← پایین-چپ.
5. کلیدها: `s` ذخیره، `r` شروع دوباره، `q` خروج.

نتیجه در `configs\calibration.yaml` ذخیره می‌شود و `main.py` و `run_vehicle_monitoring.py` خودکار از آن استفاده می‌کنند. رزولوشن تصویر هم ذخیره می‌شود، پس اگر بعداً رزولوشن استریم عوض شود، نقاط خودکار مقیاس می‌گیرند.

**بررسی درستی:** بعد از اجرای اصلی، کنار هر فرد مختصات `BEV:(x m, y m)` نمایش داده می‌شود. فردی را بین دو نقطه‌ی معلوم روی زمین حرکت دهید و تفاوت مختصات را با فاصله‌ی واقعی مقایسه کنید.

**هر بار که دوربین جابه‌جا شد یا زاویه‌اش تغییر کرد، کالیبراسیون را تکرار کنید.**

---

## ۷. منطقه‌ها و خطوط ورود/خروج (اختیاری)

برای هدف شناسایی هویت لازم نیستند؛ فقط برای شمارش ورود و خروج یا زمان حضور در یک محدوده.

**پیش از هر کار، از فایل تنظیمات نسخه‌ی پشتیبان بگیرید.** این اسکریپت‌ها `settings.yaml` را بازنویسی می‌کنند و توضیحات (کامنت‌های) داخل فایل حذف می‌شود:
```powershell
Copy-Item configs\settings.yaml configs\settings.backup.yaml
```

**خط عبور (tripwire)، برای شمارش ورود و خروج از یک در:**
```powershell
python scripts/draw_line.py --mode line --name main_gate
```
نقطه‌ی A و B را کلیک کنید. `i` جهت فلش «ورود» را برعکس می‌کند. با `s` ذخیره کنید و با `q` خارج شوید.

**دروازه‌ی ورود (چندضلعی + جهت):**
```powershell
python scripts/draw_line.py --mode zone --name company_entrance_gate
```
۳ گوشه یا بیشتر را کلیک کنید. `d` به حالت «جهت» می‌رود؛ دو کلیک از بیرون به داخل. `c` پاک می‌کند، `s` ذخیره، `q` خروج. کلیک راست آخرین نقطه را برمی‌گرداند.

**منطقه‌های حضور (مثلاً محوطه‌ی بارگیری):**
```powershell
python scripts/draw_zone.py
```
`SPACE` فریم را ثابت می‌کند. `TAB` یا `m` بین حالت منطقه و حالت کالیبراسیون جابه‌جا می‌شود. `d` آخرین منطقه را حذف می‌کند، `s` ذخیره، `q` خروج.

> کالیبراسیونی که `draw_zone.py` در `settings.yaml` می‌نویسد فقط **جایگزین پشتیبان** است. اگر `configs/calibration.yaml` (بخش ۶) وجود داشته باشد، همان استفاده می‌شود.

---

## ۸. اجرای اصلی

```powershell
python main.py --source data/input/clip1.mp4
```

**آرگومان‌ها:**

| آرگومان | پیش‌فرض | توضیح |
|---|---|---|
| `--source` | از `settings.yaml` | فایل ویدیو یا آدرس RTSP |
| `--config` | `configs/settings.yaml` | فایل تنظیمات اصلی (برای آزمایش‌های بخش ۱۱) |
| `--vehicle-config` | `configs/vehicle_monitoring.yaml` | تنظیمات لیفتراک. اگر فایل نباشد، ماژول لیفتراک اجرا نمی‌شود. |
| `--calibration` | `configs/calibration.yaml` | فایل کالیبراسیون |

**در ابتدای لاگ این خطوط را بررسی کنید:**

| خط | معنی |
|---|---|
| `Using cuda:0: NVIDIA ...` | GPU فعال است |
| `Ground calibration loaded from configs/calibration.yaml` | کالیبراسیون اعمال شده |
| `Re-ID backbone osnet_x1_0: weights ... (100% of layers matched)` | مدل Re-ID درست بارگذاری شده |

اگر به‌جای این‌ها هشدار دیدید، بخش ۱۵ را ببینید.

**توقف:** کلید `q` در پنجره‌ی ویدیو، یا `Ctrl+C` در ترمینال.

**برای اجرای سریع‌تر بدون پنجره‌ی نمایش**، در `settings.yaml` مقدار `display_gui: false` بگذارید. ویدیوی خروجی همچنان ذخیره می‌شود.

---

## ۹. خواندن خروجی‌ها

### ۹-۱. فایل‌ها

| فایل | محتوا |
|---|---|
| `output_industrial_reid.mp4` | ویدیوی برچسب‌خورده (مسیر در `system.output_video_path`) |
| `logs/identity_decisions.jsonl` | هر تصمیم هویتی، یک خط JSON |
| `data/vehicle_safety_events.csv` و `.db` | رویدادهای راننده و لیفتراک |
| `data/snapshots/vehicle_safety/` | تصویر لحظه‌ی هر رویداد راننده |
| `attendance_events.log` | رویدادهای خطوط و دروازه‌ها (اگر تعریف شده باشند) |

### ۹-۲. راهنمای تصویر: افراد

| نمایش | معنی |
|---|---|
| کادر **خاکستری** `T:12 [INIT]` | فرد تازه دیده شده و هنوز ID قطعی نگرفته (چند فریم اول) |
| کادر **سبز** `ID:7` | فرد با ID تأییدشده |
| کادر **نارنجی** `ID:7 [GALLERY LOCKED]` | فرد با دیگری هم‌پوشانی دارد؛ حافظه‌ی ظاهری‌اش موقتاً به‌روز نمی‌شود |
| خطوط فیروزه‌ای روی بدن | اسکلت (keypoint‌ها) |
| دایره‌ی فیروزه‌ای با حاشیه‌ی مشکی پایین بدن | نقطه‌ی تماس با زمین |
| `BEV:(x m, y m)` | موقعیت فرد روی زمین به متر |
| نوار بالای تصویر | FPS پردازش، تعداد افراد در فریم، تعداد کل ID‌ها، ورود/خروج |

### ۹-۳. راهنمای تصویر: لیفتراک

| رنگ کادر | وضعیت | معنی |
|---|---|---|
| خاکستری (نازک) | `UNASSIGNED` / `PARKED (No Driver)` | لیفتراک بدون راننده |
| سبز | `DRIVER_ONBOARD` | راننده داخل کابین است |
| آبی روشن | `DRIVER_EXITED_SAFE` | راننده پیاده شده، فاصله کمتر از حد هشدار |
| نارنجی | `DRIVER_EXITED_WARNING` | فاصله بین حد هشدار (۲ متر) و حد مجاز (۳ متر) |
| قرمز | `DISTANCE_VIOLATION_ALARM` | **راننده بیش از حد مجاز دور شده** |
| بنفش | `DISTANCE_UNCERTAIN` | فاصله قابل‌اندازه‌گیری نیست (مثلاً پای راننده دیده نمی‌شود) |
| قرمز تیره | `DRIVER_NOT_VISIBLE_ALARM` | راننده بیرون از کابین بوده و ناپدید شده (به‌طور پیش‌فرض خاموش) |

- خط بین راننده و لیفتراک فاصله را نشان می‌دهد: `2.40m / Max 3.0m`.
- پنل بالا-چپ فهرست لیفتراک‌ها، راننده‌ها و فاصله‌ها را دارد.
- در زمان هشدار، دور تصویر قاب قرمز و بنر `ALARM` نمایش داده می‌شود.

### ۹-۴. رویدادهای راننده (`vehicle_safety_events.csv`)

| `event_type` | معنی |
|---|---|
| `DRIVER_BOARDED` | راننده سوار لیفتراک شد |
| `DRIVER_EXITED` | راننده پیاده شد |
| `SAFE_DISTANCE_VIOLATION_ALARM` | راننده از حد مجاز دور شد |
| `DRIVER_NOT_VISIBLE_ALARM` | راننده‌ی پیاده‌شده ناپدید شد (فقط اگر فعال باشد) |

### ۹-۵. داشبورد وب (اختیاری)
اگر `dashboard.enable_web_dashboard: true` باشد:
- تصویر زنده: http://127.0.0.1:8050/video_feed
- وضعیت لیفتراک‌ها (JSON): http://127.0.0.1:8050/api/telemetry

پیش‌فرض فقط روی همین کامپیوتر در دسترس است. برای دسترسی از شبکه `host: 0.0.0.0` بگذارید؛ این داشبورد احراز هویت ندارد.

---

## ۱۰. گزارش تصمیم‌های هویتی

```powershell
python scripts/analyze_decisions.py --out logs/identity_report.txt
```

| آرگومان | توضیح |
|---|---|
| `--log` | فایل لاگ (پیش‌فرض `logs/identity_decisions.jsonl`) |
| `--out` | ذخیره‌ی گزارش متنی |
| `--json` | ذخیره‌ی خلاصه به‌صورت JSON |
| `--relative` | زمان از اولین رویداد (برای دوربین زنده) |
| `--include-rotated` | خواندن فایل قدیمی‌تر `.1` هم (لاگ بعد از ۵۰ مگابایت چرخش می‌کند) |

**بخش‌های گزارش:**

| بخش | محتوا |
|---|---|
| ۱ | تعداد هر نوع تصمیم و نرخ در دقیقه |
| ۲ | شباهت تطبیق‌های پذیرفته‌شده، به تفکیک مرحله |
| ۳ | ID‌های جدیدی که فرد مشابهی در سیستم داشتند (احتمال «یک نفر، دو ID») |
| ۴ | بازگشت افراد خارج‌شده |
| ۵ | ادغام ID‌های موقت |
| ۶ | موارد «دو نفر بیش از حد شبیه» |
| ۷ | خروج‌ها به تفکیک لبه‌ی تصویر |
| **۸** | **لحظه‌هایی که باید در ویدیو بررسی شوند، با زمان** (برای فایل ویدیو، همان زمان پخش است) |

**معنی رویدادهای لاگ:**

| رویداد | معنی |
|---|---|
| `NEW_ID` | ID جدید ساخته شد (`best_existing_sim` = شباهت به نزدیک‌ترین فرد قبلی) |
| `MATCHED` | ردیاب جدید به ID موجود وصل شد. `stage`: `STITCH` (ادامه‌ی مسیر بعد از وقفه‌ی کوتاه)، `RECOVER` (پس از پنهان‌شدن)، `REENTRY_STRICT` (بازگشت فرد خارج‌شده)، `BORDER_REENTRY` |
| `MATCH_REJECTED_AMBIGUOUS` | تطبیق رد شد، چون دو نفر بیش از حد شبیه بودند |
| `EXITED` | فرد از کادر خارج شد (`edge` = لبه‌ی خروج) یا خیلی طولانی پنهان ماند (`TIMEOUT`) |
| `MERGED` | یک ID موقت با ID قدیمی‌تر یکی شد |
| `BINDING_BROKEN` | ظاهر فرد دیگر با ID‌اش جور نبود؛ احتمالاً ردیاب روی فرد دیگری پریده بود |
| `SWAP_FIXED` | جابه‌جایی ID دو نفر تشخیص داده و اصلاح شد |
| `TRACK_RESURRECTED` | شماره‌ی ردیاب بعد از وقفه دوباره ظاهر شد و مثل فرد جدید بررسی شد |
| `PURGED` | هویت قدیمی از حافظه پاک شد (حد زمانی یا تعداد) |

---

## ۱۱. مقایسه‌ی مدل‌ها و ردیاب‌ها

**اصل کار:** یک ویدیوی ثابت، و **هر بار فقط یک تغییر**.

### ۱۱-۱. انتخاب ویدیوی آزمون
یک یا دو کلیپ ۳ تا ۵ دقیقه‌ای که این صحنه‌ها را داشته باشد:
- خروج و ورود دوباره‌ی چند نفر، ترجیحاً از یک سمت
- عبور افراد از کنار هم و پنهان‌شدن پشت لیفتراک
- سوار و پیاده‌شدن راننده

### ۱۱-۲. یادداشت «واقعیت» ویدیو (یک بار)
ویدیو را یک بار ببینید و بنویسید:
1. تعداد واقعی افراد
2. هر خروج و ورود دوباره، با زمان و توصیف فرد
3. هر سوار یا پیاده‌شدن راننده، با زمان

### ۱۱-۳. توابع کمکی (در هر پنجره‌ی جدید PowerShell یک بار اجرا کنید)
```powershell
function New-Exp {
  param([string]$Name, [string]$Tracker = "configs/custom_bytetrack.yaml",
        [string]$Pose = "yolo11m-pose.pt", [string]$ReidModel = "osnet_x1_0",
        [string]$ReidWeights = "weights/osnet_x1_0_msmt17.pt")
  (Get-Content configs\settings.yaml -Raw) `
    -replace 'tracker_config: configs/custom_bytetrack.yaml', "tracker_config: $Tracker" `
    -replace 'model_weights: yolo11m-pose.pt', "model_weights: $Pose" `
    -replace 'model_name: osnet_x1_0', "model_name: $ReidModel" `
    -replace 'weights_path: weights/osnet_x1_0_msmt17.pt', "weights_path: $ReidWeights" `
    -replace 'output_video_path: output_industrial_reid.mp4', "output_video_path: runs/$Name.mp4" `
    -replace 'decision_log_path: logs/identity_decisions.jsonl', "decision_log_path: runs/${Name}_decisions.jsonl" `
    -replace 'display_gui: true', 'display_gui: false' |
    Set-Content -Encoding utf8 "configs\exp_$Name.yaml"
}

function Run-Exp([string]$Name) {
  python main.py --config "configs/exp_$Name.yaml" --source data/input/clip1.mp4
  python scripts/analyze_decisions.py --log "runs/${Name}_decisions.jsonl" --out "runs/${Name}_report.txt"
  if (Test-Path data\vehicle_safety_events.csv) {
    Move-Item -Force data\vehicle_safety_events.csv "runs\${Name}_vehicle_events.csv"
  }
}

New-Item -ItemType Directory -Force runs
```
- `New-Exp` از `settings.yaml` یک نسخه‌ی آزمایشی با یک تغییر می‌سازد. خروجی‌ها به `runs/` می‌روند و پنجره‌ی نمایش خاموش است.
- `Run-Exp` اجرا می‌کند، گزارش می‌سازد و رویدادهای لیفتراک را جدا می‌کند.
- اگر ویدیو نام دیگری دارد، `clip1.mp4` را در `Run-Exp` عوض کنید.
- این توابع فرض می‌کنند مقادیر پیش‌فرض `settings.yaml` (ردیاب، مدل‌ها و مسیرها) دست نخورده‌اند.

### ۱۱-۴. مرحله‌ی ۱: ردیاب
```powershell
New-Exp -Name base
New-Exp -Name ocsort     -Tracker configs/trackers/factory_ocsort.yaml
New-Exp -Name deepocsort -Tracker configs/trackers/factory_deepocsort.yaml
New-Exp -Name tracktrack -Tracker configs/trackers/factory_tracktrack.yaml

Run-Exp base
Run-Exp ocsort
Run-Exp deepocsort
Run-Exp tracktrack
```

| ردیاب | ویژگی |
|---|---|
| ByteTrack (`base`) | پیش‌فرض؛ سریع و ساده |
| OC-SORT | بازسازی بهتر مسیر پس از پنهان‌شدن پشت مانع |
| Deep OC-SORT | OC-SORT + مدیریت بهتر track‌ها |
| TrackTrack | (CVPR 2025) مقاوم در هم‌پوشانی افراد |

### ۱۱-۵. مرحله‌ی ۲: مدل اسکلت (با ردیاب برنده‌ی مرحله‌ی ۱)
```powershell
New-Exp -Name pose_l -Tracker configs/trackers/factory_ocsort.yaml -Pose yolo11l-pose.pt
New-Exp -Name pose_x -Tracker configs/trackers/factory_ocsort.yaml -Pose yolo11x-pose.pt
Run-Exp pose_l
Run-Exp pose_x
```
(`factory_ocsort.yaml` را با ردیاب برنده‌ی واقعی جایگزین کنید.) مدل‌های بزرگ‌تر دقیق‌ترند ولی کندتر.

### ۱۱-۶. مرحله‌ی ۳: مدل Re-ID (با برنده‌های قبلی؛ وزن AIN از بخش ۴-۵ لازم است)
```powershell
New-Exp -Name ain -Tracker configs/trackers/factory_ocsort.yaml -Pose yolo11x-pose.pt `
        -ReidModel osnet_ain_x1_0 -ReidWeights weights/osnet_ain_x1_0_msmt17.pt
Run-Exp ain
```
OSNet-AIN برای کار در محیطی متفاوت از داده‌ی آموزشش طراحی شده است (نور، دوربین و لباس متفاوت).

### ۱۱-۷. ثبت و مقایسه‌ی نتایج
خروجی هر اجرا در `runs\`:
- `<name>.mp4`: ویدیو
- `<name>_report.txt`: گزارش هویت
- `<name>_vehicle_events.csv`: رویدادهای راننده

| اجرا | کل IDها (واقعی: __) | بازگشت درست | ID اشتباه | ID جدید برای فرد برگشتی | راننده‌ی درست | FPS |
|---|---|---|---|---|---|---|
| base | | | | | | |
| ocsort | | | | | | |
| deepocsort | | | | | | |
| tracktrack | | | | | | |
| pose_l / pose_x | | | | | | |
| ain | | | | | | |

| ستون | از کجا |
|---|---|
| کل IDها | بخش ۱ گزارش (`NEW_ID`) |
| بازگشت درست | بخش ۴ گزارش، در برابر یادداشت ۱۱-۲ |
| ID اشتباه و ID جدید برای فرد برگشتی | بخش ۸ گزارش + دیدن همان لحظه در ویدیو |
| راننده‌ی درست | `*_vehicle_events.csv`، در برابر یادداشت ۱۱-۲ |
| FPS | نوار بالای ویدیو |

**قاعده‌ی تصمیم:** «ID اشتباه» (فرد ID کس دیگری را بگیرد) از «ID جدید» بدتر است. گزینه‌ای برنده است که ID اشتباه کمتری داشته باشد و FPS‌اش قابل‌قبول بماند. برنده را در `configs\settings.yaml` اصلی اعمال کنید.

---

## ۱۲. مرجع پیکربندی

### ۱۲-۱. `configs/settings.yaml`

**`system`**

| کلید | توضیح |
|---|---|
| `device` | `cuda:0` یا `cpu`. اگر CUDA در دسترس نباشد، خودکار با پیام خطا به CPU برمی‌گردد. |
| `half_precision` | محاسبه‌ی FP16 روی GPU (سریع‌تر) |
| `display_gui` | نمایش پنجره‌ی زنده |
| `output_video_path` | مسیر ویدیوی خروجی؛ خالی = ذخیره نشود |

**`stream`**

| کلید | توضیح |
|---|---|
| `source` | فایل ویدیو یا RTSP |
| `drop_frames_on_lag` | برای دوربین زنده: همیشه جدیدترین فریم پردازش شود |
| `reconnect_delay_sec` | فاصله‌ی تلاش برای اتصال دوباره به دوربین |

**`detection`**

| کلید | توضیح |
|---|---|
| `model_weights` | `yolo11m-pose.pt` (پیش‌فرض)، `yolo11l-pose.pt` یا `yolo11x-pose.pt` |
| `tracker_config` | ردیاب: `configs/custom_bytetrack.yaml` یا یکی از `configs/trackers/factory_*.yaml` |
| `conf_thres` / `iou_thres` | آستانه‌ی تشخیص (پایین برای کارگران دور) |
| `imgsz` | اندازه‌ی ورودی مدل (حداقل ۱۲۸۰) |

**`reid`: مدل ظاهری**

| کلید | توضیح |
|---|---|
| `model_name` | `osnet_x1_0` (پیش‌فرض، دانلود خودکار) یا `osnet_ain_x1_0` (وزن دستی) |
| `weights_path` | مسیر وزن. وزنی که با مدل جور نباشد در شروع رد می‌شود. |
| `head_branch_weight` / `color_branch_weight` | وزن بالاتنه و رنگ در ویژگی نهایی |
| `min_gallery_height_px` / `min_visible_kpts` | crop‌های کوچک‌تر یا کم‌مفصل‌تر در حافظه‌ی ظاهری ذخیره نمی‌شوند |
| `anchor_min_height_px` | حداقل اندازه برای نمونه‌های دائمی هر فرد |

**`reid`: تطبیق هویت**

| کلید | پیش‌فرض | توضیح |
|---|---|---|
| `cos_sim_thresh` | 0.60 | حداقل شباهت برای بازیابی عمومی |
| `spatial_recovery_sim_thresh` | 0.48 | حداقل شباهت وقتی فرد نزدیک محل گم‌شدنش دوباره ظاهر شود |
| `stitch_sim_thresh` | 0.55 | ادامه‌ی مسیر بعد از وقفه‌ی کوتاه |
| `stitch_margin` | 0.04 | فاصله‌ی لازم بین بهترین و دومین گزینه |
| `reentry_sim_thresh` | 0.82 | **حداقل شباهت برای پس دادن ID به فرد برگشتی** |
| `reentry_margin` | 0.08 | فاصله‌ی لازم با دومین گزینه (شامل افراد حاضر در تصویر) |
| `reentry_min_samples` | 3 | حداقل نمونه‌ی تمیز پیش از تصمیم |
| `reentry_min_gap_sec` | 2.0 | بازگشت زودتر از این، شباهت بیشتری لازم دارد |
| `pool_max_samples` / `pool_window_sec` | 8 / 4.0 | ظاهر هر فرد = میانگین وزن‌دار چند crop اخیر |
| `reid_interval_sec` | 0.35 | فاصله‌ی بررسی دوباره‌ی ظاهر افراد ردیابی‌شده |
| `mismatch_min_span_sec` | 1.0 | ناهمخوانی ظاهر باید این مدت ادامه یابد تا اتصال شکسته شود |

**`reid`: خروج از کادر**

| کلید | توضیح |
|---|---|
| `exit_edges` | لبه‌هایی که راه خروج واقعی‌اند؛ مثلاً اگر بالای تصویر دیوار است، `top` را حذف کنید |
| `exit_confirm_sec` | مدت غیبت پیش از «خارج‌شده» حساب‌شدن |
| `border_margin_ratio` | پهنای نوار «نزدیک لبه» (نسبت به ابعاد تصویر) |
| `occlusion_max_sec` | طولانی‌ترین پنهان‌شدن داخل تصویر قبل از «خارج‌شده» حساب‌شدن |
| `track_gap_reset_sec` | ردیابی که بیش از این غایب بوده، مثل فرد جدید بررسی می‌شود |

**`reid`: حافظه و لاگ**

| کلید | توضیح |
|---|---|
| `identity_retention_sec` / `max_exited_identities` | چه مدت و چند نفر از افراد خارج‌شده به خاطر سپرده شوند |
| `track_state_ttl_sec` | پاک‌سازی ردیاب‌های مرده |
| `decision_log_path` | مسیر لاگ تصمیم‌ها؛ خالی = خاموش |

**`physics_bev`**

| کلید | توضیح |
|---|---|
| `max_worker_speed_mps` | حداکثر سرعت معقول راه رفتن (برای رد تطبیق‌های غیرممکن) |
| `max_internal_occlusion_m` | حداکثر جابه‌جایی فرد در زمان پنهان‌بودن |
| `src_image_pts` / `dst_meter_pts` / `calibration_ref_wh` | کالیبراسیون پشتیبان؛ اگر `calibration.yaml` باشد استفاده نمی‌شود |

**`zones`** و **`directional_tracking`**: با اسکریپت‌های بخش ۷ پر می‌شوند.

### ۱۲-۲. `configs/vehicle_monitoring.yaml`

**تشخیص وسیله**

| کلید | توضیح |
|---|---|
| `vehicle_detector_weights` | مدل تشخیص وسیله. پیش‌فرض `yolo11m.pt` (مدل عمومی COCO که کلاس لیفتراک ندارد؛ کلاس ۷ = کامیون). **با مدل اختصاصی لیفتراک جایگزین کنید.** |
| `vehicle_class_ids` | شماره‌ی کلاس وسیله در همان مدل |
| `vehicle_conf_threshold` | آستانه‌ی تشخیص وسیله |
| `vehicle_imgsz` | اندازه‌ی ورودی (۱۲۸۰ برای لیفتراک‌های دور) |
| `vehicle_half` | FP16 |
| `vehicle_detect_every_n_frames` | اجرای تشخیص وسیله هر N فریم (۲ یا ۳ برای سرعت بیشتر) |
| `vehicle_tracker_config` | ردیاب جداگانه برای وسایل |

**`pairing`: اتصال راننده**

| کلید | توضیح |
|---|---|
| `entry_iou_threshold` | حداقل هم‌پوشانی با کابین |
| `entry_confirmation_frames` / `moving_entry_confirmation_frames` | تعداد فریم‌های لازم برای تأیید سوار شدن (لیفتراک ساکن / متحرک) |
| `exit_confirmation_frames` | تعداد فریم‌های بیرون از کابین برای تأیید پیاده‌شدن |
| `comotion_min_cosine` و `comotion_*_speed_ratio` | در لیفتراک متحرک، راننده باید هم‌جهت و هم‌سرعت با آن حرکت کند |
| `driver_handover_iou` / `driver_handover_max_gap_sec` | اگر ID راننده عوض شد ولی همان بدن همان‌جا بود، اتصال حفظ شود |
| `reid_similarity_threshold` | بازیابی راننده با ظاهر پس از تغییر ID |
| `dismount_spawn_radius_m` | تشخیص راننده‌ای که از کابین پنهان پیاده می‌شود |
| `allow_provisional_ids_for_spawn` / `spawn_border_margin_px` | جلوگیری از اینکه هر عابر تازه‌ای کنار لیفتراک راننده حساب شود |
| `driver_lost_alarm_sec` | هشدار «راننده ناپدید شد»؛ **۰ = خاموش**. مقدار را با مسئول ایمنی تعیین کنید. |

**`distance`**

| کلید | توضیح |
|---|---|
| `safe_distance_threshold_m` | حد مجاز فاصله‌ی راننده از لیفتراک (هشدار قرمز) |
| `warning_distance_threshold_m` | حد هشدار نارنجی |
| `ema_smoothing_alpha` / `max_step_jump_m` | هموارسازی و حذف پرش‌های ناگهانی فاصله |

**سایر بخش‌ها**

| بخش | توضیح |
|---|---|
| `alarms` | بوق، فاصله‌ی بین هشدارها |
| `logging` | مسیر فایل‌های رویداد و اسنپ‌شات |
| `dashboard` | داشبورد وب (بخش ۹-۵) |

### ۱۲-۳. ردیاب‌ها (`configs/custom_bytetrack.yaml` و `configs/trackers/`)
آستانه‌های پایین (`track_high_thresh: 0.15`، `new_track_thresh: 0.18`) برای کارگرانِ دور با اطمینان تشخیص پایین تنظیم شده‌اند. `track_buffer` بر حسب **فریم پردازش‌شده** است، نه ثانیه.

---

## ۱۳. اجرای مستقل ماژول لیفتراک

بدون اسکلت و Re-ID؛ افراد و وسایل با یک مدل YOLO تشخیص داده می‌شوند. مناسب برای تست سریع بخش لیفتراک.
```powershell
python run_vehicle_monitoring.py --source data/input/clip1.mp4
```
در این حالت، اتصال راننده فقط بر اساس هم‌پوشانی و حرکت است و بازیابی راننده با ظاهر غیرفعال است. برای ارزیابی واقعی از `main.py` استفاده کنید.

---

## ۱۴. تست‌ها

```powershell
pytest -q
```

| فایل | چه چیزی را می‌آزماید |
|---|---|
| `test_identity_reentry.py` | خروج و ورود دوباره، جلوگیری از به ارث بردن ID، ادغام، پاک‌سازی حافظه |
| `test_identity_pooling.py` | میانگین وزن‌دار ظاهر هر فرد |
| `test_reid_batch.py` | استخراج ویژگی گروهی و فیلتر کیفیت |
| `test_reid_weights.py` | انتخاب مدل Re-ID و رد وزن‌های ناجور |
| `test_tracker_presets.py` | معتبر بودن و کارکرد ردیاب‌ها |
| `test_driver_assignment.py` | اتصال راننده، هم‌حرکتی، تغییر ID، هشدارها |
| `test_vehicle_monitoring.py` | فاصله‌ی متری و چرخه‌ی سوار / پیاده / هشدار |
| `test_calibration_and_state.py` | کالیبراسیون، تغییر نام پس از ادغام، رسم بهینه |
| `test_analyze_decisions.py` | گزارش تصمیم‌ها |

تست‌ها به GPU و ویدیو نیاز ندارند؛ از داده‌ی ساختگی استفاده می‌کنند. **پس از هر تغییر در کد، تست‌ها را اجرا کنید.**

---

## ۱۵. رفع اشکال

| پیام یا مشکل | علت و راه‌حل |
|---|---|
| `CUDA requested ... Falling back to CPU` | PyTorch نسخه‌ی CPU نصب است؛ بخش ۴-۲. |
| `Torch not compiled with CUDA enabled` | همان بالا. |
| `No module named 'tensorboard'` | `pip install tensorboard` (torchreid به آن نیاز دارد). |
| `No module named 'torchreid'` / `'scipy'` | `pip install -r requirements.txt` |
| `Failed to acquire MSMT17 weights` | دانلود خودکار شکست خورد؛ دانلود دستی (بخش ۴-۵). |
| `Re-ID model 'osnet_ain_x1_0' needs its own weights` | وزن AIN در `weights\` نیست (بخش ۴-۵). |
| `... do not fit model ...` | فایل وزن متعلق به مدل دیگری است. با این وزن، مدل عملاً تصادفی کار می‌کرد. |
| `Failed to open video source` | مسیر ویدیو یا آدرس RTSP را بررسی کنید. |
| `No ...calibration.yaml: using the physics_bev points` | بخش ۶ انجام نشده؛ فاصله‌های متری معتبر نیستند. |
| FPS خیلی پایین | ۱) CUDA فعال باشد. ۲) `vehicle_detect_every_n_frames: 2`. ۳) مدل اسکلت کوچک‌تر. ۴) `display_gui: false`. |
| لیفتراک تشخیص داده نمی‌شود | مدل پیش‌فرض کلاس لیفتراک ندارد؛ مدل اختصاصی در `vehicle_detector_weights`. |
| افراد دور ID نمی‌گیرند | crop‌ها خیلی کوچک‌اند؛ مدل اسکلت بزرگ‌تر یا `imgsz` بیشتر. |
| ID افراد زیاد عوض می‌شود | گزارش بخش ۱۰ را بسازید؛ بخش‌های ۳ و ۸ علت را نشان می‌دهند. ردیاب‌های بخش ۱۱ را امتحان کنید. |

---

## ۱۶. محدودیت‌ها، مجوزها و نکات ایمنی

**محدودیت‌های فنی**
- **یونیفرم یکسان:** وقتی همه لباس مشابه دارند، تشخیص هویت فقط با ظاهر بدن دشوار است. برای غیبت‌های کوتاه نتیجه بهتر است؛ برای غیبت‌های طولانی (نیم ساعت تا چند ساعت) احتمال ID جدید بالا می‌رود. سیستم طوری تنظیم شده که در تردید ID جدید بدهد، نه ID اشتباه.
- **یک دوربین:** ID‌ها بین دوربین‌ها مشترک نیستند.
- **پس از ری‌استارت** برنامه، ID‌ها از ۱ شروع می‌شوند (هویت‌ها فقط در حافظه‌اند).
- **آستانه‌ها** با منطق و داده‌ی ساختگی تنظیم شده‌اند و باید با ویدیوی واقعی همین کارخانه دوباره تنظیم شوند (بخش ۱۱).
- **مدل لیفتراک پیش‌فرض** کلاس لیفتراک ندارد.

**مجوزها (پیش از استفاده‌ی تجاری با واحد حقوقی بررسی شود)**
- **Ultralytics YOLO** مجوز AGPL-3.0 دارد؛ استفاده‌ی تجاری بسته معمولاً مجوز Enterprise یا رعایت AGPL را لازم دارد.
- وزن‌های **OSNet / OSNet-AIN** روی مجموعه‌داده‌ی MSMT17 آموزش دیده‌اند که برای پژوهش غیرتجاری منتشر شده است.

**ایمنی و حریم خصوصی**
- این سامانه **تجهیز ایمنی گواهی‌شده نیست** و جایگزین حفاظ فیزیکی، رویه‌ها و آموزش نمی‌شود؛ یک سامانه‌ی **کمکی** هشدار است.
- ضبط و ردیابی تصویر کارکنان باید مطابق قوانین محلی و سیاست سازمان باشد: اطلاع‌رسانی، محدودیت دسترسی و مدت نگهداری تصاویر (`data/snapshots/`، ویدیوهای خروجی).
