# Explain @ 2025-10-29T08:00:00+00:00 (UTC)

## 1) ????  ??  ??
- ???? `w_t`?**+0.613**????? +nan?  | ?????**False**
- ????`core=nan` | `all=0.5306855884393531` | `vol=1080.0`
- ??????????1??`core=0.00` / `all=0.50` / `vol=0.50`
  - ????**????**??????????median/MAD??????????????????

## 2) ?????????????Pearson?
? `warehouse/explain_signal_corr.csv`??????`news_tone_avg_OIL_CORE, news_tone_avg, news_art_cnt, w_t, ret_1h`

## 3) ????????GKG  tone + url + themes?
??? `warehouse/explain_latest_articles.csv`???? zip ???tone?url?themes ????

| zip | tone | url | themes |
|---|---:|---|---|
| 20251029080000.gkg.csv.zip |  | https://aninews.in/news/national/politics/congress-bihar-incharge-praises-mahagathbandhans-manifesto-says-alliance-focusing-on-the-issues-that-matter-to-the-people20251029121626/ | GENERAL_GOVERNMENT;USPEC_POLITICS_GENERAL1;EPU_POLICY;EPU_POLICY_CONGRESS;ELECTION;ALLIANCE;MEDIA_MSM;DEMOCRACY;TAX_POLITICAL_PARTY;TAX_POLITICAL_PARTY_NATIONAL_DEMOCRATIC_ALLIANCE;TAX_FNCACT;TAX_FNCA |
| 20251029080000.gkg.csv.zip |  | https://en.apa.az/corridors/azerbaijan-may-establish-direct-air-links-with-xian-china-482103 |  |
| 20251029080000.gkg.csv.zip |  | https://dailypost.ng/2025/10/29/carabao-cup-wolves-ready-for-tough-chelsea-test-arokodare/ |  |
| 20251029080000.gkg.csv.zip |  | https://www.tomsguide.com/wellness/smartwatches/i-walked-6-500-steps-with-the-garmin-instinct-3-vs-suunto-vertical-2-heres-the-winner | URBAN;WB_137_WATER;TAX_DISEASE;TAX_DISEASE_COUGH;TAX_ECON_PRICE;ENV_SOLAR;TAX_FNCACT;TAX_FNCACT_GUIDE;CRISISLEX_CRISISLEXREC;WB_2936_GOLD;WB_507_ENERGY_AND_EXTRACTIVES;WB_895_MINING_SYSTEMS;WB_1699_ME |
| 20251029080000.gkg.csv.zip |  | http://gps-prod-storage.cloud.caltech.edu.s3.amazonaws.com/casino/video/online-casino-vergleich-wajsz.html |  |
| 20251029080000.gkg.csv.zip |  | http://gps-prod-storage.cloud.caltech.edu.s3.amazonaws.com/casino/video/gr%c3%bcn-gold-casino-wuppertal-bsjih.html | WB_2936_GOLD;WB_507_ENERGY_AND_EXTRACTIVES;WB_895_MINING_SYSTEMS;WB_1699_METAL_ORE_MINING; |
| 20251029080000.gkg.csv.zip |  | https://en.interfax.com.ua/news/general/1116353.html |  |
| 20251029080000.gkg.csv.zip |  | http://www.nepalnational.com/news/278662269/netcore-cloud-achieves-cmmi-level-3-appraisal |  |

---
### ??????/????
- ?? `w_t` ?? `warehouse/signals_hourly_exp3.csv` ??????
- ????? `data/features_hourly.parquet` ??????
- ???????????
- ?? GKG ??? `C:\WTI\data\gdelt_raw\YYYY\MM\YYYYMMDDHHMMSS.gkg.csv.zip`????**????? 15 ?? zip** ???
