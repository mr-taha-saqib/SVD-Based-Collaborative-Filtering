# Dataset licensing assessment (Project 2)

Assessment date: 2026-11-17. This is an engineering summary of published terms, **not legal
advice**. The client's own legal review decides commercial use.

## 1. MovieLens (the dataset of the delivered model)

| Item | Finding |
|---|---|
| Source | GroupLens Research, University of Minnesota (`files.grouplens.org`) |
| Licence | Research use only. The dataset README says it "may be used for any research purposes" under conditions that include no commercial or revenue-bearing use without permission from a GroupLens faculty member, no redistribution without separate permission, and acknowledgement in publications |
| Commercial route | The University of Minnesota Technology Commercialization office lists the **MovieLens Database** (Technology ID **z05173**) as available "via a standard negotiated license agreement". No price or terms are published. Contact: license.umn.edu (product page "MovieLens Database"), Technology Commercialization, 280 McNamara Alumni Center, 200 Oak St. SE, Minneapolis, MN 55455, +1 612 624 0550; general GroupLens contact grouplens-info@umn.edu |
| Status for this delivery | **No permission has been requested or obtained.** The delivered MovieLens model is for research and reference use only |

## 2. Alternatives checked

| Dataset | User-level ratings? | Terms found | Usable commercially? |
|---|---|---|---|
| **MovieTweetings** (sidooms/MovieTweetings, last update 2021-11-24) | yes, 921,398 ratings (1–10) | Repository `LICENSE`: **MIT**, © 2013 Simon Dooms. README asks users to cite the paper | **Candidate, subject to legal review** (see §3) |
| Netflix Prize | yes | Research use only; no redistribution; no commercial use without Netflix's permission. Not distributed officially any more | no |
| Yahoo! Webscope (R-series movie ratings) | yes | Non-commercial use by academics; not shared with commercial entities | no |
| IMDb datasets | no (aggregate ratings only) | IMDb Non-Commercial Datasets licence | no, and not suitable for collaborative filtering |
| TMDB-derived Kaggle sets | no (aggregate vote averages) | TMDB API terms for non-commercial use | no |
| "The Movies Dataset" (Kaggle, labelled CC0) | yes | Its ratings files were taken from GroupLens, so the MovieLens licence still applies despite the CC0 label | no |
| Rotten Tomatoes Kaggle set (CC0) | no (aggregate scores for ~900 films) | CC0 | licence yes, but not suitable for collaborative filtering |
| Letterboxd Kaggle set (CC BY 4.0) | yes | Scraped from letterboxd.com; the uploader's licence cannot grant rights the site's terms do not | not without Letterboxd's permission |

## 3. MovieTweetings: terms and open questions

Verified from the official repository at commit `44c525d0c766944910686c60697203cda39305d6`:

* Licence file: MIT (SHA-256 `ae985d01…dcab41c`). Commercial use, modification and redistribution
  are permitted **by the author**, provided the copyright and permission notice is kept.
* Files used: `latest/ratings.dat` (SHA-256 `f8a46fe1…eb13`) and `latest/movies.dat`
  (`0403d193…001f`). `users.dat`, which maps dataset user ids to real Twitter account ids, is
  **never downloaded or read** by this project.

Questions the client's legal review must answer before commercial use:

1. **Third-party content.** The ratings were extracted from public tweets posted by the IMDb apps'
   rating widget. The MIT grant covers only the author's own rights. X's developer terms restrict
   redistribution of X content and, since June 2025, prohibit using X content to "fine-tune or
   train a foundation or frontier model"; whether those terms bind a downstream user of this
   dataset, and whether a small SVD recommender is affected, is a legal question.
2. **Personal data (GDPR).** Each rating row is linkable to a real Twitter account through the
   public `users.dat` file, so the ratings are pseudonymous personal data. A lawful basis, a
   data-protection impact assessment where required, and data-subject rights handling are needed.
3. **IMDb references.** Movie ids are IMDb ids and titles/genres come from IMDb pages. Confirm
   whether IMDb's conditions of use affect storing and displaying this metadata (not reviewed here).
4. **Attribution.** Keep the MIT notice (`data/movietweetings/LICENSE`, downloaded with the data)
   and cite Dooms et al., *MovieTweetings: a Movie Rating Dataset Collected From Twitter*,
   CrowdRec @ RecSys 2013.
5. **Freshness.** The dataset stopped updating in November 2021.

## 4. What was done

A separate pipeline (`src/run_movietweetings.py`, results in `docs/MOVIETWEETINGS_EVALUATION.md`)
applies the corrected methodology to MovieTweetings. It uses its own registry, output folder and
MLflow experiment; the MovieLens model, results and registry are unchanged.

## 5. Decision required from the client

* **Option A:** negotiate a commercial MovieLens licence with the University of Minnesota (§1).
* **Option B:** have counsel clear MovieTweetings (§3), then promote the MovieTweetings model.
* **Option C (recommended for production):** use the client's own first-party rating data, which
  is what a production recommender should be trained on anyway.

Until then both models are for research and reference use only.
