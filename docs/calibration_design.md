# Cross-Author Calibration Design — Caesar Stylometry Project

**Document type:** Scholarly design specification (no code)
**Status:** Proposed — awaiting decision before implementation
**Date:** 2026-06-16

---

## 0. Calibration Claim (Falsifiable, Up Front)

> **The method that produced the annual-composition signal in DBG (negative
> DBC-anchor Spearman r; positive Mantel r; exact-permutation p < 0.05) SHOULD,
> when applied unchanged, produce:**
>
> **Positive control — a STRONG directional signal:**
> DBC-style anchor r ≪ 0 (p < 0.05) and Mantel r ≫ 0 (p < 0.05) on a corpus
> with **known, uncontested serial composition** over a multi-year span.
>
> **Negative control — a NULL or WEAK result:**
> DBC-style anchor r ≈ 0 (p > 0.05, preferably p > 0.20) and Mantel r ≈ 0
> (p > 0.05) on a corpus with **known, uncontested concentrated/single-period
> composition**, or where the segmentation order is known to be **arbitrary
> with respect to composition date**.
>
> If the positive control yields a null, the method lacks sensitivity to
> known chronological signal and cannot support claims about Caesar.
> If the negative control yields a strong signal, the method is picking up
> narrative-structure confounds rather than genuine chronometric signal, and
> the Caesar result is artefactual.

---

## 1. Design Rationale

### 1.1 Why calibration is necessary

The Caesar DBG result rests on n = 7 books over a 6-year span (58–52 BC).
The result is statistically significant under exact permutation, directionally
unanimous across 136 robustness conditions, and survives jackknife resampling
(Script 18). But two concerns remain that only cross-author calibration can
address:

1. **Method sensitivity:** Is the method *capable* of detecting a known
   chronological signal in any author, or is the Caesar result a fluke of
   7 data points?

2. **Narrative-structure confound:** Both the DBC Anchor and Mantel tests
   treat book-order as a proxy for time. But DBG books also follow a
   chronological *narrative* order (Helvetii → Belgians → Veneti →
   Germans/Britain → Vercingetorix). Could the signal reflect narrative
   progression within a single unified work, rather than genuinely
   diachronic composition?

A positive control (known serial composition) tests sensitivity. A negative
control (known single-period composition or arbitrary segmentation) tests
specificity.

### 1.2 The mirroring principle

The calibration corpus should be run through the **same feature-extraction
and statistical pipeline** as Caesar (Scripts 07–08 for feature matrices;
Scripts 13–14 for the DBC Anchor and Mantel tests; Script 18 for jackknife).
The only change is the input texts and the year/ordering metadata. This
mirroring ensures that any difference in outcome between Caesar and the
calibration corpora reflects a difference in the underlying composition
process, not a difference in method.

---

## 2. Assessment Axes

Every candidate is scored on five axes (A–E). Each axis is rated ✓✓
(excellent), ✓ (adequate), △ (problematic), or ✗ (unusable).

| Axis | Label | Question |
|------|-------|----------|
| **A** | Date security | How confident is the scholarly consensus on the composition date(s)? Are the dates transmitted/attested, or deduced? Is there a live dispute? |
| **B** | Genre comparability | How similar is the text to Caesar's third-person *commentarius* in register, length, and rhetorical mode? Larger differences = larger confound. |
| **C** | Corpus size & units | How many datable units are available? More units = more statistical power (Caesar's n = 7 is the floor). Can the units be ordered unambiguously by composition date? |
| **D** | Text availability | Is the text available as clean, machine-readable TEI XML or equivalent from Perseus, the Latin Library, or another reputable source? |
| **E** | Unit-of-analysis fit | Can the corpus be segmented into units that are suitable for the DBC-anchor and Mantel tests? Are the units large enough for reliable feature estimation? |

---

## 3. Positive-Control Candidates (Known Serial Composition)

### 3.1 Candidate P1: Cicero, Epistulae ad Atticum (Letters to Atticus)

| Axis | Rating | Detail |
|------|--------|--------|
| **A — Date security** | ✓✓ | **Gold standard.** Shackleton Bailey's edition (1965–1970) assigns a date to every surviving letter, often to the *day*, based on internal references (named consuls, described events, travel logistics). The corpus spans 68–44 BC (24 years). The dating is as secure as anything in classical studies. No live dispute. |
| **B — Genre comparability** | △ | **Major confound.** These are private, familiar letters in the first person. They are the opposite of Caesar's polished, third-person, public-facing *commentarius*. Register is informal, syntax is elliptical, topics are personal/political gossip. Both are Latin prose by elite Romans of the same generation (Cicero was Caesar's contemporary), which mitigates the confound somewhat — the underlying grammatical system is the same. |
| **C — Corpus size & units** | ✓✓ | **426 surviving letters across 16 books.** If letters are grouped by year → ~20 yearly bins (vs. Caesar's n = 7). If individual letters are used → n ≈ 426 with individually assigned dates, offering enormous statistical power. Individual letters vary in length (some are a few lines), so a minimum-token threshold would be needed (~100 tokens). |
| **D — Text availability** | ✓ | Perseus has *some* Cicero letters but not the full *Ad Atticum* corpus. The Latin Library has the complete text. The Packard Humanities Institute (PHI) Latin corpus has all of Cicero. TEI XML is available from some sources but may require conversion. |
| **E — Unit-of-analysis fit** | ✓ | **Yearly bins (recommended).** Group letters by assigned year → 20+ units, each containing multiple letters. Each yearly bin has enough tokens for reliable feature estimation. The DBC-analogue anchor would be Cicero's latest work (e.g., *De Officiis*, 44 BC, or *Philippics*, 44–43 BC), or the latest year's letters could be held out as the anchor. |

**Scholarly basis for dates:** Shackleton Bailey, D.R. (1965–1970). *Cicero's Letters to Atticus*. Cambridge University Press. The chronology is reconstructed from: (a) consular dates mentioned in the letters; (b) references to datable political events (Caesar's crossing of the Rubicon, battles, legislation); (c) Cicero's own movements between his villas and Rome, which are independently attested. The dates are universally accepted by Ciceronian scholars.

**Genre confound mitigation:** Use function words and character n-grams as the primary feature sets. These are less sensitive to register than MFW, which will capture topic-words. Run the analysis on both the full feature panel and a function-word-only panel and report the difference.

---

### 3.2 Candidate P2: Cicero, Orationes (Speeches)

| Axis | Rating | Detail |
|------|--------|--------|
| **A — Date security** | ✓✓ | Many speeches are closely datable to the day of delivery (trial dates, senate sessions). Span: *Pro Quinctio* (81 BC) to *Philippics* (44–43 BC), covering ~38 years. The dates are based on external historical evidence (consuls, political context) and are uncontested. The main complication is that some speeches were revised for publication after delivery — the *published* text may postdate the *delivery* by months. |
| **B — Genre comparability** | △ | Forensic and deliberative oratory. Highly formal register, elaborate periodic syntax, first/second-person address. Very different from Caesar's restrained third-person narrative. However, the rhetorical polish of the speeches is closer to Caesar's polished prose than the informal *Ad Atticum* letters are. |
| **C — Corpus size & units** | ✓ | ~58 surviving speeches, of which ~35 are substantially complete. Span ~38 years. The unit of analysis would be the individual speech, ordered by delivery date. Speeches vary enormously in length — from the short *Pro Archia* (~3,000 words) to the massive *Verrines* (collectively ~65,000 words). |
| **D — Text availability** | ✓✓ | Perseus has a substantial collection of Cicero's speeches in TEI XML. The PHI corpus has the complete set. |
| **E — Unit-of-analysis fit** | △ | Speeches vary so much in length (3K–65K words) that proportion-based features on very short speeches may be unreliable. A minimum-token threshold (~2,000 tokens) would exclude several important early speeches, reducing temporal coverage. The publication-vs-delivery gap introduces uncertainty in the date assigned to the *text we have*. |

---

### 3.3 Candidate P3: Pliny the Younger, Epistulae (Letters)

| Axis | Rating | Detail |
|------|--------|--------|
| **A — Date security** | △ | **Date security is materially weaker than Cicero.** Books I–IX were *published* as a curated collection ~100–109 AD. Individual letters are broadly datable (references to known events, Pliny's career), but scholars actively debate whether the books are arranged **chronologically** (by date of the event described) or **thematically** (for *variatio* and literary effect). Sherwin-White (1966) argues for broad chronological ordering with thematic rearrangement; more recent scholarship (Gibson & Morello 2012, Whitton 2019) emphasizes the literary, non-chronological structuring. This uncertainty directly undermines the calibration claim — if we detect a trend, is it composition drift, or Pliny's deliberate arrangement? |
| **B — Genre comparability** | △ | Literary epistles, curated for publication. More polished than Cicero's private letters, but still first-person. The "literary letter" is a distinct genre from Caesar's *commentarius*. |
| **C — Corpus size & units** | ✓ | 247 letters in Books I–IX, 122 letters in Book X (Trajan correspondence). Books I–IX are the primary candidate (~100–109 AD span, ~9 years). Book X is a separate case — official correspondence, very different register. |
| **D — Text availability** | ✓ | Perseus has Pliny's letters. |
| **E — Unit-of-analysis fit** | △ | If book-level (I–IX) → n = 9, comparable to Caesar but with date-uncertainty confound. If letter-level → n = 247, but letters can be as short as a single sentence (~10 words). |

**Scholarly dispute summary:** The central debate is whether Pliny's book-order reflects his *composition* order or his *editorial* order. If Pliny wrote letters across several years but then arranged them for publication in a non-chronological sequence, our method could detect a trend (reflecting his evolving style across the underlying composition period) that doesn't match the book-number ordering. This makes Pliny a high-risk, low-confidence positive control — even a strong signal would be ambiguous in interpretation.

---

## 4. Negative-Control Candidates (Known Single-Period or Arbitrary-Order Composition)

### 4.1 Candidate N1: DBC Pseudo-Books (Caesar's De Bello Civili, Split Arbitrarily)

| Axis | Rating | Detail |
|------|--------|--------|
| **A — Date security** | ✓✓ | DBC was composed ~49–48 BC, a concentrated 1–2 year period. This is uncontested. The work covers the civil war from Caesar's crossing of the Rubicon (January 49) through the Alexandrian War. It is a single continuous narrative, not a serial publication. |
| **B — Genre comparability** | ✓✓ | **Identical author, identical genre, identical register.** This is Caesar himself, writing in the same *commentarius* mode. The only difference between DBG and DBC is that DBG spans 7 years of composition (under Hypothesis A) while DBC spans 1–2 years. This is the closest possible match — it eliminates genre, author, and register as confounds. |
| **C — Corpus size & units** | ✓ | DBC = ~37K tokens across 243 chapters. Can be split into 7 pseudo-books of equal token count (~5.3K tokens each). This matches Caesar's DBG structure (n = 7, comparable token counts). |
| **D — Text availability** | ✓✓ | Already in the repository as `data/raw/perseus/caes.bc_lat.xml`. Zero additional acquisition needed. |
| **E — Unit-of-analysis fit** | ✓✓ | **Perfect fit.** The pseudo-books are constructed to mirror DBG's structure exactly. The DBC-anchor for this test would be the *latest* pseudo-book (pseudo-book 7), held out, and the Mantel test would use pseudo-book order (1→6 or 1→7) as the predictor. The key question: does the pseudo-book ordering — which is just sequential chunks of a single continuous narrative — produce a false positive? |

**How to build pseudo-books (two variants):**

| Variant | Construction | Tests |
|---------|-------------|-------|
| **N1a — Narrative order** | Split DBC into 7 equal-token segments **in narrative order** (chapters 1→243, chunked sequentially). Label pseudo-books 1–7. | If r ≈ 0 → narrative progression within a unified work does NOT masquerade as chronometric drift. If r ≪ 0 → narrative content creates false chronometric signal — a serious validity concern. |
| **N1b — Random order** | Shuffle DBC chapters randomly, then split into 7 equal-token pseudo-books. Label arbitrarily 1–7. | Pure negative control. No narrative, chronological, or any other signal should exist. r should be indistinguishable from 0. If significant, the method is detecting structure where none exists. |

**Prediction:** Under Hypothesis A (annual DBG composition), DBG should show strong drift and DBC pseudo-books (both variants) should show r ≈ 0. Variant N1a is the more informative test — it isolates whether sequential narrative alone can produce a false chronometric signal, which is the most pressing confound concern.

**Note on N1a and the content confound:** DBC's narrative does have a temporal arc (Rubicon → Spain → Pharsalus → Alexandria), and the topics change (Italian politics → Spanish campaign → Greek campaign → Egyptian intervention). If function words and char n-grams are truly content-independent, N1a should show no signal. If N1a shows a signal, it means even our best content-independent features are picking up narrative structure, and the DBG result must be interpreted more cautiously.

---

### 4.2 Candidate N2: Suetonius, De Vita Caesarum (Lives of the Twelve Caesars)

| Axis | Rating | Detail |
|------|--------|--------|
| **A — Date security** | ✓ | Composed during Hadrian's reign (~119–122 AD), likely as a unified project presented to C. Septicius Clarus, praetorian prefect. The dedication to Septicius (who held office 119–122) provides a *terminus ante quem*. Scholars (Wallace-Hadrill 1983, Hurley 2001) agree the Lives were composed as a set within a relatively short period, not serially over many years. The dating is not as precise as Cicero's letters, but the *concentration* of composition is uncontested. |
| **B — Genre comparability** | ✓ | **Third-person biographical prose.** Closer to Caesar's *commentarius* than letters or speeches. Both are narrative Latin prose by elite Roman authors. The main difference: biography is structured by topic-rubric (appearance, virtues, vices) rather than chronological narrative. Register is more varied — Suetonius quotes documents and gossip. |
| **C — Corpus size & units** | ✓ | 8 Lives (Julius through Domitian), but the first two (Julius, Augustus) are much longer than the later ones. The Lives of Galba, Otho, Vitellius are very short fragments. The core 6–7 Lives (Julius/Domitian or Tiberius–Domitian) are a reasonable corpus. Each Life is a natural unit → n = 6–8. |
| **D — Text availability** | ✓✓ | Perseus has the complete *De Vita Caesarum*. TEI XML. |
| **E — Unit-of-analysis fit** | ✓ | Lives are natural units of ~5K–25K tokens each. Comparable to DBG books. Order the Lives by imperial succession (Julius → Augustus → … → Domitian). The key is that this ordering reflects *subject chronology* (who reigned when), not *composition chronology* (all were written together). If the method is truly chronometric, it should NOT detect drift across the Lives. If it detects drift, it's picking up either subject-content drift (later emperors are different topics) or stylistic choices that correlate with biographical subject. |

**Scholarly consensus on concentrated composition:** "It is generally agreed that the *Lives of the Caesars* was composed and published as a single work during the early years of Hadrian's reign" (Hurley, D.W., 2001. *Suetonius: Divus Claudius*. Cambridge, p. 4). Wallace-Hadrill (1983) argues that Suetonius' biographical method — a consistent topical rubrical structure applied to each emperor — reflects a single compositional campaign. The Lives were not written one at a time and published serially.

**Confound to acknowledge:** The Lives differ by *subject* (Julius' military campaigns ≠ Nero's artistic pretensions). Content-word features (MFW) will vary with topic. Function words and char n-grams should be more stable, but Suetonius' register may shift when treating "good" vs. "bad" emperors (moralizing vocabulary clusters). This is analogous to Caesar's topic confound across DBG campaigns.

---

### 4.3 Candidate N3: Sallust, Bellum Catilinae (The Catilinarian Conspiracy)

| Axis | Rating | Detail |
|------|--------|--------|
| **A — Date security** | ✓✓ | Composed and published ~42–40 BC (after Caesar's assassination, before Sallust's death in 35 BC). A single, concentrated work — Sallust's first historical monograph. No serial composition hypothesis exists. Universal scholarly consensus: composed as a unit. |
| **B — Genre comparability** | ✓✓ | **Excellent match.** Sallust writes third-person historical monograph in a deliberately archaising style — different from Caesar's plain style, but the *genre* (historical prose narrative) is the same. Both are elite Romans writing political-military history. Both write continuous narrative with speeches and digressions. |
| **C — Corpus size & units** | △ | Bellum Catilinae = ~61 chapters, ~11K total words (much shorter than DBG's 58K). Splitting into 7 pseudo-books yields units of ~1.6K words each — below the ~4K minimum of DBG's shortest book (Book III). Individual pseudo-books may be too short for reliable feature estimation. |
| **D — Text availability** | ✓✓ | Perseus has Sallust in TEI XML. |
| **E — Unit-of-analysis fit** | △ | The small total size (~11K words) limits how many pseudo-books can be created. With n = 7, each unit would be ~1.6K words — potentially noisy. With n = 5, each unit would be ~2.2K words — marginal. Joining *Bellum Catilinae* + *Bellum Iugurthinum* (~26K words, ~42–40 BC) into a combined pseudo-corpus could work (total ~37K words, comparable to DBC), but Sallust's two monographs have different topics. |

**Alternative:** Combine *Bellum Catilinae* and *Bellum Iugurthinum* (both composed ~42–40 BC, same author, same genre) into a single 7-pseudo-book corpus. Both are concentrated compositions from the same short period. The combined corpus would be ~37K words, matching DBC's size. The pseudo-book labels would be arbitrary (Catiline chapters spliced with Jugurtha chapters), so any detected "drift" would be a false positive.

---

## 5. Summary Table

### 5.1 Positive Controls

| Candidate | Date Security | Genre Match | Size & Units | Text Avail. | Unit Fit | **Overall** |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **P1 — Cicero Ad Atticum** | ✓✓ | △ | ✓✓ | ✓ | ✓ | **RECOMMENDED (primary)** |
| P2 — Cicero Speeches | ✓✓ | △ | ✓ | ✓✓ | △ | Viable secondary |
| P3 — Pliny Letters | △ | △ | ✓ | ✓ | △ | Not recommended (date/ordering dispute) |

### 5.2 Negative Controls

| Candidate | Date Security | Genre Match | Size & Units | Text Avail. | Unit Fit | **Overall** |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **N1 — DBC Pseudo-Books** | ✓✓ | ✓✓ | ✓ | ✓✓ | ✓✓ | **RECOMMENDED (primary)** |
| N2 — Suetonius Lives | ✓ | ✓ | ✓ | ✓✓ | ✓ | Recommended (secondary) |
| N3 — Sallust Catiline | ✓✓ | ✓✓ | △ | ✓✓ | △ | Viable if combined with Jugurtha |

---

## 6. Recommendations

### 6.1 Primary Positive Control: Cicero, Epistulae ad Atticum

**Justification:** The dating is as secure as classical studies offers. With ~20 yearly bins, the statistical power exceeds Caesar's n = 7 by a factor of 3, giving the calibration high sensitivity. The genre confound (letters ≠ commentarius) is real but mitigable: (a) function words and char n-grams are less register-sensitive than MFW; (b) Cicero is Caesar's exact contemporary, so the underlying Latin grammatical system is identical; (c) the calibration question is not "does Cicero drift the same way Caesar does?" but "does the method detect drift when drift is known to exist?" — a different standard than genre-matching.

**Unit of analysis:** Yearly bins. Letters grouped by assigned year → ~20 year-units, each containing multiple letters (total tokens per year: several thousand). Years with fewer than 1,000 total tokens should be merged with adjacent years.

**Anchor:** The DBC-analogue for Cicero would be his **latest securely dated work.** Candidate: *De Officiis* (44 BC, his last philosophical treatise) — or the latest year's letters (late 44–43 BC). A multi-work anchor (combining *De Officiis*, *Philippics*, and *De Amicitia/De Senectute* into a "Late Cicero" aggregate) would provide more tokens and a more robust reference point. This mirrors the DBC-as-aggregate approach used in the Caesar analysis.

**Expected result:** Strong negative Spearman r on the DBC-style anchor (early Cicero years → far from late-Cicero anchor; later years → closer). Positive Mantel r. p < 0.01 given ~20 units.

**Fallback if result is null:** If Cicero's letters show no drift despite known serial composition, the method lacks sensitivity for chronological signal. This would undermine (though not necessarily refute) the Caesar result — it could mean Caesar's signal is genuine while Cicero's familiar-letter style is too variable within-year to show cross-year drift, but it would raise a serious methodological question.

### 6.2 Primary Negative Control: DBC Pseudo-Books (Narrative Order)

**Justification:** This is the most informative negative control possible: **same author, same genre, same register, same feature-extraction pipeline, same statistical tests.** The only difference is whether the segmentation order reflects multi-year composition chronology (DBG) or arbitrary sequential chunks of a single concentrated work (DBC pseudo-books). If the method works, DBG shows drift and DBC pseudo-books show none. If both show drift, the method is detecting narrative-structure confound rather than chronometric signal.

**Unit of analysis:** 7 pseudo-books of equal token count, segmented sequentially along DBC's narrative order (chapters 1→243). Each pseudo-book = ~5.3K tokens.

**Anchor:** Pseudo-book 7 (the final segment) serves as the DBC-anchor analogue.

**Expected result:** Spearman r ≈ 0 on the DBC-anchor test. Mantel r ≈ 0. p > 0.20. Random shuffling variant (N1b) should produce even flatter results.

**If N1a shows a signal:** This would mean sequential narrative progression within a single work produces a chronometric-like signal in our feature sets. This would not necessarily refute the Caesar result (DBG books have much larger stylistic differences than sequential chunks of DBC), but it would force a more cautious interpretation — specifically, it would suggest that part of the DBG annual signal reflects narrative arc rather than composition date. This is a genuinely important methodological discovery either way.

### 6.3 Secondary Negative Control: Suetonius, De Vita Caesarum

**Justification:** A natural multi-book corpus with uncontested concentrated composition. The imperial-succession ordering is not a composition-chronology ordering — if the method is working, it should not detect drift. The genre is closer to Caesar than letters or speeches.

**Unit of analysis:** Individual Lives, ordered by imperial reign (Julius → Augustus → … → Domitian). Julius and Augustus are substantially longer → consider using only Tiberius–Domitian (6 Lives) for more balanced unit sizes.

**Anchor:** The last Life (Domitian) serves as the DBC-anchor analogue.

**Expected result:** r ≈ 0. p > 0.10.

**Confound to report:** Lives differ by subject-matter. Function words should mitigate this. If MFW shows false drift but function words do not, this further validates the function-word feature choice for the Caesar analysis.

---

## 7. Implementation Plan

### 7.1 Text Acquisition

All candidates are available from Perseus. Manual download (as done for Caesar):

| Corpus | Perseus filename / URL |
|--------|----------------------|
| Cicero, Ad Atticum | Available via PHI Latin corpus or Latin Library. Perseus has partial. |
| DBC pseudo-books | Already in repository: `data/raw/perseus/caes.bc_lat.xml` |
| Suetonius, De Vita Caesarum | `suet.caes_lat.xml` (Perseus) |

### 7.2 Pipeline Adaptation

The existing Caesar pipeline (Scripts 01–17) is designed to be corpus-agnostic at the feature-extraction and analysis stages. The adaptation requires:

1. **New parsing scripts** (minor variants of Scripts 01–03) to handle: Cicero letters (different TEI structure, letter-level segmentation); Suetonius Lives (Life-level segmentation).
2. **New metadata files** mapping each unit to its assigned year/order.
3. **The same feature-extraction scripts** (07–08) applied unchanged.
4. **The same analysis scripts** (13, 14, 18) applied unchanged — they read feature CSVs and year data, and the statistical code is corpus-blind.

### 7.3 Testing Protocol

For each calibration corpus, report:

| Test | Statistic | Expected (Positive) | Expected (Negative) |
|------|-----------|--------------------|--------------------|
| DBC-style Anchor | Spearman r (p-value) | r ≪ 0, p < 0.05 | r ≈ 0, p > 0.20 |
| Mantel | Pearson r (p-value) | r ≫ 0, p < 0.05 | r ≈ 0, p > 0.20 |
| Jackknife | Direction preserved? | Yes | Direction may flip (no signal to preserve) |

Report all results with exact-permutation p-values and bootstrap 95% CIs, exactly as done for Caesar.

---

## 8. What Success Looks Like

| Scenario | Caesar DBG | Positive Control | Negative Control | Interpretation |
|----------|-----------|-----------------|-----------------|----------------|
| **Ideal** | Annual signal ✓ | Strong drift ✓ | Null ✗ | Method is valid. Caesar result is credible. |
| **Troubling** | Annual signal ✓ | Null ✗ | Null ✗ | Method may lack sensitivity. Caesar result could be false positive. |
| **Damaging** | Annual signal ✓ | Strong drift ✓ | Strong drift ✓ | Method detects narrative structure, not chronology. Caesar result is artefactual. |
| **Null Caesar** | Null ✗ | Strong drift ✓ | Null ✗ | Caesar genuinely shows no drift. Method works; result favours Hypothesis B. |

---

## 9. Pre-Registration

This document serves as a pre-registration of the calibration design. The expected results for each candidate are stated above (Section 6). All results — whether they match expectations or not — will be reported. The calibration analysis will be run exactly as specified in Section 7, with no post-hoc adjustments to the method based on calibration outcomes.

---

*Design document: `docs/calibration_design.md`*
*Next step: decision on which candidates to pursue, followed by text acquisition and implementation.*
