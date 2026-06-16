#!/usr/bin/env python3
"""
scripts/20_calibration_corpus.py

Builds calibration corpora to the SAME schema as the Caesar corpus so
that existing feature-extraction and analysis code applies unchanged.

Corpora built:
  1. Cicero, Epistulae ad Atticum (PRIMARY POSITIVE CONTROL)
     - 426 letters across 16 books, 68–44 BC
     - Dates from <date value="-65"> attributes; fallback to <dateline> text
     - Output: yearly-binned units (~20 year-units)

  2. DBC Pseudo-Books (PRIMARY NEGATIVE CONTROL)
     - De Bello Civili split into 7 equal-token sequential chunks
     - Same author, same genre, same register as DBG
     - Narrative-order chunks test for narrative-structure confound

  3. Suetonius, De Vita Caesarum (SECONDARY NEGATIVE CONTROL)
     - Requires manual download: search Perseus for 'suet.caes_lat.xml'
     - If XML present: 8 Lives, ordered by imperial succession
     - If absent: prints clear download instructions and skips

Output schema (matching Caesar corpus):
  data/corpus/calib_<author>_units.csv
  data/corpus/calib_<author>_units_normalized.csv
  data/corpus/calib_<author>_units_normalized_lemmatized.csv

Columns: work, unit_id, order_index, known_date, date_source, text
"""

import csv
import os
import re
import sys
import logging
from collections import OrderedDict

import lxml.etree as ET

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR      = os.path.join(DATA_DIR, 'raw', 'perseus')
CORPUS_DIR   = os.path.join(DATA_DIR, 'corpus')
LOG_DIR  = os.path.join(PROJECT_ROOT, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'calibration_build.log')

os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# ORTHOGRAPHIC NORMALIZATION (same as Caesar pipeline)
# ═══════════════════════════════════════════════════════════════════════
def normalize_latin(text):
    """Lowercase + j→i, v→u (standard Latin text normalization)."""
    text = text.lower()
    text = text.replace('j', 'i')
    text = text.replace('v', 'u')
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════
def extract_text_tei(element):
    """
    Extract all text from a TEI element and its descendants.
    Handles Perseus TEI.2 quirks:
      - <reg> wraps regularised readings → include text
      - <del> wraps deleted text → skip
      - <foreign> wraps Greek/foreign → keep (it's part of the text)
      - <gap/> → skip
      - <milestone/> → skip (no text)
      - <pb/> → skip
      - <hi> → keep text
      - <quote> → keep text
      - <name>, <placeName>, <num>, <abbr> → keep text
      - <date> → skip (the date element contains chronological metadata,
                 not narrative text)
      - <opener>, <dateline>, <salute> → skip (paratext)
    """
    SKIP_TAGS = {'del', 'gap', 'milestone', 'pb', 'date', 'dateline',
                 'opener', 'salute', 'closer', 'signed', 'address'}
    parts = []

    if element.text and element.tag not in SKIP_TAGS:
        # Only add text if parent is not a skip tag
        parent_tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        if parent_tag not in SKIP_TAGS:
            parts.append(element.text)

    for child in element:
        child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if child_tag not in SKIP_TAGS:
            parts.append(extract_text_tei(child))
        if child.tail:
            parts.append(child.tail)

    return ''.join(parts)


def clean_text(text):
    """Clean extracted text: remove extra whitespace, collapse newlines."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


# ═══════════════════════════════════════════════════════════════════════
# 1. CICERO — EPISTULAE AD ATTICUM (POSITIVE CONTROL)
# ═══════════════════════════════════════════════════════════════════════
def parse_cicero_atticum(xml_path):
    """
    Parse Perseus TEI.2 Cicero Ad Atticum.
    Returns list of dicts: {unit_id, order_index, known_date, date_source, text}
    """
    log.info("Parsing Cicero, Epistulae ad Atticum...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    letters = []
    date_missing = []
    date_issues = []

    # Navigate: TEI.2 → text → body → div1(@type='book') → div2(@type='letter')
    body = root.find('.//body')
    if body is None:
        log.error("No <body> found in Cicero XML")
        return [], [], []

    books = body.findall('.//div1[@type="book"]')
    log.info(f"  Found {len(books)} books")

    letter_counter = 0
    for book_div in books:
        book_num = book_div.get('n', '?')
        letters_div = book_div.findall('.//div2[@type="letter"]')

        for letter_div in letters_div:
            letter_num = letter_div.get('n', '?')
            letter_counter += 1

            # Build standard unit_id: att_1_1, att_1_2, ...
            unit_id = f"att_{book_num}_{letter_num}"

            # ── DATE EXTRACTION ────────────────────────────────
            known_date = None
            date_source = ''

            # Strategy 1: <date> element with 'value' attribute inside the letter
            dates = letter_div.findall('.//date')
            date_values = []
            for d in dates:
                val = d.get('value', '')
                auth = d.get('authname', '')
                if val:
                    try:
                        date_values.append(int(val))
                    except ValueError:
                        pass
                elif auth:
                    try:
                        date_values.append(int(auth))
                    except ValueError:
                        pass

            if date_values:
                # Use the earliest date reference in the letter as the composition date
                known_date = min(date_values)  # negative for BC
                date_source = f'<date value="{known_date}">'

            # Strategy 2: <dateline> text (e.g., "Scr. Romae...")
            if known_date is None:
                dateline = letter_div.find('.//dateline')
                if dateline is not None and dateline.text:
                    dl_text = dateline.text.strip()
                    # Try to parse Roman consular year from dateline text
                    # e.g., "L. Iulio Caesare, C. Marcio Figulo consulibus"
                    # These are too complex to parse reliably — flag for manual check
                    date_source = f'<dateline>{dl_text[:80]}'
                    date_issues.append((unit_id, dl_text[:100]))

            if known_date is None:
                date_missing.append(unit_id)
                known_date = None
                date_source = 'MISSING'

            # ── TEXT EXTRACTION ────────────────────────────────
            # Get all <p> elements within the letter
            paragraphs = letter_div.findall('.//p')
            text_parts = []
            for p in paragraphs:
                p_text = extract_text_tei(p)
                p_text = clean_text(p_text)
                if p_text:
                    text_parts.append(p_text)

            full_text = ' '.join(text_parts)

            if len(full_text) < 50:
                log.warning(f"  Very short letter: {unit_id} ({len(full_text)} chars)")
                continue  # skip fragments

            letters.append({
                'work': 'cicero_atticum',
                'unit_id': unit_id,
                'order_index': known_date if known_date is not None else None,
                'known_date': known_date,
                'date_source': date_source,
                'text': full_text,
            })

    log.info(f"  Total letters extracted: {letter_counter}")
    log.info(f"  Letters with dates: {len(letters) - len(date_missing)}")
    log.info(f"  Letters with MISSING dates: {len(date_missing)}")
    if date_issues:
        log.info(f"  Letters with dateline-only dates (unparsed): {len(date_issues)}")

    # Sort by date
    letters.sort(key=lambda x: (x['known_date'] if x['known_date'] is not None
                                 else -9999))

    return letters, date_missing, date_issues


def build_yearly_units(letters):
    """
    Aggregate individual letters into yearly bins.
    Each year-unit contains all letters from that year.
    Returns list of dicts: {unit_id, order_index, known_date, text}
    """
    from collections import defaultdict

    year_bins = defaultdict(list)
    undated = []

    for letter in letters:
        if letter['known_date'] is not None:
            year_bins[letter['known_date']].append(letter)
        else:
            undated.append(letter)

    years = sorted(year_bins.keys())
    log.info(f"\n  Yearly bins: {len(years)} years from {years[0]} to {years[-1]}")
    log.info(f"  Undated letters (excluded from bins): {len(undated)}")

    units = []
    for idx, year in enumerate(years):
        year_letters = year_bins[year]
        combined_text = '\n\n'.join(l['text'] for l in year_letters)
        total_chars = sum(len(l['text']) for l in year_letters)
        total_tokens = len(combined_text.split())

        unit_id = f"cicero_year_{abs(year)}bc"

        units.append({
            'work': 'cicero_atticum_yearly',
            'unit_id': unit_id,
            'order_index': idx + 1,          # 1-indexed chronological order
            'known_date': year,              # negative = BC
            'date_source': f'Yearly bin: {len(year_letters)} letters',
            'text': combined_text,
            'n_letters': len(year_letters),
            'total_tokens': total_tokens,
        })

        log.info(f"    {unit_id}: {len(year_letters)} letters, "
                 f"{total_tokens} tokens")

    return units


# ═══════════════════════════════════════════════════════════════════════
# 2. DBC PSEUDO-BOOKS (PRIMARY NEGATIVE CONTROL)
# ═══════════════════════════════════════════════════════════════════════
def build_dbc_pseudo_books(xml_path, n_books=7):
    """
    Split De Bello Civili into n equal-token sequential pseudo-books.
    Uses existing caes.bc_lat.xml.

    Strategy:
      - Parse all chapters from DBC via the existing DBG parser approach
        (div2 chapters with sequential numbering)
      - Concatenate chapter texts in narrative order
      - Split into n chunks of approximately equal token count

    Returns list of dicts: {unit_id, order_index, known_date, text}
    """
    log.info("\nBuilding DBC pseudo-books (negative control)...")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # DBC uses div2 for chapters (same as Caesar DBG in 02_parse_tei.py)
    # Navigate: TEI → text → body → div[@type='book'] → div2
    # But DBC may be structured differently — let's be flexible

    chapters = []

    # Try div2 structure
    div2s = root.findall('.//div2')
    if div2s:
        for div2 in div2s:
            # Skip prefatory / non-chapter divs
            div_type = div2.get('type', '')
            if div_type and div_type not in ('chapter', 'section', ''):
                continue

            text = extract_text_tei(div2)
            text = clean_text(text)
            if len(text) > 100:
                chapters.append(text)
    else:
        # Fallback: div1 → div or milestone-based parsing
        # Try to extract all sections
        body = root.find('.//body')
        if body is not None:
            # Just grab all paragraphs as a single flow, then chunk by sentence
            all_text = extract_text_tei(body)
            all_text = clean_text(all_text)
            # Split into chapters by milestone or section break patterns
            # For DBC, chapters are typically numbered
            paragraphs = body.findall('.//p')
            for p in paragraphs:
                text = extract_text_tei(p)
                text = clean_text(text)
                if len(text) > 100:
                    chapters.append(text)

    if not chapters:
        log.error("No chapters extracted from DBC XML")
        return []

    log.info(f"  Extracted {len(chapters)} chapter-like segments from DBC")

    # Concatenate and split into n equal-token chunks
    all_chapter_texts = [c for c in chapters if c.strip()]
    full_text = ' '.join(all_chapter_texts)
    tokens = full_text.split()
    total_tokens = len(tokens)

    tokens_per_book = total_tokens // n_books
    log.info(f"  Total tokens: {total_tokens} → {tokens_per_book} per pseudo-book")

    pseudo_books = []
    for i in range(n_books):
        start = i * tokens_per_book
        end   = start + tokens_per_book if i < n_books - 1 else total_tokens
        book_tokens = tokens[start:end]
        text = ' '.join(book_tokens)

        pseudo_books.append({
            'work': 'dbc_pseudo_books',
            'unit_id': f'dbc_pseudo_{i + 1:02d}',
            'order_index': i + 1,
            'known_date': None,  # narrative order, NOT a date
            'date_source': f'DBC narrative-order pseudo-book {i + 1}/{n_books}',
            'text': text,
            'token_count': len(book_tokens),
        })

        log.info(f"    {pseudo_books[-1]['unit_id']}: "
                 f"{len(book_tokens)} tokens")

    return pseudo_books


# ═══════════════════════════════════════════════════════════════════════
# 3. SUETONIUS — DE VITA CAESARUM (SECONDARY NEGATIVE CONTROL)
# ═══════════════════════════════════════════════════════════════════════
def parse_suetonius(xml_path):
    """
    Parse Suetonius De Vita Caesarum (if available).
    Returns list of dicts: {unit_id, order_index, known_date, text}
    """
    log.info("\nParsing Suetonius, De Vita Caesarum...")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Suetonius is typically structured as:
    # TEI → text → body → div[@type='book' or @type='life'] → chapters

    lives = []
    imperial_order = [
        'julius', 'augustus', 'tiberius', 'gaius', 'caligula',
        'claudius', 'nero', 'galba', 'otho', 'vitellius',
        'vespasian', 'titus', 'domitian'
    ]

    for div in root.findall('.//div[@type]'):
        div_type = div.get('type', '')
        if div_type not in ('book', 'life', 'section'):
            continue
        n = div.get('n', '')
        head = div.find('head')
        label = ''
        if head is not None and head.text:
            label = head.text.strip().lower()

        text = extract_text_tei(div)
        text = clean_text(text)
        if len(text) < 500:
            continue

        # Determine order_index from imperial succession
        order = None
        for idx, name in enumerate(imperial_order):
            if name in label or name in n.lower():
                order = idx + 1
                break

        unit_id = f"suet_{n.replace(' ', '_').lower()}"

        lives.append({
            'work': 'suetonius',
            'unit_id': unit_id,
            'order_index': order,
            'known_date': None,   # all composed ~119–122 AD
            'date_source': f'Imperial succession order',
            'text': text,
        })

    log.info(f"  Extracted {len(lives)} Lives")
    return lives


# ═══════════════════════════════════════════════════════════════════════
# LEMMATIZATION (same CLTK pipeline as Caesar)
# ═══════════════════════════════════════════════════════════════════════
def lemmatize_corpus(units, corpus_name):
    """
    Apply CLTK lemmatization to each unit.
    Adds 'tokens' and 'lemmas' columns.
    Falls back gracefully if CLTK/Stanza is unavailable or stalls.
    """
    log.info(f"\nLemmatizing {corpus_name} ({len(units)} units)...")

    try:
        from cltk import NLP
        nlp = NLP(language_code="lat", suppress_banner=True)
    except Exception as e:
        log.warning(f"  CLTK NLP unavailable: {e}")
        log.warning(f"  Skipping lemmatization — normalized CSV only.")
        return units, False

    lemmatized_count = 0
    for i, unit in enumerate(units):
        text = unit['text']
        if len(text) < 50:
            continue

        try:
            doc = nlp.analyze(text=text[:8000])  # safety cap
            tokens = [w.string for w in doc.words]
            lemmas = [w.lemma if w.lemma else w.string for w in doc.words]
            unit['tokens'] = ' '.join(tokens)
            unit['lemmas'] = ' '.join(lemmas)
            lemmatized_count += 1
        except Exception as e:
            log.warning(f"  Lemmatization failed for {unit['unit_id']}: {e}")
            unit['tokens'] = text
            unit['lemmas'] = text

        if (i + 1) % 5 == 0:
            log.info(f"    {i + 1}/{len(units)} lemmatized...")

    log.info(f"  Lemmatized: {lemmatized_count}/{len(units)} units")
    return units, True


# ═══════════════════════════════════════════════════════════════════════
# CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════
def write_corpus_csv(units, filename, has_date=True, has_lemmas=False):
    """Write corpus units to CSV matching Caesar schema."""
    path = os.path.join(CORPUS_DIR, filename)

    if has_lemmas:
        fieldnames = ['work', 'unit_id', 'order_index', 'known_date',
                      'date_source', 'text', 'tokens', 'lemmas']
    elif has_date:
        fieldnames = ['work', 'unit_id', 'order_index', 'known_date',
                      'date_source', 'text']
    else:
        fieldnames = ['work', 'unit_id', 'order_index', 'date_source', 'text']

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        for unit in units:
            row = {k: unit.get(k, '') for k in fieldnames}
            writer.writerow(row)

    log.info(f"  Wrote: {path} ({len(units)} rows)")
    return path


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════
def validate_units(units, corpus_name):
    """Print validation summary for a set of units."""
    log.info(f"\n{'=' * 50}")
    log.info(f"  VALIDATION: {corpus_name}")
    log.info(f"{'=' * 50}")
    log.info(f"  Total units: {len(units)}")

    if not units:
        log.warning("  NO UNITS — corpus is empty!")
        return

    token_counts = [len(u['text'].split()) for u in units]
    log.info(f"  Token counts — min: {min(token_counts)}, "
             f"median: {sorted(token_counts)[len(token_counts)//2]}, "
             f"max: {max(token_counts)}")
    log.info(f"  Units < 1000 tokens: "
             f"{sum(1 for t in token_counts if t < 1000)}")

    # Date range
    dates = [u['known_date'] for u in units
             if u.get('known_date') is not None]
    if dates:
        log.info(f"  Date range: {min(dates)} to {max(dates)} "
                 f"({max(dates) - min(dates)} years)")
        log.info(f"  Units with missing dates: "
                 f"{len(units) - len(dates)}/{len(units)}")

    # Order index check
    order_indices = [u.get('order_index') for u in units]
    missing_order = sum(1 for o in order_indices if o is None)
    if missing_order:
        log.warning(f"  Units with missing order_index: {missing_order}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("  CALIBRATION CORPUS BUILDER")
    log.info("=" * 60)

    results_summary = []

    # ════════════════════════════════════════════════════════════════
    # 1. CICERO — POSITIVE CONTROL
    # ════════════════════════════════════════════════════════════════
    cicero_xml = os.path.join(RAW_DIR, 'cicero_atticum.xml')

    if os.path.exists(cicero_xml):
        letters, date_missing, date_issues = parse_cicero_atticum(cicero_xml)

        if letters:
            # Save individual letters
            write_corpus_csv(letters, 'calib_cicero_atticum_letters.csv')
            validate_units(letters, "Cicero Ad Atticum (letters)")

            # Build yearly bins
            yearly_units = build_yearly_units(letters)
            write_corpus_csv(yearly_units, 'calib_cicero_atticum_yearly.csv')
            validate_units(yearly_units, "Cicero Ad Atticum (yearly)")

            # Normalize
            for u in yearly_units:
                u['text'] = normalize_latin(u['text'])
            write_corpus_csv(yearly_units,
                             'calib_cicero_atticum_yearly_normalized.csv')

            # Lemmatize
            yearly_units, lem_ok = lemmatize_corpus(
                yearly_units, "Cicero Ad Atticum yearly")
            if lem_ok:
                write_corpus_csv(yearly_units,
                                 'calib_cicero_atticum_yearly_normalized_lemmatized.csv',
                                 has_lemmas=True)

            results_summary.append({
                'corpus': 'Cicero Ad Atticum (yearly)',
                'units': len(yearly_units),
                'date_range': f"{yearly_units[0]['known_date']} to "
                              f"{yearly_units[-1]['known_date']}",
                'status': 'OK',
            })

            # Flag date issues
            if date_missing:
                log.warning(f"\n  ⚠ {len(date_missing)} letters have MISSING dates:")
                for dm in date_missing[:10]:
                    log.warning(f"    {dm}")
                if len(date_missing) > 10:
                    log.warning(f"    ... and {len(date_missing) - 10} more")
    else:
        log.error(f"\n  ✗ Cicero XML not found: {cicero_xml}")
        log.error(f"    Download from Perseus: Cicero, Epistulae ad Atticum")
        log.error(f"    Save as: {cicero_xml}")
        results_summary.append({
            'corpus': 'Cicero Ad Atticum',
            'units': 0,
            'date_range': 'N/A',
            'status': 'XML NOT FOUND',
        })

    # ════════════════════════════════════════════════════════════════
    # 2. DBC PSEUDO-BOOKS — NEGATIVE CONTROL
    # ════════════════════════════════════════════════════════════════
    dbc_xml = os.path.join(RAW_DIR, 'caes.bc_lat.xml')

    if os.path.exists(dbc_xml):
        pseudo_books = build_dbc_pseudo_books(dbc_xml, n_books=7)

        if pseudo_books:
            write_corpus_csv(pseudo_books, 'calib_dbc_pseudo_books.csv',
                             has_date=False)
            validate_units(pseudo_books, "DBC Pseudo-Books")

            # Normalize
            for u in pseudo_books:
                u['text'] = normalize_latin(u['text'])
            write_corpus_csv(pseudo_books,
                             'calib_dbc_pseudo_books_normalized.csv',
                             has_date=False)

            # Lemmatize
            pseudo_books, lem_ok = lemmatize_corpus(
                pseudo_books, "DBC Pseudo-Books")
            if lem_ok:
                write_corpus_csv(pseudo_books,
                                 'calib_dbc_pseudo_books_normalized_lemmatized.csv',
                                 has_date=False, has_lemmas=True)

            results_summary.append({
                'corpus': 'DBC Pseudo-Books (neg control)',
                'units': len(pseudo_books),
                'date_range': 'N/A (narrative order)',
                'status': 'OK',
            })
    else:
        log.error(f"\n  ✗ DBC XML not found: {dbc_xml}")
        results_summary.append({
            'corpus': 'DBC Pseudo-Books',
            'units': 0,
            'date_range': 'N/A',
            'status': 'XML NOT FOUND',
        })

    # ════════════════════════════════════════════════════════════════
    # 3. SUETONIUS — SECONDARY NEGATIVE CONTROL
    # ════════════════════════════════════════════════════════════════
    suet_xml = os.path.join(RAW_DIR, 'suet.caes_lat.xml')
    # Also try alternative filename
    suet_xml_alt = os.path.join(RAW_DIR, 'suetonius_lives.xml')

    suet_path = None
    if os.path.exists(suet_xml):
        suet_path = suet_xml
    elif os.path.exists(suet_xml_alt):
        suet_path = suet_xml_alt

    if suet_path:
        lives = parse_suetonius(suet_path)

        if lives:
            write_corpus_csv(lives, 'calib_suetonius_lives.csv',
                             has_date=False)
            validate_units(lives, "Suetonius De Vita Caesarum")

            # Normalize
            for u in lives:
                u['text'] = normalize_latin(u['text'])
            write_corpus_csv(lives, 'calib_suetonius_lives_normalized.csv',
                             has_date=False)

            # Lemmatize
            lives, lem_ok = lemmatize_corpus(lives, "Suetonius Lives")
            if lem_ok:
                write_corpus_csv(lives,
                                 'calib_suetonius_lives_normalized_lemmatized.csv',
                                 has_date=False, has_lemmas=True)

            results_summary.append({
                'corpus': 'Suetonius Lives (neg control)',
                'units': len(lives),
                'date_range': '119–122 AD (all)',
                'status': 'OK',
            })
    else:
        log.warning(f"\n  ⚠ Suetonius XML not found.")
        log.warning(f"    Checked: {suet_xml}")
        log.warning(f"    Checked: {suet_xml_alt}")
        log.warning(f"")
        log.warning(f"    DOWNLOAD INSTRUCTIONS:")
        log.warning(f"    1. Go to: https://github.com/PerseusDL/canonical-latinLit")
        log.warning(f"       or: https://catalog.perseus.org/")
        log.warning(f"    2. Search for: Suetonius, De Vita Caesarum")
        log.warning(f"    3. Download the TEI XML file")
        log.warning(f"    4. Save as: {suet_xml}")
        log.warning(f"    5. Re-run: python scripts/20_calibration_corpus.py")
        results_summary.append({
            'corpus': 'Suetonius Lives',
            'units': 0,
            'date_range': 'N/A',
            'status': 'XML NOT FOUND — see download instructions above',
        })

    # ════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════
    log.info(f"\n{'=' * 60}")
    log.info(f"  BUILD SUMMARY")
    log.info(f"{'=' * 60}")
    log.info(f"  {'Corpus':<35s} {'Units':>6s} {'Date Range':<25s} {'Status':<20s}")
    log.info(f"  {'─' * 35} {'─' * 6} {'─' * 25} {'─' * 20}")
    for r in results_summary:
        log.info(f"  {r['corpus']:<35s} {r['units']:>6d} "
                 f"{r['date_range']:<25s} {r['status']:<20s}")
    log.info(f"\n  All CSVs saved to: {CORPUS_DIR}/")
    log.info(f"  Build log: {LOG_FILE}")
    log.info(f"\n{'=' * 60}")
    log.info(f"  CORPUS BUILD COMPLETE")
    log.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
