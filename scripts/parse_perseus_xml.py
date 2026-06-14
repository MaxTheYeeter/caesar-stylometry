#!/usr/bin/env python3

"""
parse_perseus_xml.py

Extract chapter-level Latin text from Perseus TEI XML.

Handles:

De Bello Civili
    div1(book) -> div2(chapter)

De Bello Gallico
    div1(Book) with milestone unit="chapter"

Output:
    data/processed/chapters/
"""

from pathlib import Path
from lxml import etree
import re

INPUT_DIR = Path("data/raw/perseus")
OUTPUT_DIR = Path("data/processed/chapters")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CLEANING
# --------------------------------------------------

def clean_text(text):

    if not text:
        return ""

    # Remove editorial square brackets
    text = re.sub(r"\[[^\]]*\]", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def safe_id(value):

    if value is None:
        return "unknown"

    value = value.strip()

    if value.isdigit():
        return f"{int(value):03d}"

    value = value.replace("(", "_")
    value = value.replace(")", "")

    return value


# --------------------------------------------------
# DE BELLO CIVILI
# --------------------------------------------------

def parse_civili(root, work_code):

    books = root.xpath(".//div1[@type='book']")

    exported = 0

    for book in books:

        book_n = int(book.get("n"))

        chapters = book.xpath("./div2[@type='chapter']")

        print(
            f"  Book {book_n}: {len(chapters)} chapters"
        )

        for chapter in chapters:

            chapter_n = chapter.get("n")

            chapter_copy = etree.fromstring(
                etree.tostring(chapter)
            )

            etree.strip_elements(
                chapter_copy,
                "note",
                "milestone",
                "pb",
                "lb",
                with_tail=False
            )

            text = " ".join(
                chapter_copy.itertext()
            )

            text = clean_text(text)

            filename = (
                f"{work_code}_"
                f"book{book_n:02d}_"
                f"ch{safe_id(chapter_n)}.txt"
            )

            with open(
                OUTPUT_DIR / filename,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(text)

            exported += 1

    return exported


# --------------------------------------------------
# DE BELLO GALLICO
# --------------------------------------------------

def parse_gallico(root, work_code):

    books = root.xpath(".//div1")

    exported = 0

    for book in books:

        book_n = int(book.get("n"))

        print(f"  Book {book_n}")

        current_chapter = None
        chapter_text = []

        for elem in book.iter():

            if elem.tag == "milestone":

                if elem.get("unit") == "chapter":

                    if current_chapter is not None:

                        text = clean_text(
                            " ".join(chapter_text)
                        )

                        if text:

                            filename = (
                                f"{work_code}_"
                                f"book{book_n:02d}_"
                                f"ch{safe_id(current_chapter)}.txt"
                            )

                            with open(
                                OUTPUT_DIR / filename,
                                "w",
                                encoding="utf-8"
                            ) as f:

                                f.write(text)

                            exported += 1

                    current_chapter = elem.get("n")
                    chapter_text = []

            if elem.text:
                chapter_text.append(elem.text)

            if elem.tail:
                chapter_text.append(elem.tail)

        # final chapter

        if current_chapter is not None:

            text = clean_text(
                " ".join(chapter_text)
            )

            if text:

                filename = (
                    f"{work_code}_"
                    f"book{book_n:02d}_"
                    f"ch{safe_id(current_chapter)}.txt"
                )

                with open(
                    OUTPUT_DIR / filename,
                    "w",
                    encoding="utf-8"
                ) as f:

                    f.write(text)

                exported += 1

    return exported


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def process_file(xml_path):

    print(f"\nProcessing: {xml_path.name}")

    parser = etree.XMLParser(recover=True)

    tree = etree.parse(
        str(xml_path),
        parser
    )

    root = tree.getroot()

    filename = xml_path.name.lower()

    if "bc" in filename:

        print("Detected De Bello Civili")

        exported = parse_civili(
            root,
            "dbc"
        )

    elif "bg" in filename:

        print("Detected De Bello Gallico")

        exported = parse_gallico(
            root,
            "dbg"
        )

    else:

        print(
            "Unknown work type, skipping."
        )

        return

    print(
        f"Exported {exported} chapters"
    )


def main():

    xml_files = sorted(
        INPUT_DIR.glob("*.xml")
    )

    for xml_file in xml_files:

        try:

            process_file(xml_file)

        except Exception as e:

            print(
                f"ERROR: {xml_file.name}"
            )

            print(e)

    print("\nFinished.")


if __name__ == "__main__":
    main()