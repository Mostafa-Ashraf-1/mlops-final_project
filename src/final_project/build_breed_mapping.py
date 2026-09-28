"""Build a unique breed-to-animal-type mapping from Oxford Pet XML annotations.

The notebook's ``parse_breed`` function determines the breed from an annotation
file name, and each XML file supplies the matching animal type (``cat`` or
``dog``) at ``object/name``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_breed(fname: str) -> str:
    """Return the breed portion of a pet filename.

    This is the same parsing logic used in ``notebooks/notebook.ipynb``.  The
    basename is used so the function works with either a filename or a path.
    """
    parts = Path(fname).stem.split("_")
    return " ".join(parts[:-1])


def build_breed_mapping(xml_directory: Path) -> dict[str, str]:
    """Read XML annotations and return one ``breed: cat|dog`` entry per breed.

    Duplicate annotations for the same breed are collapsed.  A conflicting type
    for one breed is an invalid dataset and raises ``ValueError`` instead of
    silently emitting an incorrect JSON mapping.
    """
    mapping: dict[str, str] = {}

    for xml_file in sorted(xml_directory.glob("*.xml")):
        breed = parse_breed(xml_file.name)
        animal_type = ET.parse(xml_file).findtext("object/name")
        if animal_type not in {"cat", "dog"}:
            raise ValueError(
                f"{xml_file}: expected object/name to be 'cat' or 'dog', got {animal_type!r}"
            )

        existing_type = mapping.setdefault(breed, animal_type)
        if existing_type != animal_type:
            raise ValueError(
                f"Conflicting animal types for breed {breed!r}: "
                f"{existing_type!r} and {animal_type!r}"
            )

    return mapping


def write_breed_mapping(xml_directory: Path, output_file: Path) -> dict[str, str]:
    """Build the mapping and write it as a sorted, duplicate-free JSON object."""
    mapping = build_breed_mapping(xml_directory)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return mapping


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "xml_directory",
        nargs="?",
        type=Path,
        default=project_root / "Data/oxford-iiit-pet/annotations/xmls",
        help="directory containing Oxford Pet XML files",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        type=Path,
        default=project_root / "artifacts/breed_animal_types.json",
        help="JSON file to create",
    )
    args = parser.parse_args()

    mapping = write_breed_mapping(args.xml_directory, args.output_file)
    print(f"Wrote {len(mapping)} unique breed mappings to {args.output_file}")


if __name__ == "__main__":
    main()
