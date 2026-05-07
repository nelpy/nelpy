"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
mod_symbol = ""

root = Path(__file__).parent.parent
src = root

for path in sorted(src.rglob("*.py")):
    if path.relative_to(src).parts[0] != "nelpy":
        continue

    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    # Skip __init__.py files entirely (don't document them)
    if parts[-1] == "__init__":
        continue

    # Skip temporary files
    if parts[-1] == "temp":
        continue

    # Skip private modules (starting with underscore)
    elif parts[-1].startswith("_"):
        continue

    # Skip known problematic modules
    if "contrib" in parts:
        continue

    # skip homography files
    if "homography" in parts:
        continue

    # skip version.py
    if parts[-1] == "version":
        continue

    nav_parts = [f"{mod_symbol} {part}" for part in parts]
    nav[tuple(nav_parts)] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print(f"---\ntitle: {identifier}\n---\n\n::: {identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
