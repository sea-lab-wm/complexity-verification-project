import re
from pathlib import Path
from collections import defaultdict


### FOR JaTyC
def group_warnings_by_snippet(warnings_file, project_root, output_file):
    """
    Reads a warnings file and groups warnings by dataset, filename, and snippet ID.
    Writes results in the format:

    dataset - <dataset>, FileName - <filename> snippet# - <method_id>
    <warning 1>
    <warning 2>
    ...
    """
    
    warnings_file = Path(warnings_file)
    project_root = Path(project_root)
    output_file = Path(output_file)

    # Regex to extract Java file path and line number
    warning_re = re.compile(r"(dataset(\d+)/src/main/java/.+?\.java):(\d+):")
    
    # Regex to detect method/snippet start
    method_re = re.compile(r"/\*+\s*Method\s+(\d+)\s*\*+/")

    # Read warnings
    warnings_list = []
    with open(warnings_file, "r") as f:
        warning_lines = f.readlines()

    # Group warnings with continuation lines
    idx = 0
    while idx < len(warning_lines):
        line = warning_lines[idx].rstrip()
        m = warning_re.search(line)
        if m:
            java_path = Path(m.group(1))
            dataset = m.group(2)
            line_no = int(m.group(3))
            # collect continuation lines (the indented lines below the warning)
            continuation = []
            idx += 1
            while idx < len(warning_lines) and warning_lines[idx].startswith(' '):
                continuation.append(warning_lines[idx].rstrip())
                idx += 1
            full_warning = '\n'.join([line] + continuation)
            warnings_list.append({
                "java_path": java_path,
                "dataset": dataset,
                "line_no": line_no,
                "warning_text": full_warning
            })
        else:
            idx += 1

    # Group warnings by (dataset, filename, snippet)
    grouped_warnings = defaultdict(list)

    for w in warnings_list:
        java_file = project_root / w["java_path"]
        filename = w["java_path"].name

        if not java_file.exists():
            snippet_id = "Unknown"
        else:
            with open(java_file, "r") as f:
                lines = f.readlines()

            # Build method/snippet ranges
            method_ranges = []
            current_method = None
            start_line = None
            for idx_line, line_content in enumerate(lines, start=1):
                m = method_re.search(line_content)
                if m:
                    if current_method is not None:
                        method_ranges.append((current_method, start_line, idx_line-1))
                    current_method = m.group(1)
                    start_line = idx_line
            if current_method is not None:
                method_ranges.append((current_method, start_line, len(lines)))

            # Find snippet containing the warning
            snippet_id = "Unknown"
            for method_no, start, end in method_ranges:
                if start <= w["line_no"] <= end:
                    snippet_id = method_no
                    break

        key = f"dataset - {w['dataset']}, FileName - {filename} snippet# - {snippet_id}"
        grouped_warnings[key].append(w["warning_text"])

    # Write grouped output
    with open(output_file, "w") as f:
        for key, warnings_in_snippet in grouped_warnings.items():
            f.write(key + "\n")
            for warning_text in warnings_in_snippet:
                f.write(warning_text + "\n\n")  # extra newline between warnings
            f.write("\n")  # extra newline after each snippet group

    print(f"Done. Output written to {output_file}")

# -----------------------------
# Example usage
# -----------------------------
# group_warnings_by_snippet("/home/kgdesilva/Desktop/TOSEM/complexity-verification-project/data/JaTyC_8.txt", ".", "/home/kgdesilva/Desktop/TOSEM/complexity-verification-project/dataset8/src/main/output.txt")

### FOR OpenJML
def group_openjml_warnings_by_snippet(
    warnings_file,
    java_source_root,
    dataset_id,
    output_file
):
    """
    Supports OpenJML warning format like:
    ./ATunes.java:32: verify: ...
    """

    warnings_file = Path(warnings_file)
    java_source_root = Path(java_source_root)
    output_file = Path(output_file)

    # OpenJML warning regex
    openjml_re = re.compile(r"\./(.+?\.java):(\d+):")

    # Method/snippet marker
    method_re = re.compile(r"/\*+\s*Method\s+(\d+)\s*\*+/")

    # -------- Read warnings (with continuation lines) --------
    warnings = []
    with open(warnings_file, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        m = openjml_re.search(line)
        if m:
            filename = m.group(1)
            line_no = int(m.group(2))

            continuation = []
            i += 1
            while i < len(lines) and not lines[i].startswith("./"):
                continuation.append(lines[i].rstrip())
                i += 1

            full_warning = "\n".join([line] + continuation)

            warnings.append({
                "dataset": dataset_id,
                "filename": filename,
                "line_no": line_no,
                "warning_text": full_warning
            })
        else:
            i += 1

    # -------- Group by snippet --------
    grouped = defaultdict(list)

    for w in warnings:
        java_file = java_source_root / w["filename"]

        snippet_id = "Unknown"
        if java_file.exists():
            with open(java_file, "r") as f:
                src_lines = f.readlines()

            method_ranges = []
            current = None
            start = None

            for idx, src_line in enumerate(src_lines, start=1):
                m = method_re.search(src_line)
                if m:
                    if current is not None:
                        method_ranges.append((current, start, idx - 1))
                    current = m.group(1)
                    start = idx

            if current is not None:
                method_ranges.append((current, start, len(src_lines)))

            for mid, s, e in method_ranges:
                if s <= w["line_no"] <= e:
                    snippet_id = mid
                    break

        key = f"dataset - {w['dataset']}, FileName - {w['filename']} snippet# - {snippet_id}"
        grouped[key].append(w["warning_text"])

    # -------- Write output --------
    with open(output_file, "w") as f:
        for header, warnings in grouped.items():
            f.write(header + "\n")
            for w in warnings:
                f.write(w + "\n\n")
            f.write("\n")

    print(f"Done. Output written to {output_file}")


import re
from pathlib import Path
from collections import defaultdict


def group_checker_framework_warnings_by_snippet(
    warnings_file,
    java_source_root,
    dataset_id,
    output_file
):
    """
    Reads Checker Framework warning files and groups warnings by snippet/method.
    Each warning may span multiple lines:
    
    /path/to/File.java:<line>: warning: ...
        <code line>
        ^
      additional info...
    """

    warnings_file = Path(warnings_file)
    java_source_root = Path(java_source_root)
    output_file = Path(output_file)

    # Regex to detect the first line of a warning
    checker_re = re.compile(r".+/(.+\.java):(\d+):\s+warning:")

    # Method/snippet marker
    method_re = re.compile(r"/\*+\s*Method\s+(\d+)\s*\*+/")

    # -------- Read warnings with continuation lines --------
    warnings = []
    with open(warnings_file, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        m = checker_re.search(line)
        if m:
            filename = m.group(1)
            line_no = int(m.group(2))

            continuation = []
            i += 1
            # Collect all indented lines below the first warning line
            while i < len(lines) and (lines[i].startswith(' ') or lines[i].startswith('\t')):
                continuation.append(lines[i].rstrip())
                i += 1

            full_warning = "\n".join([line] + continuation)

            warnings.append({
                "dataset": dataset_id,
                "filename": filename,
                "line_no": line_no,
                "warning_text": full_warning
            })
        else:
            i += 1

    # -------- Group warnings by snippet --------
    grouped = defaultdict(list)

    for w in warnings:
        java_file = java_source_root / w["filename"]

        snippet_id = "Unknown"
        if java_file.exists():
            with open(java_file, "r") as f:
                src_lines = f.readlines()

            method_ranges = []
            current = None
            start = None

            for idx, src_line in enumerate(src_lines, start=1):
                mm = method_re.search(src_line)
                if mm:
                    if current is not None:
                        method_ranges.append((current, start, idx - 1))
                    current = mm.group(1)
                    start = idx

            if current is not None:
                method_ranges.append((current, start, len(src_lines)))

            for mid, s, e in method_ranges:
                if s <= w["line_no"] <= e:
                    snippet_id = mid
                    break

        key = f"dataset - {w['dataset']}, FileName - {w['filename']} snippet# - {snippet_id}"
        grouped[key].append(w["warning_text"])

    # -------- Write output --------
    with open(output_file, "w") as f:
        for header, warnings_in_snippet in grouped.items():
            f.write(header + "\n")
            for w in warnings_in_snippet:
                f.write(w + "\n\n")
            f.write("\n")

    print(f"Done. Output written to {output_file}")



group_checker_framework_warnings_by_snippet(
    warnings_file="/home/kgdesilva/Desktop/TOSEM/complexity-verification-project/data/ch-63.txt",
    java_source_root="/home/kgdesilva/Desktop/TOSEM/complexity-verification-project/dataset63/src/main/java",
    dataset_id="63",
    output_file="/home/kgdesilva/Desktop/TOSEM/complexity-verification-project/dataset63/src/main/output-och.txt"
)