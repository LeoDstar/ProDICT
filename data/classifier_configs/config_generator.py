from pathlib import Path
import re

BASE = Path("BRCA_model_settings.yaml")          # your blueprint file
OUTDIR = Path("configs_out")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Each entry is a list -> becomes TARGET_CLASS : ["...","..."]
TARGETS = [
    ["ACC"],
    ["ACYC"],
    ["ANGS"],
    ["ARMS"],
    ["ASPS"],
    ["BA", "ANGS"],
    ["BA"],
    ["BRCA"],
    ["CHDM"],
    ["CHS"],
    ["COAD", "READ", "COADREAD"],
    ["COAD"],
    ["DDLS"],
    ["DFSP"],
    ["DIFG"],
    ["DSRCT"],
    ["EHAE"],
    ["EPIS"],
    ["ERMS"],
    ["ES"],
    ["GINET"],
    ["GIST"],
    ["IHCH"],
    ["LGFMS"],
    ["LMS", "ULMS"],
    ["LMS"],
    ["LUAD"],
    ["MEL", "UM"],
    ["MEL"],
    ["MFH"],
    ["MPNST"],
    ["MRLS"],
    ["MYEC"],
    ["OS"],
    ["PAAD"],
    ["PANET"],
    ["PLEMESO", "PEMESO"],
    ["SCRMS"],
    ["SDCA", "MYEC"],
    ["SDCA"],
    ["SFT"],
    ["SYNS"],
    ["THYC"],
    ["THYM"],
    ["ULMS"],
    ["UM"],
    ["RMS", "ARMS", "ERMS", "SCRMS"],
]

text = BASE.read_text(encoding="utf-8")

def format_inline_yaml_list(values: list[str]) -> str:
    # Produces ["BA", "ANGS"]
    inside = ", ".join([f'"{v}"' for v in values])
    return f"[{inside}]"

def replace_target_class_line(yaml_text: str, values: list[str]) -> str:
    new_list = format_inline_yaml_list(values)

    # Match:
    # TARGET_CLASS : ["BRCA"]  # comment
    # capturing prefix (including spaces around colon) + list + optional trailing comment
    pattern = r'(?m)^(TARGET_CLASS\s*:\s*)\[[^\]]*\](\s*(#.*)?)$'

    def repl(m: re.Match) -> str:
        prefix = m.group(1)         # e.g. "TARGET_CLASS : "
        suffix = m.group(2) or ""   # e.g. "  # Change this..."
        return f"{prefix}{new_list}{suffix}"

    new_text, n = re.subn(pattern, repl, yaml_text, count=1)
    if n == 0:
        raise ValueError("Could not find a TARGET_CLASS line of the form: TARGET_CLASS : [ ... ]")
    return new_text

def file_stem(values: list[str]) -> str:
    return "_".join(values)

for values in TARGETS:
    out_text = replace_target_class_line(text, values)
    out_path = OUTDIR / f"entity_model_config_{file_stem(values)}.yaml"
    out_path.write_text(out_text, encoding="utf-8")
    print("Wrote:", out_path)