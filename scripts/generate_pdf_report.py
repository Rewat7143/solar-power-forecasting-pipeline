"""
Config-driven styled PDF report generator.

Usage:
    python scripts/generate_pdf_report.py
    python scripts/generate_pdf_report.py --config report_profiles.json
    python scripts/generate_pdf_report.py --profile university
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "report_profiles.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_text(path: Path) -> str:
    if not path.exists():
        return f"File not found: {path.as_posix()}"
    return path.read_text(encoding="utf-8", errors="ignore")


def human_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def list_workspace_files(root: Path, ignored: set[str]) -> List[Tuple[str, int]]:
    records: List[Tuple[str, int]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignored]
        for name in filenames:
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            try:
                records.append((rel, path.stat().st_size))
            except OSError:
                continue
    return sorted(records)


def top_level_summary(root: Path, ignored: set[str]) -> List[Tuple[str, str, str]]:
    rows = []
    for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if p.name in ignored:
            continue
        if p.is_dir():
            file_count = 0
            for _, dirnames, filenames in os.walk(p):
                dirnames[:] = [d for d in dirnames if d not in ignored]
                file_count += len(filenames)
            rows.append((p.name + "/", "directory", str(file_count)))
        else:
            rows.append((p.name, "file", human_size(p.stat().st_size)))
    return rows


def fit_image(path: Path, max_width: float, max_height: float) -> Image:
    img = Image(str(path))
    iw, ih = img.imageWidth, img.imageHeight
    if iw == 0 or ih == 0:
        return img
    scale = min(max_width / iw, max_height / ih)
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    return img


def build_table(data: List[List[str]], header_color: str, zebra_color: str, col_widths: Iterable[float] | None = None) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(zebra_color)]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d0d0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def make_page_decorator(footer_left: str, footer_right: str):
    def draw_footer(canv: canvas.Canvas, doc):
        canv.saveState()
        canv.setFont("Helvetica", 8)
        canv.setFillColor(colors.HexColor("#666666"))
        canv.drawString(doc.leftMargin, 0.75 * cm, footer_left)
        canv.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.75 * cm, f"{footer_right} | Page {canv.getPageNumber()}")
        canv.restoreState()

    return draw_footer


def make_styles(style_cfg: dict) -> dict:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=int(style_cfg.get("title_size", 20)),
            textColor=colors.HexColor(style_cfg.get("primary_color", "#1f3c88")),
            leading=int(style_cfg.get("title_size", 20)) + 4,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=int(style_cfg.get("h1_size", 14)),
            textColor=colors.HexColor(style_cfg.get("primary_color", "#1f3c88")),
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=int(style_cfg.get("h2_size", 11)),
            textColor=colors.HexColor(style_cfg.get("secondary_color", "#1f3c88")),
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=int(style_cfg.get("body_size", 9)),
            leading=int(style_cfg.get("body_size", 9)) + 3,
        ),
        "mono": ParagraphStyle(
            "mono",
            parent=base["Code"],
            fontName="Courier",
            fontSize=int(style_cfg.get("mono_size", 7)),
            leading=int(style_cfg.get("mono_size", 7)) + 2,
        ),
    }
    return styles


def add_paragraph(story: list, text: str, style, space_cm: float = 0.2):
    story.append(Paragraph(text, style))
    story.append(Spacer(1, space_cm * cm))


def build_story(root: Path, config: dict, style_cfg: dict, metadata: dict) -> list:
    styles = make_styles(style_cfg)
    metrics = load_json(root / config["paths"]["metrics_json"])
    manifest = load_json(root / config["paths"]["manifest_json"])
    image_dir = root / config["paths"]["image_dir"]
    text_files = config.get("appendix_files", [])
    ignored_dirs = set(config.get("ignored_dirs", []))

    workspace_files = list_workspace_files(root, ignored_dirs)
    top_summary = top_level_summary(root, ignored_dirs)

    story: list = []
    generated_at = datetime.now().strftime(metadata.get("date_format", "%Y-%m-%d %H:%M:%S"))

    story.append(Paragraph(metadata.get("title", "Solar Forecasting Report"), styles["title"]))
    add_paragraph(story, metadata.get("subtitle", "Comprehensive project report"), styles["body"])
    add_paragraph(story, f"Generated on: {generated_at}", styles["body"])
    if metadata.get("author"):
        add_paragraph(story, f"Prepared by: {metadata['author']}", styles["body"])
    if metadata.get("institution"):
        add_paragraph(story, f"Institution: {metadata['institution']}", styles["body"])

    add_paragraph(story, "1. Workspace Snapshot", styles["h1"])
    add_paragraph(story, f"Workspace root: {root.as_posix()}", styles["body"])
    add_paragraph(story, f"Files included in inventory: {len(workspace_files)}", styles["body"])
    summary_data = [["Entry", "Type", "Size / Count"]] + [[a, b, c] for a, b, c in top_summary]
    story.append(build_table(summary_data, style_cfg["table_header_color"], style_cfg["table_zebra_color"], [7.2 * cm, 3 * cm, 5.9 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    add_paragraph(story, "2. Model Registry", styles["h1"])
    saved_models = manifest.get("saved_models", {})
    if saved_models:
        model_rows = [["Model", "Family", "Type", "Calibrator", "Path"]]
        for name, info in saved_models.items():
            model_rows.append([
                str(name),
                str(info.get("family", "")),
                str(info.get("model_type", "")),
                str(info.get("has_calibrator", "")),
                str(info.get("saved_path", "")),
            ])
        story.append(build_table(model_rows, style_cfg["table_header_color"], style_cfg["table_zebra_color"], [4.5 * cm, 2 * cm, 4.3 * cm, 2 * cm, 3.3 * cm]))
        story.append(Spacer(1, 0.3 * cm))

    add_paragraph(story, "3. Performance Metrics", styles["h1"])
    add_paragraph(story, f"Run date: {metrics.get('run_date', 'N/A')} | Source: {metrics.get('source', 'N/A')}", styles["body"])
    for key, title in [("one_step", "One-step Metrics"), ("recursive_daylight", "Recursive Daylight Metrics")]:
        rows = metrics.get(key, [])
        if rows:
            add_paragraph(story, title, styles["h2"])
            table_rows = [["Model", "MAE", "RMSE", "R2"]]
            for row in rows:
                table_rows.append([
                    str(row.get("Model", "")),
                    f"{float(row.get('MAE', 0.0)):.6f}",
                    f"{float(row.get('RMSE', 0.0)):.6f}",
                    f"{float(row.get('R2', 0.0)):.6f}",
                ])
            story.append(build_table(table_rows, style_cfg["table_header_color"], style_cfg["table_zebra_color"], [6.7 * cm, 3 * cm, 3 * cm, 3 * cm]))
            story.append(Spacer(1, 0.2 * cm))

    sh = metrics.get("stacked_holdout_daylight", {})
    if sh:
        add_paragraph(story, "Stacked Holdout Daylight", styles["h2"])
        hold = [["MAE", "RMSE", "R2"], [f"{float(sh.get('MAE', 0.0)):.6f}", f"{float(sh.get('RMSE', 0.0)):.6f}", f"{float(sh.get('R2', 0.0)):.6f}"]]
        story.append(build_table(hold, style_cfg["table_header_color"], style_cfg["table_zebra_color"], [5 * cm, 5 * cm, 5 * cm]))

    story.append(PageBreak())
    add_paragraph(story, "4. Visual Artifacts", styles["h1"])
    image_files = sorted(image_dir.glob("*.png")) if image_dir.exists() else []
    add_paragraph(story, f"Found {len(image_files)} PNG artifacts in {image_dir.as_posix()}.", styles["body"])
    for image_path in image_files:
        add_paragraph(story, image_path.name, styles["h2"])
        story.append(fit_image(image_path, max_width=17.7 * cm, max_height=11 * cm))
        story.append(Spacer(1, 0.35 * cm))

    story.append(PageBreak())
    add_paragraph(story, "5. Documentation Appendix", styles["h1"])
    for rel in text_files:
        story.append(PageBreak())
        add_paragraph(story, rel, styles["h2"])
        path = root / rel
        text = safe_read_text(path)
        if rel.endswith(".json"):
            parsed = load_json(path)
            if parsed:
                text = json.dumps(parsed, indent=2)
        story.append(Preformatted(text, styles["mono"]))

    story.append(PageBreak())
    add_paragraph(story, "6. Full Workspace Inventory", styles["h1"])
    add_paragraph(story, "Complete file list with sizes.", styles["body"])
    inventory = [["Path", "Size"]] + [[rel, human_size(size)] for rel, size in workspace_files]
    chunk_size = int(config.get("inventory_chunk_size", 70))
    for i in range(0, len(inventory), chunk_size):
        story.append(build_table(inventory[i : i + chunk_size], style_cfg["table_header_color"], style_cfg["table_zebra_color"], [13.5 * cm, 2.8 * cm]))
        story.append(Spacer(1, 0.2 * cm))

    return story


def generate_from_profile(root: Path, config: dict, profile: dict) -> Path:
    style_name = profile["style"]
    style_cfg = config["styles"][style_name]
    metadata = dict(config.get("metadata", {}))
    metadata.update(profile.get("metadata_overrides", {}))
    output_path = root / profile["output"]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=float(style_cfg.get("margin_right_cm", 1.2)) * cm,
        leftMargin=float(style_cfg.get("margin_left_cm", 1.2)) * cm,
        topMargin=float(style_cfg.get("margin_top_cm", 1.2)) * cm,
        bottomMargin=float(style_cfg.get("margin_bottom_cm", 1.2)) * cm,
        title=metadata.get("title", "Solar Forecasting Report"),
        author=metadata.get("author", "Auto-generated"),
    )

    story = build_story(root, config, style_cfg, metadata)
    footer = make_page_decorator(metadata.get("footer_left", "Solar Forecasting"), metadata.get("footer_right", style_name.upper()))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate styled PDF reports from project artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to JSON config file.")
    parser.add_argument("--profile", default="all", help="Profile name from config profiles list, or 'all'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_json(config_path)
    profiles = config.get("profiles", [])
    if not profiles:
        raise ValueError("No profiles configured. Add at least one profile in report_profiles.json")

    selected_profiles = profiles
    if args.profile != "all":
        selected_profiles = [p for p in profiles if p.get("name") == args.profile]
        if not selected_profiles:
            raise ValueError(f"Profile not found: {args.profile}")

    generated: List[Path] = []
    for profile in selected_profiles:
        out = generate_from_profile(ROOT, config, profile)
        generated.append(out)

    for path in generated:
        print(f"PDF report generated: {path}")


if __name__ == "__main__":
    main()
