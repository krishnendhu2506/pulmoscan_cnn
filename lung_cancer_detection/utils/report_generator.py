import os
import textwrap
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics


def _draw_section_title(c, text, x, y, width):
    c.setFillColor(colors.HexColor("#e6f1fb"))
    c.roundRect(x, y - 20, width, 24, 8, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#1f4f7a"))
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x + 12, y - 14, text)


def _fmt(value, fallback="N/A"):
    if value is None:
        return fallback
    if isinstance(value, str) and value.strip() == "":
        return fallback
    return str(value)


def _interpretation_text(predicted_class, confidence):
    strength = "high" if confidence >= 85 else "moderate" if confidence >= 70 else "low"
    return (
        f"The model indicates a {strength}-confidence pattern consistent with {predicted_class}. "
        "This interpretation should be correlated with clinical findings, radiology review, "
        "and any prior imaging before clinical decisions are made."
    )


def _draw_card(c, x, y, w, h, fill="#f6fbff", stroke="#d6e7f7", radius=10):
    c.setFillColor(colors.HexColor(fill))
    c.setStrokeColor(colors.HexColor(stroke))
    c.setLineWidth(1)
    c.roundRect(x, y, w, h, radius, stroke=1, fill=1)


def _safe_draw_image(c, image_path, x, y, w, h):
    try:
        image = ImageReader(image_path)
        c.drawImage(image, x, y, width=w, height=h, preserveAspectRatio=True, mask="auto")
        return True
    except Exception:
        return False


def _wrap_to_width(text, font_name, font_size, max_width):
    words = (text or "").split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_wrapped_text(c, x, y, text, max_width, font_name, font_size, leading):
    c.setFont(font_name, font_size)
    lines = _wrap_to_width(text, font_name, font_size, max_width)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def generate_report(
    report_path,
    patient,
    image_path,
    predicted_class,
    confidence,
    probs,
    prediction_id=None,
    model_name="PulmoScan CNN v1",
    scan_date=None,
    logo_path=None,
):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter

    margin_x = 36
    column_gap = 18
    col_w = (width - (margin_x * 2) - column_gap) / 2

    # Header
    header_logo_size = 44
    header_y = height - 72
    header_text_x = margin_x
    if logo_path and os.path.exists(logo_path):
        if _safe_draw_image(c, logo_path, margin_x, header_y - 6, header_logo_size, header_logo_size):
            header_text_x = margin_x + header_logo_size + 12
    else:
        # Fallback mark if logo is missing.
        c.setFillColor(colors.HexColor("#2f6da8"))
        c.circle(margin_x + 18, height - 52, 18, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(margin_x + 18, height - 56, "PS")
        header_text_x = margin_x + 50

    c.setFillColor(colors.HexColor("#1b3c5a"))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(header_text_x, height - 45, "Lung Cancer")
    c.setFont("Helvetica-Bold", 18)
    c.drawString(header_text_x, height - 68, "Diagnosis Report")
    c.setFont("Helvetica", 11)
    c.setFillColor(colors.HexColor("#1b3c5a"))
    c.drawRightString(width - margin_x, height - 50, _fmt(scan_date, datetime.now().strftime("%B %d, %Y")))
    c.setStrokeColor(colors.HexColor("#d6e7f7"))
    c.setLineWidth(1)
    c.line(margin_x, height - 80, width - margin_x, height - 80)

    report_id = f"RPT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    pred_id = str(prediction_id) if prediction_id is not None else "N/A"

    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica", 9)
    c.drawRightString(width - 40, height - 45, f"Report ID: {report_id}")
    c.drawRightString(width - 40, height - 60, f"Prediction ID: {pred_id}")

    # Patient Information card
    card_y = height - 250
    card_h = 150
    _draw_card(c, margin_x, card_y, width - (margin_x * 2), card_h)
    _draw_section_title(c, "Patient Information", margin_x + 10, card_y + card_h - 10, 180)
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#1b2a41"))
    row_y = card_y + card_h - 50
    left_x = margin_x + 20
    mid_x = margin_x + 180
    right_x = margin_x + 360
    c.drawString(left_x, row_y, "Patient ID")
    c.drawString(mid_x, row_y, _fmt(patient["patient_id"]))
    c.drawString(right_x, row_y, "Scan Date:")
    c.drawString(right_x + 80, row_y, _fmt(scan_date))
    row_y -= 24
    c.drawString(left_x, row_y, "Name")
    c.drawString(mid_x, row_y, _fmt(patient["name"]))
    row_y -= 24
    c.drawString(left_x, row_y, "Age")
    c.drawString(mid_x, row_y, _fmt(patient["age"]))
    row_y -= 24
    c.drawString(left_x, row_y, "Gender")
    c.drawString(mid_x, row_y, _fmt(patient["gender"]))

    # Left column: Clinical Information
    left_col_x = margin_x
    left_col_y = 290
    left_col_h = 200
    _draw_card(c, left_col_x, left_col_y, col_w, left_col_h)
    _draw_section_title(c, "Clinical Information", left_col_x + 10, left_col_y + left_col_h - 10, 180)
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#1b2a41"))
    padding = 14
    info_x = left_col_x + padding
    info_y = left_col_y + left_col_h - 50
    info_max_w = col_w - (padding * 2)
    info_bottom = left_col_y + 16
    clinical_lines = [
        f"Smoking History: {_fmt(patient['smoking_history'])} ({_fmt(patient['years_of_smoking'])} years)",
        f"Family History of Lung Cancer: {_fmt(patient['family_history_lung_cancer'])}",
        f"Air Pollution Exposure: {_fmt(patient['air_pollution_exposure'])}",
        f"Occupational Exposure: {_fmt(patient['occupational_exposure'])}",
        f"Persistent Cough: {_fmt(patient['persistent_cough'])}",
        f"Unexplained Weight Loss: {_fmt(patient['unexplained_weight_loss'])}",
    ]
    for line in clinical_lines:
        if info_y <= info_bottom:
            break
        info_y = _draw_wrapped_text(c, info_x, info_y, line, info_max_w, "Helvetica", 9, 12) - 6

    # Right column: AI Prediction Result
    right_col_x = margin_x + col_w + column_gap
    right_col_y = 290
    right_col_h = 200
    _draw_card(c, right_col_x, right_col_y, col_w, right_col_h)
    _draw_section_title(c, "AI Prediction Result", right_col_x + 10, right_col_y + right_col_h - 10, 190)
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#1b2a41"))
    pad = 14
    pred_x = right_col_x + pad
    pred_y = right_col_y + right_col_h - 50
    pred_max_w = col_w - (pad * 2)
    c.drawString(pred_x, pred_y, f"Prediction: {predicted_class}")
    pred_y -= 18
    c.drawString(pred_x, pred_y, "Probability Distribution:")
    pred_y -= 16
    pred_bottom = right_col_y + 18
    for label, value in probs.items():
        if pred_y <= pred_bottom + 16:
            break
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#1b2a41"))
        label_col_w = 112
        pct_col_w = 30
        gap = 8
        label_right_x = pred_x + label_col_w
        c.drawRightString(label_right_x, pred_y, f"{label}:")
        bar_x = label_right_x + gap
        bar_w = max(40, pred_max_w - label_col_w - gap - pct_col_w)
        bar_h = 10
        c.setFillColor(colors.HexColor("#e2e8f0"))
        c.roundRect(bar_x, pred_y - 4, bar_w, bar_h, 4, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#f2c94c") if label != "Squamous Cell Carcinoma" else colors.HexColor("#f2994a"))
        c.roundRect(bar_x, pred_y - 4, bar_w * (value / 100.0), bar_h, 4, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#1b2a41"))
        c.drawRightString(right_col_x + col_w - pad, pred_y, f"{value:.0f}%")
        pred_y -= 18
    pred_y -= 2
    if pred_y > pred_bottom + 30:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(pred_x, pred_y, "AI Interpretation")
        c.setFont("Helvetica", 9)
        interpretation = _interpretation_text(predicted_class, confidence)
        line_y = pred_y - 16
        c.setFillColor(colors.HexColor("#1b2a41"))
        _draw_wrapped_text(c, pred_x, line_y, interpretation, pred_max_w, "Helvetica", 8.5, 11)

    # Left column: CT Scan Image
    img_card_y = 70
    img_card_h = 200
    _draw_card(c, left_col_x, img_card_y, col_w, img_card_h)
    _draw_section_title(c, "CT Scan Image", left_col_x + 10, img_card_y + img_card_h - 10, 140)
    try:
        image = ImageReader(image_path)
        c.drawImage(
            image,
            left_col_x + 14,
            img_card_y + 20,
            width=col_w - 28,
            height=img_card_h - 60,
            preserveAspectRatio=True,
            mask="auto",
        )
    except Exception:
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor("#1b2a41"))
        c.drawString(left_col_x + 14, img_card_y + img_card_h / 2, "Unable to render CT scan image.")

    # Disclaimer
    disclaimer_y = 70
    disclaimer_h = 200
    _draw_card(c, right_col_x, disclaimer_y, col_w, disclaimer_h, fill="#fff6e6", stroke="#f0d7a6")
    _draw_section_title(c, "Disclaimer", right_col_x + 10, disclaimer_y + disclaimer_h - 10, 120)
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#5a4a2a"))
    disclaimer_text = (
        "This report is generated by an AI-based diagnostic support system. "
        "It is intended to assist healthcare professionals and should not be "
        "used as the sole basis for a diagnosis. Always consult a qualified "
        "medical professional for an accurate assessment."
    )
    line_y = disclaimer_y + disclaimer_h - 40
    _draw_wrapped_text(
        c,
        right_col_x + 14,
        line_y,
        disclaimer_text,
        col_w - 28,
        "Helvetica",
        9,
        12,
    )

    c.showPage()
    c.save()
