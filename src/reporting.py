# src/reporting.py
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def build_report_bytes(title: str, scenario: dict, results: dict) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, title)
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Senaryo Girdileri")
    y -= 18
    c.setFont("Helvetica", 10)
    for k, v in scenario.items():
        c.drawString(50, y, f"- {k}: {v}")
        y -= 14
        if y < 80:
            c.showPage(); y = h - 50

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Sonuçlar")
    y -= 18
    c.setFont("Helvetica", 10)
    for k, v in results.items():
        c.drawString(50, y, f"- {k}: {v}")
        y -= 14
        if y < 80:
            c.showPage(); y = h - 50

    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes
