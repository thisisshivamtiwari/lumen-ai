"""Generate explanatory README SVGs; no dataset or model outputs are used."""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent / 'assets'


def generate(dark=False):
    bg, fg, muted, panel, line = ('#101323', '#f1f5ff', '#a9b4d0', '#191e35', '#363e60') if dark else ('#f5f6ff', '#18213c', '#596580', '#ffffff', '#d8dded')
    theme = 'dark' if dark else 'light'
    def svg(body, height, title):
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title"><title id="title">{escape(title)}</title><rect width="1200" height="{height}" rx="24" fill="{bg}"/>{body}</svg>\n'
    def text(x, y, value, size=18, color=None, weight=400):
        return f'<text x="{x}" y="{y}" fill="{color or fg}" font-family="system-ui, -apple-system, sans-serif" font-size="{size}" font-weight="{weight}">{escape(value)}</text>'
    body = text(56, 61, 'DECODE & DETECT  /  XRPL', 14, muted, 600)
    body += text(56, 139, 'Lumen.ai', 62, fg, 750)
    body += text(58, 187, 'Following the flow. Ranking the risk.', 24, muted)
    body += text(58, 239, 'Behavioural features  ·  Graph context  ·  Transaction ranking', 16, muted)
    pts = [(790, 80), (925, 42), (1060, 80), (790, 230), (925, 275), (1060, 230), (925, 158)]
    for a, b in [(0, 1), (1, 2), (0, 3), (2, 5), (3, 4), (4, 5), (0, 6), (6, 5), (1, 6), (6, 4)]:
        x, y = pts[a]; xx, yy = pts[b]
        accent = (a, b) in [(0, 6), (6, 5)]
        body += f'<path d="M{x} {y} L{xx} {yy}" stroke="{"#8b7bf6" if accent else line}" stroke-width="{4 if accent else 2}" fill="none"/>'
    for i, (x, y) in enumerate(pts):
        body += f'<circle cx="{x}" cy="{y}" r="{18 if i == 6 else 12}" fill="{"#8b7bf6" if i == 6 else panel}" stroke="{"#a89cff" if i == 6 else line}" stroke-width="3"/>'
    (OUT / f'lumen-hero-{theme}.svg').write_text(svg(body, 320, 'Lumen.ai: behavioural and graph context for synthetic XRPL payment ranking'))
    cards = [('01', 'Organiser tables', 'Transactions + transfers'), ('02', 'Economic payments', 'One edge per transaction'), ('03', 'Behaviour + graph', 'Wallets, bursts and paths'), ('04', 'Models + context', 'CatBoost and neighbours'), ('05', 'Ranked scores', 'One score per payment')]
    body = text(32, 39, 'FROM PAYMENT ACTIVITY TO A REVIEW QUEUE', 14, muted, 600)
    for i, (n, title, detail) in enumerate(cards):
        x = 28 + 235 * i
        body += f'<rect x="{x}" y="65" width="204" height="131" rx="14" fill="{panel}" stroke="{line}"/>'
        body += text(x + 16, 94, n, 15, '#8b7bf6', 700)
        body += text(x + 16, 132, title, 16, fg, 600)
        body += text(x + 16, 163, detail, 12, muted)
        if i < 4:
            body += f'<path d="M{x + 211} 130 h16 m-5 -5 5 5 -5 5" fill="none" stroke="{muted}" stroke-width="2"/>'
    body += text(32, 231, 'Illustrative architecture. No labels, transaction values or performance measurements are drawn here.', 13, muted)
    (OUT / f'lumen-pipeline-{theme}.svg').write_text(svg(body, 260, 'Lumen pipeline: tables, payments, features, models, ranked scores'))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    generate(False)
    generate(True)
