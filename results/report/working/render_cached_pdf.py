"""Use the packaged DOCX rasterizer with an already exported Word PDF.

The bundled LibreOffice conversion is unavailable on this host. This adapter
only replaces conversion; pdf2image, naming and rasterization remain canonical.
"""
from pathlib import Path
import argparse
import importlib.util
import os
import json
from PIL import Image, ImageDraw

SKILL = Path(r'C:/Users/paulb/.codex/plugins/cache/openai-primary-runtime/documents/26.1005.12141/skills/documents')
POPPLER = Path(r'C:/Users/paulb/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')

parser = argparse.ArgumentParser()
parser.add_argument('docx')
parser.add_argument('pdf')
parser.add_argument('out')
parser.add_argument('--dpi', type=int, default=120)
parser.add_argument('--sheet-size', type=int, default=6)
args = parser.parse_args()
os.environ['PATH'] = str(POPPLER) + os.pathsep + os.environ['PATH']
spec = importlib.util.spec_from_file_location('render_docx', SKILL / 'render_docx.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
pdf = str(Path(args.pdf).resolve())
renderer.convert_to_pdf = lambda *a, **k: (pdf, 'Word exported PDF supplied by task-local adapter')
pages = renderer.rasterize(args.docx, args.out, args.dpi, verbose=False, emit_pdf=False)
out = Path(args.out)
for begin in range(0, len(pages), args.sheet_size):
    batch = pages[begin:begin + args.sheet_size]
    cols = 2 if args.sheet_size <= 6 else 4
    tilew = 550 if cols == 2 else 360
    tileh = int(tilew * 1.414) + 25
    rows = (len(batch) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tilew, rows * tileh), '#cccccc')
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(batch):
        im = Image.open(path).convert('RGB')
        im.thumbnail((tilew - 6, tileh - 30))
        x, y = (i % cols) * tilew, (i // cols) * tileh
        sheet.paste(im, (x + 3, y + 25))
        draw.text((x + 10, y + 5), f'Page {begin+i+1}', fill='black')
    sheet.save(out / f'contact-{begin+1}-{begin+len(batch)}.png')
(out / 'render_manifest.json').write_text(json.dumps({'pages': len(pages), 'dpi': args.dpi, 'converter': 'Word COM', 'rasterizer': str(SKILL / 'render_docx.py'), 'images': pages}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'pages': len(pages), 'out': str(out)}, ensure_ascii=False))
