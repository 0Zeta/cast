"""Compile Typst panels and rasterize or combine the resulting PDF pages."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tomllib


def style(figures: Path) -> dict:
    tokens = tomllib.loads((figures / 'shared/styles/tokens.toml').read_text())
    return {'style': {**tokens['qualitative'], 'font': tokens['fonts']['qualitative']},
            'colors': tokens['colors'], 'fonts': tokens['fonts']}


def render_panel(figures: Path, panel: dict, assets: Path, destination: Path,
                 *, proposals=1, scores=None, title='', decoded=False) -> tuple[Path, list[dict]]:
    return render_panels(figures, [{
        'data': panel, 'assets': assets, 'media': panel['media'],
        'scores': panel['selected_tracks'] if scores is None else scores,
        'proposals': proposals, 'title': title, 'decoded': decoded,
    }], destination, build=assets.parent)


def render_panels(figures: Path, entries: list[dict], destination: Path,
                  *, build: Path, style_overrides: dict | None = None) -> tuple[Path, list[dict]]:
    """Compile one or several accepted-layout panels on one standalone page."""
    import typst
    build.mkdir(parents=True, exist_ok=True)
    source = build / 'figure.typ'
    shutil.copyfile(figures / 'fig4-qualitative-examples/figure.typ', source)
    panels = []
    for entry in entries:
        assets = Path(entry['assets'])
        scores = entry['scores']
        absent = set(scores) - set(entry['data']['tracks'])
        if absent:
            raise ValueError(f"{entry['data']['case_name']}: absent tracks: {sorted(absent)}")
        panels.append({key: value for key, value in entry.items() if key != 'assets'} | {
            'asset_root': assets.relative_to(build).as_posix(),
            'flux_markers': entry.get('flux_markers', False)})
    spec = {**style(figures), 'layout': 'column', 'panels': panels}
    spec['style'].update(style_overrides or {})
    spec_path = build / f'{destination.stem}.json'
    spec_path.write_text(json.dumps(spec, allow_nan=False))
    compiler = typst.Compiler(str(source), root=str(build),
                              font_paths=[str(figures / 'shared/fonts')],
                              sys_inputs={'spec': spec_path.name})
    destination.parent.mkdir(parents=True, exist_ok=True)
    compiler.compile(output=str(destination), timestamp=0)
    geometry = json.loads(compiler.query('metadata', field='value'))
    for entry in entries:
        own_geometry = [row for row in geometry if row['case'] == entry['data']['case_name']]
        (Path(entry['assets']) / 'geometry.json').write_text(json.dumps(own_geometry, indent=2) + '\n')
    return destination, geometry


def render_source(source: Path, destination: Path, *, root: Path, inputs=None) -> Path:
    import typst
    destination.parent.mkdir(parents=True, exist_ok=True)
    typst.compile(str(source), output=str(destination), root=str(root), timestamp=0,
                  font_paths=[str(Path(__file__).resolve().parent / 'fonts')],
                  sys_inputs=inputs or {})
    return destination


def rasterize(pdf: Path, png: Path, scale=2.0):
    import pypdfium2 as pdfium
    with pdfium.PdfDocument(pdf) as document:
        if len(document) != 1:
            raise ValueError(f'expected one panel page: {pdf}')
        page = document[0]
        bitmap = page.render(scale=scale)
        result = bitmap.to_pil().convert('RGB')
        result.save(png)
        bitmap.close()
        page.close()
    return result


def gallery(pdfs: list[Path], destination: Path) -> None:
    import pypdfium2 as pdfium
    with pdfium.PdfDocument.new() as result:
        for pdf in pdfs:
            with pdfium.PdfDocument(pdf) as source:
                result.import_pages(source)
        result.save(destination)
