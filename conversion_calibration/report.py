"""Self-contained HTML report for the fixed calibration experiment."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


COLORS={"train":"#4f8cff","synthetic_validation":"#ffb34f","development":"#d86fe8","final":"#50d890"}


def read_csv(path:Path)->list[dict[str,str]]:
    with path.open(newline="") as handle:return list(csv.DictReader(handle))


def scatter(rows:list[dict[str,str]],width:int=760,height:int=430)->str:
    margin=58; values=[float(r["measured_ms"]) for r in rows]+[float(r["predicted_ms"]) for r in rows]; maximum=max(values)*1.07; minimum=min(values)*0.93
    def x(v:float)->float:return margin+(v-minimum)/(maximum-minimum)*(width-2*margin)
    def y(v:float)->float:return height-margin-(v-minimum)/(maximum-minimum)*(height-2*margin)
    points=[]
    for row in rows:
        points.append(f'<circle cx="{x(float(row["measured_ms"])):.2f}" cy="{y(float(row["predicted_ms"])):.2f}" r="4" fill="{COLORS[row["split"]]}" opacity=".82"><title>{html.escape(row["case_id"])}: measured {float(row["measured_ms"]):.6f} ms, predicted {float(row["predicted_ms"]):.6f} ms</title></circle>')
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="Measured versus predicted time">
<line x1="{x(minimum):.2f}" y1="{y(minimum):.2f}" x2="{x(maximum):.2f}" y2="{y(maximum):.2f}" stroke="#8da1bc" stroke-dasharray="6 5"/>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#5e718c"/><line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#5e718c"/>
{''.join(points)}<text x="{width/2}" y="{height-12}" text-anchor="middle">Measured kernel time (ms)</text><text transform="translate(16 {height/2}) rotate(-90)" text-anchor="middle">Predicted kernel time (ms)</text></svg>'''


def residuals(rows:list[dict[str,str]],width:int=760,height:int=430)->str:
    selected=[r for r in rows if r["split"] in {"synthetic_validation","development","final"}]; max_abs=max(abs(float(r["predicted_ms"])-float(r["measured_ms"])) for r in selected) or 1
    margin=58; slot=(width-2*margin)/len(selected); center=height/2
    bars=[]
    for i,row in enumerate(selected):
        value=float(row["predicted_ms"])-float(row["measured_ms"]); h=value/max_abs*(height/2-margin); top=center-h if h>=0 else center; bars.append(f'<rect x="{margin+i*slot+1:.2f}" y="{top:.2f}" width="{max(slot-2,1):.2f}" height="{abs(h):.2f}" fill="{COLORS[row["split"]]}"><title>{html.escape(row["case_id"])}: {value:+.6f} ms</title></rect>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Held-out residuals"><line x1="{margin}" y1="{center}" x2="{width-margin}" y2="{center}" stroke="#8da1bc"/>{"".join(bars)}<text transform="translate(16 {height/2}) rotate(-90)" text-anchor="middle">Predicted − measured (ms)</text></svg>'


def build(run_dir:Path,output:Path)->None:
    analysis=json.loads((run_dir/"analysis"/"analysis.json").read_text()); model=json.loads((run_dir/"analysis"/"model.json").read_text()); rows=read_csv(run_dir/"analysis"/"predictions.csv"); features={r["case_id"]:r for r in read_csv(run_dir/"features.csv")}
    acceptance=analysis["acceptance"]; qc=analysis["quality_control"]
    final=[r for r in rows if r["split"]=="final"]
    table_rows="".join(f'<tr><td><code>{html.escape(r["case_id"])}</code></td><td>{float(r["measured_ms"]):.6f}</td><td>{float(r["predicted_ms"]):.6f}</td><td>{float(r["ape_percent"]):.2f}%</td><td>{html.escape(r["bottleneck"])}</td><td>{int(float(features[r["case_id"]]["registers_per_thread"]))}</td><td>{float(features[r["case_id"]]["estimated_occupancy"])*100:.0f}%</td><td>{int(float(features[r["case_id"]]["critical_dependency_depth"]))}</td></tr>' for r in final)
    params="".join(f'<tr><td>{html.escape(name)}</td><td>{value:.7g}</td></tr>' for name,value in model["parameters_ms"].items())
    assumptions="".join(f'<li><code>{html.escape(c["case_id"])}</code>: {html.escape(", ".join(c["assumptions"]) or "none")}</li>' for c in analysis["manifest"]["cases"] if c["split"]=="final")
    status="PASS" if acceptance["all_pass"] and qc["timing_pass"] else "DOES NOT PASS"
    doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H200 conversion-cost calibration</title><style>
:root{{--bg:#0b111b;--panel:#121c2b;--text:#e9f0f8;--muted:#9bb0c8;--line:#2a3d55;--accent:#6ba3ff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:42px 24px 80px}}h1{{font-size:38px;margin:0 0 8px}}h2{{margin-top:42px}}.lead,.muted{{color:var(--muted)}}.status{{display:inline-block;padding:8px 14px;border:1px solid var(--line);border-radius:99px;background:var(--panel);font-weight:700}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}.card,.plot{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:18px}}.metric{{font-size:26px;font-weight:750}}svg{{width:100%;height:auto;background:#0f1826;border-radius:9px}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{text-align:left;padding:9px 11px;border-bottom:1px solid var(--line)}}th{{color:var(--muted)}}code{{color:#a9c8ff}}a{{color:#8eb7ff}}.legend span{{margin-right:18px}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}}</style></head><body><main>
<h1>H200 conversion-cost calibration</h1><p class="lead">Fixed scalar x1 DOT, two uint32 streams, FP64 arithmetic, N=2²⁷, 512×256 first reduction, and uniform Philox4x32-10 raw codes.</p><p class="status">{status}</p>
<section class="grid"><div class="card"><div class="muted">Training inventory</div><div class="metric">112</div></div><div class="card"><div class="muted">Synthetic validation median / p90 APE</div><div class="metric">{acceptance['synthetic_validation_median_ape_percent']:.2f}% / {acceptance['synthetic_validation_p90_ape_percent']:.2f}%</div></div><div class="card"><div class="muted">Final median / max APE</div><div class="metric">{acceptance['final_real_median_ape_percent']:.2f}% / {acceptance['final_real_max_ape_percent']:.2f}%</div></div><div class="card"><div class="muted">Final Spearman</div><div class="metric">{acceptance['final_real_spearman']:.3f}</div></div></section>
<h2>Measured versus predicted</h2><p class="legend">{''.join(f'<span><i class="dot" style="background:{color}"></i>{split}</span>' for split,color in COLORS.items())}</p><div class="plot">{scatter(rows)}</div>
<h2>Held-out residuals</h2><div class="plot">{residuals(rows)}</div>
<h2>Untouched final formats</h2><div style="overflow:auto"><table><thead><tr><th>Case</th><th>Measured ms</th><th>Predicted ms</th><th>APE</th><th>Limit</th><th>Regs</th><th>Occupancy</th><th>Depth</th></tr></thead><tbody>{table_rows}</tbody></table></div>
<h2>Model</h2><p>The frozen form is <code>fixed + max(pipeline work × nonnegative cost) + nonnegative dependency/divergence/spill penalties</code>. It was fit only to the 112 training cases. Frozen model SHA-256: <code>{model['frozen_sha256']}</code>.</p><div style="overflow:auto"><table><thead><tr><th>Coefficient</th><th>ms per normalized unit</th></tr></thead><tbody>{params}</tbody></table></div>
<h2>Quality control</h2><ul><li>Maximum beginning/end anchor drift: {qc['anchor_drift_max']*100:.2f}% (limit 2%).</li><li>Noisy cases: {qc['noisy_case_count']} / 135; noisy means IQR/median &gt;2% (limit 2% of cases).</li><li>Timing QC: {'pass' if qc['timing_pass'] else 'fail'}.</li></ul>
<h2>Final-case metadata assumptions</h2><ul>{assumptions}</ul>
<h2>Artifacts</h2><ul><li><a href="../timing_samples.csv">raw timing samples</a></li><li><a href="../features.csv">SASS/resource features</a></li><li><a href="model.json">frozen model</a></li><li><a href="predictions.csv">all predictions</a></li><li><a href="../sass.txt">complete SASS dump</a></li><li><a href="../environment.txt">environment and GPU metadata</a></li></ul>
</main></body></html>'''
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(doc,encoding="utf-8")
