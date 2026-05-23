"""
Mercedes-Benz Customer Feedback Insights Dashboard
==================================================
Flask app that:
    * lets a service engineer upload a feedback CSV
    * runs the test_run pipeline on it
    * surfaces the resulting clusters as actionable failure insights

Endpoints
---------
    GET  /                    overview + cluster cards
    GET  /cluster/<int:cid>   single-cluster detail page
    GET  /scatter             UMAP 2D scatter view of every record
    POST /upload              CSV upload -> runs pipeline -> redirect to /
    GET  /api/clusters        JSON of clusters
    GET  /api/stats           JSON of pipeline stats
    GET  /api/scatter         JSON of 2D coords + cluster colors
"""
from __future__ import annotations
import json, os, subprocess, sys
from collections import Counter
from pathlib import Path

from flask import (Flask, abort, flash, jsonify, redirect, render_template,
                   request, url_for)
from werkzeug.utils import secure_filename

ROOT       = Path(__file__).resolve().parents[1]
RAW_DIR    = ROOT / "data" / "raw"
OUT_DIR    = ROOT / "outputs"
PIPE_SCRIPT = ROOT / "scripts" / "test_run.py"

app = Flask(__name__,
            template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))
app.secret_key = "mercedes-feedback-dashboard"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB CSV

# -------------------------------------------------------------------- helpers
def load_clusters() -> list[dict]:
    p = OUT_DIR / "clusters_final.json"
    if not p.exists(): return []
    return json.loads(p.read_text(encoding="utf-8"))

def load_stats() -> dict:
    p = OUT_DIR / "pipeline_stats.json"
    if not p.exists(): return {}
    return json.loads(p.read_text(encoding="utf-8"))

def severity(size: int, total: int) -> str:
    if total == 0: return "low"
    pct = size / total
    if pct > 0.10: return "critical"
    if pct > 0.05: return "high"
    if pct > 0.02: return "medium"
    return "low"

def component_summary(clusters):
    """Aggregate cluster sizes by top_component for the bar chart."""
    by_comp = Counter()
    for c in clusters:
        comp = c.get("top_component") or "unidentified"
        by_comp[comp] += c["size"]
    return [{"component": k, "count": v}
            for k, v in by_comp.most_common(15)]

def domain_summary(clusters):
    by_dom = Counter()
    for c in clusters:
        for d in c.get("domains", []):
            by_dom[d] += c["size"]
    return [{"domain": k, "count": v} for k, v in by_dom.most_common()]

# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    clusters = load_clusters()
    stats = load_stats()
    total = sum(c["size"] for c in clusters) or 1

    domain_filter   = request.args.get("domain", "all")
    language_filter = request.args.get("language", "all")
    sort_by         = request.args.get("sort", "size")

    filtered = clusters
    if domain_filter != "all":
        filtered = [c for c in filtered if domain_filter in c.get("domains", [])]
    if language_filter != "all":
        filtered = [c for c in filtered if language_filter in c.get("languages", [])]

    if sort_by == "size":
        filtered.sort(key=lambda c: c["size"], reverse=True)
    elif sort_by == "label":
        filtered.sort(key=lambda c: c["label"].lower())

    for c in filtered:
        c["severity"] = severity(c["size"], total)
        c["share_pct"] = round(c["size"] / total * 100, 1)

    return render_template(
        "index.html",
        clusters=filtered,
        stats=stats,
        total_in_clusters=total,
        all_domains=sorted({d for c in clusters for d in c.get("domains", [])}),
        all_languages=sorted({l for c in clusters for l in c.get("languages", [])}),
        domain_filter=domain_filter,
        language_filter=language_filter,
        sort_by=sort_by,
        component_chart=component_summary(clusters),
        domain_chart=domain_summary(clusters),
    )

@app.route("/cluster/<int:cid>")
def cluster_detail(cid):
    clusters = load_clusters()
    cluster = next((c for c in clusters if c["cluster_id"] == cid), None)
    if not cluster:
        abort(404)
    total = sum(c["size"] for c in clusters) or 1
    cluster["severity"] = severity(cluster["size"], total)
    cluster["share_pct"] = round(cluster["size"] / total * 100, 1)
    return render_template("cluster.html", cluster=cluster, stats=load_stats())

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("csv_file")
    if not f or not f.filename.endswith(".csv"):
        flash("Please upload a .csv file with a 'feedback' column.")
        return redirect(url_for("index"))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / "all_feedback.csv"
    f.save(dest)
    flash(f"Uploaded {f.filename}. Running pipeline...")
    try:
        # run the pipeline synchronously (small data); for production use a queue
        result = subprocess.run([sys.executable, str(PIPE_SCRIPT)],
                                capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            flash(f"Pipeline failed:\n{result.stderr[-500:]}")
        else:
            stats = load_stats()
            flash(f"Pipeline complete in {stats.get('runtime_seconds')}s — "
                  f"{stats.get('n_clusters_final')} clusters from "
                  f"{stats.get('records_after_clean_dedup')} cleaned records.")
    except subprocess.TimeoutExpired:
        flash("Pipeline timed out (>10 min). Try a smaller file.")
    return redirect(url_for("index"))

@app.route("/api/clusters")
def api_clusters():
    return jsonify(load_clusters())

@app.route("/api/stats")
def api_stats():
    return jsonify(load_stats())

@app.route("/api/scatter")
def api_scatter():
    """Return per-record 2D UMAP coords + final cluster id for the scatter plot."""
    import csv as _csv
    p = OUT_DIR / "embeddings_2d.csv"
    if not p.exists():
        # fall back to record_assignments_final.csv if it has x_2d/y_2d
        p2 = OUT_DIR / "record_assignments_final.csv"
        if p2.exists():
            with p2.open() as f:
                rows = list(_csv.DictReader(f))
            if rows and "x_2d" in rows[0]:
                return jsonify([
                    {"id": r.get("record_id", ""),
                     "x": float(r["x_2d"]), "y": float(r["y_2d"]),
                     "cluster": int(float(r.get("cluster_id_final", -1)))}
                    for r in rows if r.get("x_2d")
                ])
        return jsonify([])
    with p.open() as f:
        rows = list(_csv.DictReader(f))
    return jsonify([
        {"id": r["record_id"],
         "x": float(r["x_2d"]), "y": float(r["y_2d"]),
         "cluster": int(float(r["cluster_id_final"]))}
        for r in rows
    ])

@app.route("/scatter")
def scatter_view():
    """Full-page UMAP scatter visualization."""
    return render_template("scatter.html", stats=load_stats(),
                           clusters=load_clusters())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
