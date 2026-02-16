import json

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Detection Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 14px;
  color: #2a2a2a;
  background: #f5f5f7;
}}

.layout {{ display: flex; height: 100vh; }}

.main {{
  flex: 1;
  overflow-y: auto;
  padding: 24px 32px;
}}

.side-panel {{
  width: 440px;
  border-left: 1px solid #e0e0e0;
  overflow-y: auto;
  padding: 20px;
  display: none;
  flex-shrink: 0;
  background: #fff;
  box-shadow: -2px 0 8px rgba(0,0,0,0.04);
}}
.side-panel.open {{ display: block; }}

h1 {{
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 4px;
  color: #1a1a1a;
}}

.controls {{
  position: sticky;
  top: 0;
  background: #f5f5f7;
  padding: 12px 0 16px;
  border-bottom: 1px solid #e0e0e0;
  margin-bottom: 16px;
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.controls button {{
  padding: 6px 16px;
  border: 1px solid #d0d0d0;
  background: #fff;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  color: #555;
  transition: all 0.15s ease;
}}
.controls button:hover {{ background: #eee; }}
.controls button.active {{
  background: #3a3a3a;
  color: #fff;
  border-color: #3a3a3a;
}}

.block {{
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 12.5px;
  padding: 12px 14px;
  margin-bottom: 8px;
  border-radius: 6px;
  cursor: pointer;
  border: 2px solid transparent;
  line-height: 1.55;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}
.block:hover {{
  border-color: #aaa;
  box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}}
.block.selected {{
  border-color: #555;
  box-shadow: 0 2px 8px rgba(0,0,0,0.14);
}}

.block-meta {{
  font-size: 11px;
  color: #888;
  margin-bottom: 3px;
  font-family: sans-serif;
  padding-left: 2px;
}}

.side-panel h3 {{
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 10px;
  color: #1a1a1a;
}}
.side-panel .close-btn {{
  float: right;
  cursor: pointer;
  background: none;
  border: none;
  font-size: 20px;
  line-height: 1;
  color: #999;
  transition: color 0.1s;
}}
.side-panel .close-btn:hover {{ color: #333; }}

.sentence {{
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 12.5px;
  padding: 6px 8px;
  margin-bottom: 3px;
  border-radius: 4px;
  line-height: 1.55;
}}
.sentence-meta {{
  font-size: 10px;
  color: #999;
  text-align: right;
  margin-bottom: 6px;
}}
</style>
</head>
<body>
<div class="layout">
  <div class="main" id="main">
    <h1>AI Detection Report</h1>
    <div class="controls">
      <button id="btn-chrono" class="active" onclick="sortBy('chrono')">Chronological</button>
      <button id="btn-score" onclick="sortBy('score')">By AI probability</button>
      <span id="summary" style="margin-left: auto; font-size: 12px; color: #999;"></span>
    </div>
    <div id="blocks"></div>
  </div>
  <div class="side-panel" id="sidePanel">
    <button class="close-btn" onclick="closePanel()">&times;</button>
    <h3>Sentence breakdown</h3>
    <p id="panel-meta" style="font-size:12px;color:#888;margin-bottom:12px;"></p>
    <div id="panel-sentences"></div>
  </div>
</div>

<script>
const DATA = {data_json};
const MIN_SENTENCE_LEN = {min_sentence_len};

let currentSort = 'chrono';
let selectedIdx = null;

function scoreToColor(score) {{
  // Softer palette: white at 0, muted salmon/rose at 1
  const r = 255;
  const g = Math.round(255 - score * 110);
  const b = Math.round(255 - score * 100);
  return `rgb(${{r}}, ${{g}}, ${{b}})`;
}}

function renderBlocks() {{
  const container = document.getElementById('blocks');
  container.innerHTML = '';

  let indices = DATA.map((_, i) => i);
  if (currentSort === 'score') {{
    indices.sort((a, b) => DATA[b].score - DATA[a].score);
  }}

  indices.forEach(i => {{
    const d = DATA[i];
    const meta = document.createElement('div');
    meta.className = 'block-meta';
    meta.textContent = `Cell ${{d.index}}  \u00b7  Line ${{d.line_start}}  \u00b7  P(AI) = ${{(d.score * 100).toFixed(1)}}%`;

    const block = document.createElement('pre');
    block.className = 'block' + (selectedIdx === i ? ' selected' : '');
    block.style.backgroundColor = scoreToColor(d.score);
    block.textContent = d.content;
    block.dataset.idx = i;
    block.onclick = () => selectBlock(i);

    container.appendChild(meta);
    container.appendChild(block);
  }});

  document.getElementById('summary').textContent =
    `${{DATA.length}} blocks  \u00b7  mean P(AI) = ${{(DATA.reduce((s, d) => s + d.score, 0) / DATA.length * 100).toFixed(1)}}%`;
}}

function selectBlock(i) {{
  selectedIdx = i;
  renderBlocks();
  showSentences(i);
}}

function showSentences(i) {{
  const panel = document.getElementById('sidePanel');
  const d = DATA[i];
  panel.classList.add('open');

  document.getElementById('panel-meta').textContent =
    `Cell ${{d.index}}  \u00b7  Line ${{d.line_start}}  \u00b7  Overall P(AI) = ${{(d.score * 100).toFixed(1)}}%`;

  const container = document.getElementById('panel-sentences');
  container.innerHTML = '';

  const filtered = (d.sentence_scores || []).filter(s => s.sentence.length >= MIN_SENTENCE_LEN);

  if (filtered.length === 0) {{
    container.innerHTML = '<p style="color:#aaa;font-size:12px;">No sentence-level data available for this block.</p>';
    return;
  }}

  filtered.forEach(s => {{
    const div = document.createElement('div');
    div.className = 'sentence';
    div.style.backgroundColor = scoreToColor(s.score);
    div.textContent = s.sentence;

    const meta = document.createElement('div');
    meta.className = 'sentence-meta';
    meta.textContent = `${{(s.score * 100).toFixed(1)}}%`;

    container.appendChild(div);
    container.appendChild(meta);
  }});
}}

function closePanel() {{
  document.getElementById('sidePanel').classList.remove('open');
  selectedIdx = null;
  renderBlocks();
}}

function sortBy(mode) {{
  currentSort = mode;
  document.getElementById('btn-chrono').classList.toggle('active', mode === 'chrono');
  document.getElementById('btn-score').classList.toggle('active', mode === 'score');
  renderBlocks();
}}

renderBlocks();
</script>
</body>
</html>
"""


def generate_html(scored_cells, output_path, min_sentence_len=10):
    """Generate an interactive HTML report from scored cells."""
    data = []
    for cell in scored_cells:
        data.append(
            {
                "content": cell["content"],
                "index": cell["index"],
                "line_start": cell["line_start"],
                "score": cell["score"],
                "sentence_scores": cell.get("sentence_scores", []),
            }
        )

    data_json = json.dumps(data, ensure_ascii=False)
    html_content = HTML_TEMPLATE.format(data_json=data_json, min_sentence_len=min_sentence_len)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Report saved to {output_path}")
