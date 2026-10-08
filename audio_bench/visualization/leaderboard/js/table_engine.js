(function () {
  // ===================================================================
  // Table engine: every score cell carries data-f = the id of the node
  // (in the score graph emitted by the Python side) it displays. Leaves are
  // (model, dataset) scores; means and display scaling are inner nodes. When
  // datasets or models are toggled, every cell, CI, rank colour, tooltip,
  // aggregate column and row order is recomputed from that graph.
  // ===================================================================
  var DATA = JSON.parse(document.getElementById('report-data').textContent);
  var NODES = DATA.nodes;
  var COLORS = DATA.colors;
  var NL = String.fromCharCode(10);
  var dsOn = DATA.datasets.map(function () { return true; });
  DATA.off.forEach(function (i) { dsOn[i] = false; });
  // Language mask, independent of the dataset selection: a dataset counts
  // when it is checked and one of its languages ("FR-EN" -> FR, EN) is on.
  var dsLangs = DATA.datasets.map(function (d) { return d[1].split('-'); });
  var langOn = {};
  dsLangs.forEach(function (ls) { ls.forEach(function (l) { langOn[l] = true; }); });
  function langOk(i) { return dsLangs[i].some(function (l) { return langOn[l]; }); }
  function active(i) { return dsOn[i] && langOk(i); }

  // --- Graph evaluation (memoized per refresh) ---
  var memo = new Array(NODES.length);
  var ascMemo = new Array(NODES.length);

  // sum() as CPython >= 3.12 computes it for floats (Neumaier compensation),
  // so means tie exactly where they tie on the Python side.
  function pysum(xs) {
    var s = 0, c = 0;
    for (var i = 0; i < xs.length; i++) {
      var x = xs[i], t = s + x;
      if (Math.abs(s) >= Math.abs(x)) c += (s - t) + x; else c += (x - t) + s;
      s = t;
    }
    return c && isFinite(c) ? s + c : s;
  }

  function value(id) {
    if (memo[id] !== undefined) return memo[id];
    var n = NODES[id], v = null;
    if (n[0] === 'L') {
      v = active(n[1]) ? n[2] : null;
    } else if (n[0] === 'M') {
      var xs = [];
      n[1].forEach(function (c) {
        var cv = value(c);
        if (cv !== null) xs.push(cv);
      });
      v = xs.length ? pysum(xs) / xs.length : null;
    } else {  // 'D'
      var x = value(n[1]);
      v = x === null ? null : Math.min(n[2] ? x * 100 : x, 100);
    }
    memo[id] = v;
    return v;
  }

  function isAsc(id) {  // lower is better (mirrors task_ascending)
    if (ascMemo[id] !== undefined) return ascMemo[id];
    var n = NODES[id], a;
    if (n[0] === 'L') a = n[3];
    else if (n[0] === 'D') a = isAsc(n[1]);
    else a = n[1].every(isAsc);
    ascMemo[id] = a;
    return a;
  }

  // CI half-width in display units, or null. A leaf uses its own std; a mean
  // pools the per-sample scores of its enabled leaves (like np.std(pooled)).
  function ciOf(id, pct) {
    var n = NODES[id];
    if (n[0] === 'D') n = NODES[id = n[1]];
    var std, cnt;
    if (n[0] === 'L') {
      if (!active(n[1]) || n[4][3] === null || !n[4][0]) return null;
      std = n[4][3]; cnt = n[4][0];
    } else {
      var acc = { n: 0, s: 0, ss: 0 };
      (function gather(i) {
        var x = NODES[i];
        if (x[0] === 'L') {
          if (active(x[1]) && x[4][0]) { acc.n += x[4][0]; acc.s += x[4][1]; acc.ss += x[4][2]; }
        } else if (x[0] === 'D') gather(x[1]);
        else x[1].forEach(gather);
      })(id);
      if (!acc.n) return null;
      var mean = acc.s / acc.n;
      std = Math.sqrt(Math.max(0, acc.ss / acc.n - mean * mean));
      cnt = acc.n;
    }
    var ci = 1.96 * std / Math.sqrt(cnt);
    return { ci: pct ? ci * 100 : ci, n: cnt };
  }

  // Match Python's f"{v:.nf}" (round-half-even on exact ties).
  var formatters = {};
  function fmt(v, digits) {
    var f = formatters[digits];
    if (f === undefined) {
      try {
        f = new Intl.NumberFormat('en-US', {
          minimumFractionDigits: digits, maximumFractionDigits: digits,
          roundingMode: 'halfEven', useGrouping: false });
      } catch (e) { f = null; }
      formatters[digits] = f;
    }
    return f ? f.format(v) : v.toFixed(digits);
  }

  function aggVal(x) {  // aggregate payload value: inline float or {n: node id}
    return typeof x === 'number' ? x : value(x.n);
  }

  // --- Tables ---
  var tables = Array.prototype.slice.call(document.querySelectorAll('table.ov-tbl'))
    .filter(function (t) { return t.tBodies.length && t.querySelector('td[data-f]'); })
    .map(function (tbl) {
      var payload = document.querySelector('script.agg-data[data-table="' + tbl.id + '"]');
      // Column index -> header cell (data columns span both header rows).
      var headers = {};
      if (tbl.tHead && tbl.tHead.rows.length) {
        var c = 0;
        Array.prototype.forEach.call(tbl.tHead.rows[0].cells, function (th) {
          if (th.rowSpan === 2) headers[c] = th;
          c += th.colSpan;
        });
      }
      var cols = {};
      Array.prototype.forEach.call(tbl.querySelectorAll('td[data-f]'), function (td) {
        (cols[td.cellIndex] = cols[td.cellIndex] || []).push(td);
      });
      return { tbl: tbl, headers: headers, cols: cols,
               agg: payload ? JSON.parse(payload.textContent) : null };
    });

  function rowOn(tr) { return !tr.classList.contains('xp-hidden'); }

  function computeAggs(data, on) {
    var rank = {}, mm = {}, zs = {};
    data.items.forEach(function (item) {
      var ms = [], hib = [], raw = {};
      Object.keys(item.s).forEach(function (m) {
        if (!on[m]) return;
        var v = aggVal(item.s[m]);
        if (v === null) return;
        ms.push(m); hib.push(item.asc ? 100 - v : v);
        raw[m] = item.r ? aggVal(item.r[m]) : v;
      });
      if (!ms.length) return;
      var lo = Math.min.apply(null, hib), hi = Math.max.apply(null, hib);
      var mean = hib.reduce(function (a, b) { return a + b; }, 0) / hib.length;
      var std = Math.sqrt(hib.reduce(function (a, b) { return a + (b - mean) * (b - mean); }, 0) / hib.length);
      // Avg Rank uses the unclamped scores when provided ("r").
      ms.slice().sort(function (a, b) { return item.asc ? raw[a] - raw[b] : raw[b] - raw[a]; })
        .forEach(function (m, r) { (rank[m] = rank[m] || []).push(r + 1); });
      ms.forEach(function (m, i) {
        var v = hib[i];
        (mm[m] = mm[m] || []).push(hi > lo ? (v - lo) / (hi - lo) : 1);
        (zs[m] = zs[m] || []).push(std > 0 ? (v - mean) / std : 0);
      });
    });
    function avg(obj) {
      var out = {};
      Object.keys(obj).forEach(function (m) {
        out[m] = pysum(obj[m]) / obj[m].length;
      });
      return out;
    }
    return { avg_rank: avg(rank), minmax: avg(mm), zscore: avg(zs) };
  }

  // Colour 1st / 2nd / before-last / last among *ranked* rows (best first).
  function rankColors(ranked) {
    var color = new Map(), n = ranked.length;
    if (n >= 1) color.set(ranked[0], COLORS.first);
    if (n >= 2) color.set(ranked[1], COLORS.second);
    if (n >= 3) color.set(ranked[n - 1], COLORS.last);
    if (n >= 4) color.set(ranked[n - 2], COLORS.before_last);
    return color;
  }

  // Only touch the DOM when a cell actually changes (keeps toggling fast).
  function setCell(td, html, bg) {
    if (td._html !== html) { td.innerHTML = html; td._html = html; }
    if (td._bg !== bg) { td.style.background = bg; td._bg = bg; }
  }

  function setMissing(td) {
    setCell(td, '-', COLORS.missing);
    td.removeAttribute('data-tip');
    td._tip = undefined;
  }

  function fillTooltip(template, ranks) {
    var out = [];
    template.split(NL).forEach(function (line) {
      var dropped = false;
      var text = line.replace(/\[\[([^\]]*)\]\]/g, function (_, tok) {
        var p = tok.split(':'), id = +p[1];
        if (p[0] === 'V') {
          var v = value(id);
          if (v === null) { dropped = true; return ''; }
          return fmt(v, 2);
        }
        if (p[0] === 'R') {
          var r = ranks[id];
          if (!r) return '';
          return p[2] === 'p' ? r + 'e — ' : ' (' + r + 'e)';
        }
        if (p[0] === 'C') {
          var ci = value(id) === null ? null : ciOf(id, p[2] === '1');
          if (!ci) return '';
          if (p[3] === 's') return '±' + fmt(ci.ci, 2);
          var v0 = value(id);
          return ' [' + fmt(v0 - ci.ci, 2) + ', ' + fmt(v0 + ci.ci, 2) + '], n=' + ci.n;
        }
        return '';
      });
      if (!dropped) out.push(text);
    });
    return out.join(NL);
  }

  function refreshTable(t) {
    var body = t.tbl.tBodies[0];
    var trs = Array.prototype.slice.call(body.rows).filter(function (tr) { return tr.hasAttribute('data-model'); });
    var on = {};
    trs.forEach(function (tr) { if (rowOn(tr)) on[tr.getAttribute('data-model')] = true; });

    // 1. Aggregates + row order (by the first aggregate). Rows are sorted
    // first, so that ties in the other aggregates are coloured in row order.
    if (t.agg) {
      var vals = computeAggs(t.agg, on);
      var first = t.agg.aggs[0];
      if (first) {
        var fv = vals[first.name];
        trs.sort(function (a, b) {
          var ma = a.getAttribute('data-model'), mb = b.getAttribute('data-model');
          var ha = ma in fv, hb = mb in fv;
          if (ha !== hb) return ha ? -1 : 1;
          if (!ha) return 0;
          var d = fv[ma] - fv[mb];
          return first.hib ? -d : d;
        });
        var same = trs.every(function (tr, i) { return body.rows[i] === tr; });
        if (!same) trs.forEach(function (tr) { body.appendChild(tr); });
      }
      t.agg.aggs.forEach(function (agg) {
        var v = vals[agg.name];
        var ranked = trs.filter(function (tr) { return on[tr.getAttribute('data-model')] && tr.getAttribute('data-model') in v; })
          .sort(function (a, b) {
            var d = v[a.getAttribute('data-model')] - v[b.getAttribute('data-model')];
            return agg.hib ? -d : d;
          });
        var color = rankColors(ranked);
        trs.forEach(function (tr) {
          var td = tr.querySelector('td[data-agg="' + agg.name + '"]');
          if (!td) return;
          var m = tr.getAttribute('data-model');
          if (m in v) setCell(td, fmt(v[m], agg.digits), color.get(tr) || '');
          else setCell(td, '-', COLORS.missing);
        });
      });
    }

    // 2. Per-column values, ranks and colours over the visible rows.
    var ranks = {};       // node id -> rank within its column
    var rowHasData = new Map();
    Object.keys(t.cols).forEach(function (ci) {
      var cells = t.cols[ci];
      var asc = isAsc(+cells[0].getAttribute('data-f'));
      var present = cells.filter(function (td) {
        var ok = value(+td.getAttribute('data-f')) !== null;
        if (ok) rowHasData.set(td.parentNode, true);
        return ok && rowOn(td.parentNode);
      });
      // Stable sort in (current) row order, as Python's _full_ranks.
      present.sort(function (a, b) { return a.parentNode.rowIndex - b.parentNode.rowIndex; });
      var ranked = present.slice().sort(function (a, b) {
        var d = value(+a.getAttribute('data-f')) - value(+b.getAttribute('data-f'));
        return asc ? d : -d;
      });
      ranked.forEach(function (td, i) { ranks[+td.getAttribute('data-f')] = i + 1; });
      var color = rankColors(ranked);
      cells.forEach(function (td) {
        var id = +td.getAttribute('data-f');
        var v = value(id);
        if (v === null) { setMissing(td); return; }
        var html = fmt(v, 2);
        if (td.hasAttribute('data-ci')) {
          var c = ciOf(id, td.getAttribute('data-ci') === '1');
          if (c) html += ' <span class="ci">±' + fmt(c.ci, 2) + '</span>';
        }
        setCell(td, html, color.get(td) || '');
      });
      // Hide a column with no value for any visible model.
      var hide = !present.length;
      trs.forEach(function (tr) {  // includes the "-" cells of models without data
        if (tr.cells[ci]) tr.cells[ci].classList.toggle('ds-hidden', hide);
      });
      if (t.headers[ci]) t.headers[ci].classList.toggle('ds-hidden', hide);
    });

    // 3. Tooltips (need the ranks of every column).
    Object.keys(t.cols).forEach(function (ci) {
      t.cols[ci].forEach(function (td) {
        var tt = td.getAttribute('data-tt');
        if (!tt || value(+td.getAttribute('data-f')) === null) return;
        var tip = fillTooltip(tt, ranks);
        if (td._tip !== tip) { td.setAttribute('data-tip', tip); td._tip = tip; }
        td.removeAttribute('title');
      });
    });

    // 4. Rows without any value left (all their datasets unchecked).
    trs.forEach(function (tr) { tr.classList.toggle('ds-empty', !rowHasData.get(tr)); });
    var anyCol = Object.keys(t.cols).some(function (ci) {
      return !t.cols[ci][0].classList.contains('ds-hidden');
    });
    t.tbl.classList.toggle('ds-empty-tbl', !anyCol);
  }

  // Hide figure wrappers whose tables are all empty, then empty sections.
  function refreshSections() {
    document.querySelectorAll('.figure-wrapper').forEach(function (w) {
      var tbls = w.querySelectorAll('table.ov-tbl');
      var empty = tbls.length > 0 && Array.prototype.every.call(tbls, function (tb) {
        return tb.classList.contains('ds-empty-tbl');
      });
      w.classList.toggle('ds-hidden', empty);
    });
    document.querySelectorAll('section.category').forEach(function (sec) {
      var ws = sec.querySelectorAll('.figure-wrapper');
      var empty = ws.length > 0 && Array.prototype.every.call(ws, function (w) {
        return w.classList.contains('ds-hidden');
      });
      sec.classList.toggle('ds-hidden', empty);
      var link = document.querySelector('nav.sidebar a[href="#' + sec.id + '"]');
      if (link) link.parentNode.classList.toggle('ds-hidden', empty);
    });
  }

  function refresh() {
    memo = new Array(NODES.length);
    tables.forEach(refreshTable);
    refreshSections();
  }
  window.refreshReportTables = refresh;

  // ===================================================================
  // Dataset filter panel: Task -> "LANG · dataset" checkboxes.
  // ===================================================================
  var treeEl = document.getElementById('ds-filter-tree');
  var toggleBtn = document.getElementById('ds-toggle-all');
  var resetBtn = document.getElementById('ds-reset');
  var leafBoxes = [];   // dataset idx -> checkbox
  var leafItems = [];   // dataset idx -> <li> (dimmed when its languages are off)
  var groupBoxes = [];  // { cb, leaves: [idx] }

  // Task -> language -> dataset. Tasks with a single language skip the
  // language level (their leaves read "LANG · dataset").
  function langKey(l) {  // FR, EN first, like the tables
    var i = ['FR', 'EN'].indexOf(l.split('-')[0]);
    return (i === -1 ? '2' : '' + i) + l;
  }

  function groupNode(container, label, idxs) {  // returns the <ul> for children
    var li = document.createElement('li');
    var details = document.createElement('details');
    var summary = document.createElement('summary');
    var caret = document.createElement('span');
    caret.className = 'xp-caret';
    caret.textContent = '▸';
    var gcb = document.createElement('input');
    gcb.type = 'checkbox';
    var text = document.createElement('span');
    text.className = 'xp-group';
    text.textContent = label;
    var count = document.createElement('span');
    count.className = 'xp-count';
    summary.appendChild(caret); summary.appendChild(gcb);
    summary.appendChild(text); summary.appendChild(count);
    details.appendChild(summary);
    var ul = document.createElement('ul');
    details.appendChild(ul);
    li.appendChild(details);
    container.appendChild(li);
    gcb.addEventListener('click', function (e) { e.stopPropagation(); });
    gcb.addEventListener('change', function () {
      idxs.forEach(function (i) { dsOn[i] = gcb.checked; });
      update();
    });
    groupBoxes.push({ cb: gcb, leaves: idxs, count: count });
    return ul;
  }

  function leafNode(container, i, label) {
    var li = document.createElement('li');
    var lab = document.createElement('label');
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.addEventListener('change', function () { dsOn[i] = cb.checked; update(); });
    leafBoxes[i] = cb;
    leafItems[i] = li;
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(label));
    li.appendChild(lab);
    container.appendChild(li);
  }

  var byTask = {};
  DATA.datasets.forEach(function (d, i) {
    var langs = byTask[d[0]] = byTask[d[0]] || {};
    (langs[d[1]] = langs[d[1]] || []).push(i);
  });
  var rootUl = document.createElement('ul');
  Object.keys(byTask).sort().forEach(function (task) {
    var langs = Object.keys(byTask[task]).sort(function (a, b) {
      return langKey(a) < langKey(b) ? -1 : langKey(a) > langKey(b) ? 1 : 0;
    });
    var byName = function (a, b) {
      var x = DATA.datasets[a][2], y = DATA.datasets[b][2];
      return x < y ? -1 : x > y ? 1 : 0;
    };
    var all = [];
    langs.forEach(function (l) { all = all.concat(byTask[task][l].sort(byName)); });
    var taskUl = groupNode(rootUl, task, all);
    if (langs.length === 1) {
      all.forEach(function (i) {
        leafNode(taskUl, i, DATA.datasets[i][1] + ' · ' + DATA.datasets[i][2]);
      });
      return;
    }
    langs.forEach(function (l) {
      var idxs = byTask[task][l];
      var langUl = groupNode(taskUl, l, idxs);
      idxs.forEach(function (i) { leafNode(langUl, i, DATA.datasets[i][2]); });
    });
  });
  treeEl.appendChild(rootUl);

  function syncBoxes() {
    leafBoxes.forEach(function (cb, i) { if (cb) cb.checked = dsOn[i]; });
    leafItems.forEach(function (li, i) { if (li) li.classList.toggle('flt-masked', !langOk(i)); });
    groupBoxes.forEach(function (g) {
      var n = g.leaves.filter(function (i) { return dsOn[i]; }).length;
      g.cb.checked = n === g.leaves.length;
      g.cb.indeterminate = n > 0 && n < g.leaves.length;
      g.count.textContent = '(' + n + '/' + g.leaves.length + ')';
    });
    toggleBtn.textContent = dsOn.every(Boolean) ? 'Uncheck all' : 'Check all';
  }

  function update() { syncBoxes(); refresh(); }

  toggleBtn.addEventListener('click', function () {
    var next = !dsOn.every(Boolean);
    dsOn = dsOn.map(function () { return next; });
    update();
  });
  function selectDefault(superCats) {
    dsOn = DATA.super_cats.map(function (sc) { return !superCats || superCats.indexOf(sc) !== -1; });
    DATA.off.forEach(function (i) { dsOn[i] = false; });
    update();
  }
  resetBtn.addEventListener('click', function () { selectDefault(null); });
  document.getElementById('ds-core').addEventListener('click', function () {
    selectDefault(['ASR', 'AST', 'QA']);
  });

  // ===================================================================
  // Language filter panel: one checkbox per language.
  // ===================================================================
  var langTreeEl = document.getElementById('lang-filter-tree');
  var langToggleBtn = document.getElementById('lang-toggle-all');
  var langs = Object.keys(langOn).sort(function (a, b) {
    return langKey(a) < langKey(b) ? -1 : langKey(a) > langKey(b) ? 1 : 0;
  });
  var langBoxes = {};
  var langUl = document.createElement('ul');
  langs.forEach(function (l) {
    var n = dsLangs.filter(function (ls) { return ls.indexOf(l) !== -1; }).length;
    var li = document.createElement('li');
    var lab = document.createElement('label');
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.addEventListener('change', function () { langOn[l] = cb.checked; updateLangs(); });
    langBoxes[l] = cb;
    var count = document.createElement('span');
    count.className = 'xp-count';
    count.textContent = '(' + n + ')';
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(l));
    lab.appendChild(count);
    li.appendChild(lab);
    langUl.appendChild(li);
  });
  langTreeEl.appendChild(langUl);

  function allLangsOn() { return langs.every(function (l) { return langOn[l]; }); }

  function updateLangs() {
    langs.forEach(function (l) { langBoxes[l].checked = langOn[l]; });
    langToggleBtn.textContent = allLangsOn() ? 'Uncheck all' : 'Check all';
    update();
  }

  function selectLangs(only) {
    langs.forEach(function (l) { langOn[l] = !only || only.indexOf(l) !== -1; });
    updateLangs();
  }
  langToggleBtn.addEventListener('click', function () {
    var next = !allLangsOn();
    langs.forEach(function (l) { langOn[l] = next; });
    updateLangs();
  });
  document.getElementById('lang-fren').addEventListener('click', function () {
    selectLangs(['FR', 'EN']);
  });

  updateLangs();
})();
