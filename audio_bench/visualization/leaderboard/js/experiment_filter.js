(function () {
  var rows = Array.prototype.slice.call(document.querySelectorAll('tr[data-model]'));
  var models = [];
  rows.forEach(function (tr) {
    var m = tr.getAttribute('data-model');
    if (models.indexOf(m) === -1) models.push(m);
  });
  models.sort();
  if (!models.length) return;

  var treeEl = document.getElementById('xp-filter-tree');
  var toggleBtn = document.getElementById('xp-toggle-all');
  var checked = {};
  // Models listed in hidden_models (config) start unchecked.
  var offModels = JSON.parse(document.getElementById('report-data').textContent).off_models || [];
  models.forEach(function (m) { checked[m] = offModels.indexOf(m) === -1; });

  // -------------------------------------------------------------------
  // Build a trie of models, tokenized on "/" and "_" (keeping the
  // delimiter that preceded each token), so experiments that share a
  // path/name prefix ("LINAGORA/Canary_Luciole-1B_..._v3_buckets_...")
  // group and nest automatically -- no hardcoded naming knowledge needed.
  // -------------------------------------------------------------------
  function tokenize(name) {
    var parts = name.split(/([/_])/);
    var tokens = [], delims = [];
    for (var i = 0; i < parts.length; i += 2) tokens.push(parts[i]);
    for (var j = 1; j < parts.length; j += 2) delims.push(parts[j]);
    return { tokens: tokens, delims: delims };
  }

  function newNode() { return { children: new Map(), models: [] }; }
  var root = newNode();
  models.forEach(function (m) {
    var t = tokenize(m);
    var cur = root;
    for (var i = 0; i < t.tokens.length; i++) {
      var tok = t.tokens[i];
      var delim = i === 0 ? null : t.delims[i - 1];
      if (!cur.children.has(tok)) cur.children.set(tok, { delim: delim, node: newNode() });
      cur = cur.children.get(tok).node;
    }
    cur.models.push(m);
  });

  // Compress chains of single, model-less children into one label so the
  // tree only branches where models actually diverge.
  function compressChildren(node) {
    var out = [];
    node.children.forEach(function (edge, tok) {
      var label = (edge.delim || '') + tok;
      var child = edge.node;
      while (child.models.length === 0 && child.children.size === 1) {
        var onlyTok, onlyEdge;
        child.children.forEach(function (e, t) { onlyTok = t; onlyEdge = e; });
        label += (onlyEdge.delim || '') + onlyTok;
        child = onlyEdge.node;
      }
      out.push({ label: label, node: child });
    });
    return out;
  }

  // -------------------------------------------------------------------
  // Render. Returns the list of leaf model names under the rendered node,
  // and registers group checkboxes so their tri-state can be refreshed.
  // -------------------------------------------------------------------
  var groupBoxes = []; // { cb, leaves: [model,...] }
  var leafBoxes = {};  // model -> checkbox element
  var leafAliases = {}; // model -> span showing its display name when renamed

  // -------------------------------------------------------------------
  // Display-only renaming (for clean screenshots, e.g. model cards).
  // data-model and the filter tree keep the real name; only the visible
  // name cells, tooltips and Plotly labels change. Saved per browser.
  // -------------------------------------------------------------------
  var RENAME_KEY = 'audiobench-xp-renames';
  var renames = {};
  try { renames = JSON.parse(localStorage.getItem(RENAME_KEY) || '{}') || {}; } catch (e) { renames = {}; }

  function displayName(m) { return renames[m] || m; }

  function saveRenames() {
    try { localStorage.setItem(RENAME_KEY, JSON.stringify(renames)); } catch (e) {}
  }

  function askRename(m) {
    var v = window.prompt('New name for ' + m + ' (empty = original name):', displayName(m));
    if (v === null) return;
    v = v.trim();
    if (v && v !== m) renames[m] = v; else delete renames[m];
    saveRenames();
    applyRenames();
  }

  // Longest names first so a name that prefixes another is not replaced inside it.
  window.xpRenameText = function (text) {
    Object.keys(renames).sort(function (a, b) { return b.length - a.length; }).forEach(function (m) {
      text = text.split(m).join(renames[m]);
    });
    return text;
  };

  var isModel = {};
  models.forEach(function (m) { isModel[m] = true; });

  function hasModel(v) {
    if (typeof v === 'string') return isModel[v] === true;
    return Array.isArray(v) && v.some(function (x) { return typeof x === 'string' && isModel[x] === true; });
  }

  function renameValue(v) {
    if (typeof v === 'string') return isModel[v] === true ? displayName(v) : v;
    if (Array.isArray(v)) return v.map(renameValue);
    return v;
  }

  // Each figure remembers, once, the original value of every trace field that
  // holds a model name; renaming rewrites those fields in place and redraws
  // only the figures that reference a model (one redraw per figure).
  var PLOT_FIELDS = ['name', 'x', 'y', 'text', 'hovertext', 'legendgroup', 'labels', 'theta'];
  function renamePlots() {
    if (!window.Plotly) return;
    document.querySelectorAll('.js-plotly-plot').forEach(function (gd) {
      if (!gd.data) return;
      if (!gd._xpOrig) {
        if (!Object.keys(renames).length) return;
        gd._xpOrig = [];
        gd.data.forEach(function (tr, i) {
          PLOT_FIELDS.forEach(function (f) {
            if (hasModel(tr[f])) gd._xpOrig.push({ i: i, f: f, v: tr[f] });
          });
        });
      }
      if (!gd._xpOrig.length) return;
      gd._xpOrig.forEach(function (o) { gd.data[o.i][o.f] = renameValue(o.v); });
      window.Plotly.redraw(gd);
    });
  }

  function applyRenames() {
    rows.forEach(function (tr) {
      var td = tr.querySelector('td.mname');
      if (td) (td.querySelector('a') || td).textContent = displayName(tr.getAttribute('data-model'));
    });
    models.forEach(function (m) {
      var a = leafAliases[m];
      if (a) a.textContent = renames[m] ? ' → ' + renames[m] : '';
    });
    renamePlots();
  }

  document.addEventListener('dblclick', function (e) {
    var td = e.target.closest('td.mname');
    if (!td || !td.parentNode.hasAttribute('data-model')) return;
    if (e.target.closest('a')) return;  // a linked name: rename it from the sidebar
    askRename(td.parentNode.getAttribute('data-model'));
  });

  document.getElementById('xp-rename-reset').addEventListener('click', function () {
    if (!Object.keys(renames).length) return;
    if (!window.confirm('Restore the original names for all experiments?')) return;
    renames = {};
    saveRenames();
    applyRenames();
  });

  function renderLeaf(container, modelName) {
    var li = document.createElement('li');
    var label = document.createElement('label');
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = checked[modelName];
    cb.addEventListener('change', function () {
      checked[modelName] = cb.checked;
      refreshGroups();
      applyFilter();
    });
    leafBoxes[modelName] = cb;
    label.appendChild(cb);
    var name = document.createElement('span');
    name.textContent = modelName;
    label.appendChild(name);
    var alias = document.createElement('span');
    alias.className = 'xp-alias';
    label.appendChild(alias);
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'xp-rename';
    btn.title = 'Rename (display only)';
    btn.textContent = '✎';
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      askRename(modelName);
    });
    label.appendChild(btn);
    leafAliases[modelName] = alias;
    li.appendChild(label);
    container.appendChild(li);
  }

  function renderNode(container, label, node, depth) {
    var childEntries = compressChildren(node);
    var isPureLeaf = node.models.length > 0 && childEntries.length === 0;

    if (isPureLeaf) {
      node.models.forEach(function (m) { renderLeaf(container, m); });
      return node.models.slice();
    }

    var li = document.createElement('li');
    var details = document.createElement('details');
    details.open = depth < 1;
    var summary = document.createElement('summary');
    var caret = document.createElement('span');
    caret.className = 'xp-caret';
    caret.textContent = '▸';
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    var text = document.createElement('span');
    text.className = 'xp-group';
    text.textContent = label;
    summary.appendChild(caret);
    summary.appendChild(cb);
    summary.appendChild(text);
    var count = document.createElement('span');
    count.className = 'xp-count';
    summary.appendChild(count);
    details.appendChild(summary);

    var ul = document.createElement('ul');
    var leaves = [];
    // A model that terminates exactly at this group node (rare: its name
    // is itself a prefix of a sibling's name) is shown as its own leaf too.
    node.models.forEach(function (m) {
      renderLeaf(ul, m);
      leaves.push(m);
    });
    childEntries.forEach(function (entry) {
      leaves = leaves.concat(renderNode(ul, entry.label, entry.node, depth + 1));
    });
    details.appendChild(ul);
    li.appendChild(details);
    container.appendChild(li);

    count.textContent = '(' + leaves.length + ')';
    cb.addEventListener('click', function (e) { e.stopPropagation(); });
    cb.addEventListener('change', function () {
      leaves.forEach(function (m) {
        checked[m] = cb.checked;
        var lb = leafBoxes[m];
        if (lb) lb.checked = cb.checked;
      });
      refreshGroups();
      applyFilter();
    });
    groupBoxes.push({ cb: cb, leaves: leaves });
    return leaves;
  }

  var rootUl = document.createElement('ul');
  compressChildren(root).forEach(function (entry) {
    renderNode(rootUl, entry.label, entry.node, 0);
  });
  treeEl.appendChild(rootUl);

  function refreshGroups() {
    groupBoxes.forEach(function (g) {
      var n = g.leaves.filter(function (m) { return checked[m]; }).length;
      g.cb.checked = n === g.leaves.length;
      g.cb.indeterminate = n > 0 && n < g.leaves.length;
    });
  }

  function applyFilter() {
    rows.forEach(function (tr) {
      var m = tr.getAttribute('data-model');
      tr.classList.toggle('xp-hidden', !checked[m]);
    });
    if (window.refreshReportTables) window.refreshReportTables();
    var allChecked = models.every(function (m) { return checked[m]; });
    toggleBtn.textContent = allChecked ? 'Uncheck all' : 'Check all';
    updateCount();
  }

  function updateCount() {
    document.getElementById('xp-count').textContent =
      '(' + models.filter(function (m) { return checked[m]; }).length + '/' + models.length + ')';
  }

  toggleBtn.addEventListener('click', function () {
    var allChecked = models.every(function (m) { return checked[m]; });
    var next = !allChecked;
    models.forEach(function (m) { checked[m] = next; });
    Object.keys(leafBoxes).forEach(function (m) { leafBoxes[m].checked = next; });
    refreshGroups();
    applyFilter();
  });

  refreshGroups();
  applyFilter();
  applyRenames();
})();
