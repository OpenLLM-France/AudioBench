(function () {
  var tip = document.getElementById('celltip');
  var NL = String.fromCharCode(10);

  function stripEnd(str) {
    while (str.length && str.charAt(str.length - 1) === ' ') str = str.slice(0, -1);
    return str;
  }

  function render(text) {
    if (window.xpRenameText) text = window.xpRenameText(text);
    tip.textContent = '';
    var grid = null;
    text.split(NL).forEach(function (line) {
      if (line === '') return;
      var idx = line.indexOf(': ');
      if (idx === -1) {                       // header / summary line -> full width
        grid = null;
        var h = document.createElement('div');
        h.className = 'ct-head';
        h.textContent = line;
        tip.appendChild(h);
        return;
      }
      if (!grid) {                            // start a fresh aligned block
        grid = document.createElement('div');
        grid.className = 'ct-grid';
        tip.appendChild(grid);
      }
      var k = line.slice(0, idx), v = line.slice(idx + 2), r = '';
      if (v.charAt(v.length - 1) === ')') {   // peel a trailing "(3e)" rank marker
        var op = v.lastIndexOf('(');
        if (op !== -1) { r = v.slice(op + 1, v.length - 1); v = stripEnd(v.slice(0, op)); }
      }
      var ke = document.createElement('span'); ke.className = 'ct-k'; ke.textContent = k;
      var ve = document.createElement('span'); ve.className = 'ct-v'; ve.textContent = v;
      var re = document.createElement('span'); re.className = 'ct-r'; re.textContent = r;
      grid.appendChild(ke); grid.appendChild(ve); grid.appendChild(re);
    });
  }

  function position(e) {
    var pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    var x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > window.innerWidth - 8)  x = e.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = e.clientY - h - pad;
    tip.style.left = Math.max(4, x) + 'px';
    tip.style.top  = Math.max(4, y) + 'px';
  }

  // Move native title="" onto data-tip so the browser tooltip does not compete.
  document.querySelectorAll('td[title], th[title]').forEach(function (el) {
    el.setAttribute('data-tip', el.getAttribute('title'));
    el.removeAttribute('title');
  });

  document.addEventListener('mouseover', function (e) {
    var el = e.target.closest('[data-tip]');
    if (!el) return;
    render(el.getAttribute('data-tip'));
    tip.classList.add('show');
    position(e);
  });
  document.addEventListener('mousemove', function (e) {
    if (tip.classList.contains('show')) position(e);
  });
  document.addEventListener('mouseout', function (e) {
    var el = e.target.closest('[data-tip]');
    if (!el) return;
    if (e.relatedTarget && el.contains(e.relatedTarget)) return;
    tip.classList.remove('show');
  });
})();
