(function () {
  // "PNG" button above each table: exports the table as currently shown
  // (filters, renames, hidden rows/columns), without the +/- column toggles.
  function slug(s) {
    return (s || '').trim().replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '');
  }

  function fileName(tbl) {
    var sec = tbl.closest('section.category');
    var h = sec && sec.querySelector('h2');
    return [slug(h && h.textContent), slug(tbl.id)].filter(Boolean).join('_') + '.png';
  }

  document.querySelectorAll('table.ov-tbl').forEach(function (tbl) {
    var bar = document.createElement('div');
    bar.className = 'tbl-tools';
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = 'PNG';
    btn.title = 'Télécharger cette table en PNG';
    bar.appendChild(btn);
    tbl.parentNode.insertBefore(bar, tbl);

    btn.addEventListener('click', function () {
      if (!window.htmlToImage) { window.alert('html-to-image non chargé (pas de connexion au CDN ?)'); return; }
      btn.disabled = true;
      window.htmlToImage.toPng(tbl, {
        pixelRatio: 2,
        backgroundColor: '#ffffff',
        style: { margin: '0' },
        filter: function (node) { return !(node.classList && node.classList.contains('toggle-btn')); }
      }).then(function (url) {
        var a = document.createElement('a');
        a.href = url;
        a.download = fileName(tbl);
        document.body.appendChild(a);
        a.click();
        a.remove();
      }).catch(function (err) {
        window.alert('Export PNG impossible : ' + err);
      }).then(function () { btn.disabled = false; });
    });
  });
})();
