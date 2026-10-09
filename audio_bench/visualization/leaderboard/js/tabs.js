// Tabs: one section shown at a time, picked by the URL hash (#cat-...), like
// the ASR leaderboard. Sections emptied by the filters lose their tab; when the
// current one goes, the first remaining tab is shown.
(function () {
  var links = Array.prototype.slice.call(document.querySelectorAll('#tabs a'));
  var sections = Array.prototype.slice.call(document.querySelectorAll('section.category'));

  function available(sec) { return !sec.classList.contains('ds-hidden'); }

  function sync() {
    var id = location.hash.slice(1);
    var current = sections.filter(function (s) { return s.id === id && available(s); })[0]
      || sections.filter(available)[0];
    sections.forEach(function (s) {
      var show = s === current;
      if (show === !s.hidden) return;
      s.hidden = !show;
      // Plotly lays out hidden figures at a wrong width: redo it once shown.
      if (show && window.Plotly) s.querySelectorAll('.js-plotly-plot').forEach(function (p) {
        window.Plotly.Plots.resize(p);
      });
    });
    links.forEach(function (a) {
      if (current && a.getAttribute('href') === '#' + current.id) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    });
  }

  window.addEventListener('hashchange', function () { sync(); window.scrollTo(0, 0); });
  window.reportTabs = { sync: sync };
  sync();

  // Close the filter dropdowns when clicking elsewhere.
  document.addEventListener('click', function (e) {
    document.querySelectorAll('details.dropdown[open]').forEach(function (d) {
      if (!d.contains(e.target)) d.open = false;
    });
  });
})();
