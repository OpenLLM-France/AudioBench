// Inline onclick handlers of the tables: expand/collapse the per-dataset
// sub-columns, and switch a task summary between its Language and Sub-task views.

function toggleCols(btn, tblId, group) {
  var tbl = document.getElementById(tblId);
  var cells = tbl.querySelectorAll('[data-group="' + group + '"]');
  if (!cells.length) return;
  var show = cells[0].style.display !== 'table-cell';
  for (var i = 0; i < cells.length; i++)
    cells[i].style.display = show ? 'table-cell' : 'none';
  btn.textContent = show ? '−' : '+';
}

function toggleOvTask(btn, task) {
  var tbl = btn.closest('table');
  var cells = tbl.querySelectorAll('[data-task="' + task + '"]');
  if (!cells.length) return;
  var show = cells[0].style.display !== 'table-cell';
  for (var i = 0; i < cells.length; i++)
    cells[i].style.display = show ? 'table-cell' : 'none';
  btn.textContent = show ? '−' : '+';
}

function toggleSumView(id, view) {
  var langDiv = document.getElementById('sv-lang-' + id);
  var subDiv = document.getElementById('sv-sub-' + id);
  var bar = document.getElementById('tbar-' + id);
  if (!langDiv || !subDiv || !bar) return;
  var btns = bar.querySelectorAll('button');
  if (view === 'lang') {
    langDiv.style.display = '';
    subDiv.style.display = 'none';
    btns[0].classList.add('active');
    btns[1].classList.remove('active');
  } else {
    langDiv.style.display = 'none';
    subDiv.style.display = '';
    btns[0].classList.remove('active');
    btns[1].classList.add('active');
  }
}
