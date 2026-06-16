/* deck.js — presentation runtime for the bespoke slide framework.
   - Review mode (default): slides stacked, scrollable, scaled to fit width.
   - Present mode (press F or P): one slide at a time, scaled to fill any
     viewport, with slide crossfades and a per-slide content BUILD —
     each advance reveals the next fragment (bullet, card, diagram part);
     once a slide is fully built, the next advance moves to the next slide.
   - Navigation: arrow keys, Page Up/Down, Space (any presenter remote),
     swipe, and click-to-advance.
   - F = fullscreen + present · P = present · Esc = exit · B/. = blank.
   Print/PDF export is unaffected — @media print neutralises everything here. */
(function () {
  var deck = document.querySelector('.deck');
  if (!deck) return;
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var total = slides.length;
  var cur = 0, step = 0, presenting = false, built = false;

  /* ---- page numbers ---- */
  slides.forEach(function (s, i) {
    s.querySelectorAll('[data-pagenum]').forEach(function (el) {
      el.textContent = 'Page ' + (i + 1) + ' of ' + total;
    });
    s.querySelectorAll('[data-pageonly]').forEach(function (el) {
      el.textContent = (i + 1) + ' / ' + total;
    });
  });

  /* ---- on-screen furniture ---- */
  var progress = document.createElement('div');
  progress.className = 'deck-progress';
  document.body.appendChild(progress);

  var hint = document.createElement('div');
  hint.className = 'deck-hint';
  hint.innerHTML = 'Press <b>F</b> to present';
  hint.addEventListener('click', function () { enterPresent(true); });
  document.body.appendChild(hint);

  /* ---- responsive scaling ---- */
  function fitPresent() {
    var k = Math.min(window.innerWidth / 1280, window.innerHeight / 720);
    document.documentElement.style.setProperty('--scale', k);
  }
  function fitReview() {
    deck.style.zoom = Math.min(1, (window.innerWidth - 24) / 1280);
  }
  function onResize() { presenting ? fitPresent() : fitReview(); }

  /* ---- fullscreen helpers (promise-safe) ---- */
  function reqFs() {
    var el = document.documentElement;
    if (!el.requestFullscreen) return;
    var p = el.requestFullscreen();
    if (p && p.catch) p.catch(function () {});
  }
  function exitFs() {
    if (document.fullscreenElement && document.exitFullscreen) {
      var p = document.exitFullscreen();
      if (p && p.catch) p.catch(function () {});
    }
  }

  /* ---- build the per-slide fragment list (within-slide animation) ---- */
  function buildSlides() {
    if (built) return;
    built = true;
    /* containers whose children become individual fragments */
    var GRID = '.bullets,.grid2,.grid3,.kpi,.ei-zoo,.toc,.ei-flow,' +
               '.stat-row,.ei-modegrid,.cols,.fbox';
    slides.forEach(function (slide) {
      var frags = [];
      /* title, section-divider and closing slides build as a whole */
      if (!slide.matches('.slide--section, .slide--title, .slide--closing')) {
        var body = slide.querySelector('.slide-body');
        if (body) {
          Array.prototype.forEach.call(body.children, function (cc) {
            if (cc.matches('.slide-foot')) return;
            if (cc.children.length && cc.matches(GRID)) {
              Array.prototype.forEach.call(cc.children, function (g) { frags.push(g); });
            } else {
              frags.push(cc);
            }
          });
        }
      }
      frags.forEach(function (f) { f.classList.add('frag'); });
      slide._frags = frags;
    });
  }
  function stepCount() { return (slides[cur]._frags || []).length; }

  /* ---- render current state ----
     paint() snaps every fragment to its built state with NO transition,
     except the one at revealIdx, which gets .anim so it animates in.
     This keeps slide changes flash-free: content is already in place. */
  function paint(revealIdx) {
    slides.forEach(function (s, i) { s.classList.toggle('active', i === cur); });
    var fr = slides[cur]._frags || [];
    fr.forEach(function (f, j) {
      if (j === revealIdx) {
        f.classList.add('anim', 'shown');
      } else {
        f.classList.remove('anim');
        f.classList.toggle('shown', j < step);
      }
    });
    progress.style.width = ((cur + 1) / total * 100) + '%';
  }

  /* ---- navigation ---- */
  function next() {
    if (!presenting) { if (cur < total - 1) { cur++; scrollCur(); } return; }
    if (step < stepCount()) { step++; paint(step - 1); }   /* reveal next fragment */
    else if (cur < total - 1) { cur++; step = 0; paint(); } /* next slide, snapped */
  }
  function prev() {
    if (!presenting) { if (cur > 0) { cur--; scrollCur(); } return; }
    if (step > 0) { step--; paint(); }                      /* hide last fragment */
    else if (cur > 0) { cur--; step = stepCount(); paint(); } /* prev slide, fully built */
  }
  function go(n) {
    cur = Math.max(0, Math.min(total - 1, n));
    step = 0;
    paint();
    if (!presenting) scrollCur();
  }
  function scrollCur() {
    paint();
    slides[cur].scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  /* ---- mode switching ---- */
  /* re-enable transitions only after the mode switch has painted, so
     entering/leaving present mode snaps with no burst of the old layout */
  function unfreeze() {
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        document.body.classList.remove('no-anim');
      });
    });
  }
  function enterPresent(withFs) {
    buildSlides();
    presenting = true;
    step = 0;
    document.body.classList.add('no-anim', 'present');
    hint.style.display = 'none';
    deck.style.zoom = '1';
    fitPresent();
    paint();
    if (withFs) reqFs();
    unfreeze();
  }
  function exitPresent() {
    presenting = false;
    document.body.classList.add('no-anim');
    document.body.classList.remove('present', 'blanked');
    hint.style.display = '';
    exitFs();
    fitReview();
    paint();
    slides[cur].scrollIntoView({ block: 'center' });
    unfreeze();
  }

  /* ---- keyboard / clicker ---- */
  document.addEventListener('keydown', function (e) {
    var k = e.key;
    if (k === 'ArrowRight' || k === 'ArrowDown' || k === 'PageDown' || k === ' ') {
      next(); e.preventDefault();
    } else if (k === 'ArrowLeft' || k === 'ArrowUp' || k === 'PageUp') {
      prev(); e.preventDefault();
    } else if (k === 'Home') { go(0); e.preventDefault(); }
    else if (k === 'End') { go(total - 1); e.preventDefault(); }
    else if (k === 'f' || k === 'F') {
      if (!presenting) enterPresent(true);
      else if (document.fullscreenElement) exitFs();
      else reqFs();
    } else if (k === 'p' || k === 'P') {
      presenting ? exitPresent() : enterPresent(false);
    } else if (k === 'Escape') {
      if (presenting) exitPresent();
    } else if (k === 'b' || k === 'B' || k === '.') {
      if (presenting) document.body.classList.toggle('blanked');
    }
  });

  /* ---- touch (tablet / phone) ---- */
  var tx = 0, ty = 0;
  document.addEventListener('touchstart', function (e) {
    tx = e.touches[0].clientX; ty = e.touches[0].clientY;
  }, { passive: true });
  document.addEventListener('touchend', function (e) {
    var dx = e.changedTouches[0].clientX - tx, dy = e.changedTouches[0].clientY - ty;
    if (Math.abs(dx) > 55 && Math.abs(dx) > Math.abs(dy) * 1.6) {
      dx < 0 ? next() : prev();
    }
  }, { passive: true });

  /* ---- click to advance (present mode only) ---- */
  document.addEventListener('click', function (e) {
    if (!presenting || e.target.closest('a') || e.target.closest('.deck-hint')) return;
    (e.clientX < window.innerWidth * 0.22) ? prev() : next();
  });

  window.addEventListener('resize', onResize);
  document.addEventListener('fullscreenchange', function () {
    if (presenting) fitPresent();
  });

  /* ---- init: review mode ---- */
  fitReview();
  paint();
})();
