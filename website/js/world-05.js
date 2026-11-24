(() => {
  const panels = [...document.querySelectorAll('[data-model]')];
  // Both descriptions remain visible when JavaScript is unavailable.
  panels.slice(1).forEach((panel) => {
    const button = panel.querySelector('.model-toggle');
    const content = document.getElementById(button.getAttribute('aria-controls'));
    panel.classList.remove('is-open');
    button.setAttribute('aria-expanded', 'false');
    button.querySelector('.model-symbol').textContent = '+';
    content.hidden = true;
  });
  panels.forEach((panel) => {
    const toggle = panel.querySelector('.model-toggle');
    toggle.addEventListener('click', () => {
      panels.forEach((item) => {
        const button = item.querySelector('.model-toggle');
        const content = document.getElementById(button.getAttribute('aria-controls'));
        const isOpen = item === panel;
        item.classList.toggle('is-open', isOpen);
        button.setAttribute('aria-expanded', String(isOpen));
        button.querySelector('.model-symbol').textContent = isOpen ? '−' : '+';
        content.hidden = !isOpen;
      });
    });
  });

  // Essential interaction stays independent of the optional animation library.
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = src;
      script.async = true;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }
  loadScript('https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/gsap.min.js')
    .then(() => loadScript('https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/ScrollTrigger.min.js'))
    .then(() => {
      if (!window.gsap || !window.ScrollTrigger) return;
      gsap.registerPlugin(ScrollTrigger);
      gsap.matchMedia().add('(prefers-reduced-motion: no-preference)', () => {
        gsap.utils.toArray('.feature-media img').forEach((image) => {
          gsap.fromTo(image, { scale: 0.96, opacity: 0.86 }, {
            scale: 1, opacity: 1, ease: 'none', scrollTrigger: { trigger: image, start: 'top 90%', end: 'top 48%', scrub: true }
          });
        });
        gsap.utils.toArray('.intro-title, .intro-content, .feature-text, .research-heading, .research-work, .archive-copy').forEach((element) => {
          gsap.fromTo(element, { y: 14 }, {
            y: 0, ease: 'none', scrollTrigger: { trigger: element, start: 'top 95%', end: 'top 75%', scrub: true }
          });
        });
      });
    })
    .catch(() => {});
})();
