(() => {
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const entrances = new Set();
  reduced.addEventListener?.('change', (event) => {
    if (!event.matches) return;
    entrances.forEach((animation) => animation.cancel());
    entrances.clear();
  });
  const gallery = document.querySelector('.worlds-grid');
  document.querySelectorAll('[data-scroll]').forEach((button) => button.addEventListener('click', () => {
    const first = gallery.querySelector('.world-card');
    const distance = first.getBoundingClientRect().width + parseFloat(getComputedStyle(gallery).gap || 0);
    gallery.scrollBy({left: distance * Number(button.dataset.scroll), behavior: reduced.matches ? 'auto' : 'smooth'});
  }));
  const cards = [...document.querySelectorAll('.world-card')];
  cards.forEach((card) => {
    const video = card.querySelector('video');
    card.querySelector('summary').addEventListener('click', () => {
      if (!card.open) {
        cards.forEach((other) => {
          if (other !== card && other.open) {
            other.querySelector('video').pause();
            other.open = false;
          }
        });
        video.play().catch(() => {});
      }
    });
    card.addEventListener('toggle', () => { if (!card.open) video.pause(); });
  });
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries, self) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        if (!reduced.matches) {
          const animation = entry.target.animate([{transform:'translateY(12px)'},{transform:'translateY(0)'}], {duration:700,easing:'ease-out'});
          entrances.add(animation);
          animation.onfinish = () => entrances.delete(animation);
        }
        self.unobserve(entry.target);
      });
    }, {threshold:.08});
    document.querySelectorAll('.facts div, .feature-list details, .world-card, .resource-links a').forEach((item) => observer.observe(item));
  }
  if (!reduced.matches) {
    const gsapScript = document.createElement('script');
    gsapScript.src = 'https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/gsap.min.js';
    gsapScript.onload = () => {
      const triggerScript = document.createElement('script');
      triggerScript.src = 'https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/ScrollTrigger.min.js';
      triggerScript.onload = () => {
        if (reduced.matches || !window.gsap || !window.ScrollTrigger) return;
        gsap.registerPlugin(ScrollTrigger);
        gsap.matchMedia().add('(prefers-reduced-motion: no-preference)', () => {
          gsap.fromTo('.hero-media video', {scale: .96}, {scale: 1, ease: 'none', scrollTrigger: {trigger: '.hero-media', start: 'top bottom', end: 'bottom center', scrub: true}});
          gsap.fromTo('.overview > p:not(.fact-note)', {opacity: .6}, {opacity: 1, ease: 'none', scrollTrigger: {trigger: '.overview', start: 'top 90%', end: 'center 55%', scrub: true}});
        });
      };
      document.head.appendChild(triggerScript);
    };
    document.head.appendChild(gsapScript);
  }
})();
