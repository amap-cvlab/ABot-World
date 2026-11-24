(() => {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const Hls = window.Hls;

  document.querySelectorAll('video[data-hls-status]').forEach((video) => {
    const localFile = location.protocol === 'file:';
    const fileFallback = localFile ? video.querySelector('source[data-file-fallback]') : null;
    const source = fileFallback && video.canPlayType(fileFallback.type) ? fileFallback : video.querySelector('source');
    const status = document.getElementById(video.dataset.hlsStatus);
    const start = video.parentElement.querySelector('.hls-start');
    const startLabel = start.textContent;
    const sourceUrl = source.src;
    const autoplay = video.hasAttribute('data-hls-autoplay');
    const nativeSupport = video.canPlayType(source.type);
    // The OSS host rejects Hls.js requests from file origins; play those streams natively.
    const useHls = !localFile && Hls?.isSupported() && !(nativeSupport && 'ManagedMediaSource' in window);
    let unavailable = false;

    const showUnavailable = () => {
      unavailable = true;
      start.hidden = true;
      status.hidden = false;
    };

    if (!useHls && !nativeSupport) {
      if (localFile) status.textContent = 'Open the local HTTP preview to play this video.';
      showUnavailable();
      return;
    }

    video.controls = false;
    if (autoplay) {
      reducedMotion.addEventListener('change', (event) => {
        if (event.matches) video.pause();
      });
    }

    video.addEventListener('playing', () => {
      start.hidden = true;
      video.controls = true;
      status.hidden = true;
    });
    video.addEventListener('error', () => {
      if (video.error) showUnavailable();
    });

    const hls = useHls ? new Hls({
      capLevelToPlayerSize: true,
      // Enough for the 720p launch film when the browser has no network estimate.
      abrEwmaDefaultEstimate: 7_000_000,
      maxBufferLength: 12,
      maxMaxBufferLength: 24,
    }) : null;

    const startPlayback = () => {
      const restoreFocus = document.activeElement === start;
      start.disabled = true;
      start.textContent = 'Loading…';
      if (hls && !hls.url) {
        const connection = navigator.connection;
        if (connection?.saveData || ['slow-2g', '2g', '3g'].includes(connection?.effectiveType)
            || (connection?.downlink > 0 && connection.downlink < 2)) {
          hls.config.startLevel = 0;
        } else if (connection?.downlink > 0) {
          // downlink is reported in Mbps. Leave 10% headroom for the first request.
          hls.bandwidthEstimate = connection.downlink * 0.9 * 1_000_000;
        }
        hls.loadSource(sourceUrl);
      }

      video.play().then(() => {
        if (restoreFocus) video.focus({ preventScroll: true });
      }, () => {
        if (unavailable) return;
        start.textContent = startLabel;
        start.disabled = false;
        start.hidden = false;
        if (restoreFocus) start.focus({ preventScroll: true });
      });
    };

    const readyToStart = () => {
      if (autoplay && !reducedMotion.matches) startPlayback();
      else start.hidden = false;
    };

    start.addEventListener('click', startPlayback);
    if (hls) {
      hls.on(Hls.Events.ERROR, (_event, { fatal }) => {
        if (!fatal) return;
        showUnavailable();
        video.pause();
        hls.destroy();
      });
      // Finish binding MediaSource before offering play, and load the playlist from that same click.
      hls.once(Hls.Events.MEDIA_ATTACHED, readyToStart);
      hls.attachMedia(video);
    } else {
      video.src = sourceUrl;
      readyToStart();
    }
  });
})();
