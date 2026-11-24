const worldsGrid = document.getElementById("worldsGrid");
const worldsSection = document.querySelector(".worlds-section");
const worldsRail = document.querySelector(".worlds-rail");
const prevBtn = document.querySelector(".world-nav-prev");
const nextBtn = document.querySelector(".world-nav-next");
const playerView = document.getElementById("worldPlayerView");
const playerVideo = document.getElementById("worldPlayerVideo");

const sourceCards = [...worldsGrid.querySelectorAll(".world-card")];
const AUTO_SCROLL_SPEED = 0.095;
const USER_SCROLL_GAIN = 1.55;
let segmentWidth = 0;
let isAdjustingScroll = false;
let isPlayerMode = false;
let isPointerInsideWorlds = false;
let isWorldsVisible = false;
let isTouchDevice = window.matchMedia("(hover: none), (pointer: coarse)").matches;
let lastAutoScrollTime = 0;
let outsideCloseGuardUntil = 0;

function getGap() {
  const style = getComputedStyle(worldsGrid);
  return Number.parseFloat(style.columnGap || style.gap || "0");
}

function getStep() {
  const firstCard = worldsGrid.querySelector(".world-card");
  if (!firstCard) return 260;
  return firstCard.getBoundingClientRect().width + getGap();
}

function buildLoopTrack() {
  const prependFragment = document.createDocumentFragment();
  const appendFragment = document.createDocumentFragment();

  sourceCards.forEach((card) => {
    appendFragment.appendChild(card.cloneNode(true));
  });
  sourceCards.forEach((card) => {
    prependFragment.appendChild(card.cloneNode(true));
  });

  worldsGrid.prepend(prependFragment);
  worldsGrid.append(appendFragment);
}

function refreshSegmentWidth() {
  segmentWidth = worldsGrid.scrollWidth / 3;
}

function setMiddleTrackPosition() {
  refreshSegmentWidth();
  worldsGrid.scrollLeft = segmentWidth;
}

function normalizeScrollLoop() {
  if (isPlayerMode || isAdjustingScroll || segmentWidth === 0) return;

  const left = worldsGrid.scrollLeft;
  const lowerBound = segmentWidth * 0.5;
  const upperBound = segmentWidth * 1.5;

  if (left < lowerBound || left > upperBound) {
    isAdjustingScroll = true;
    worldsGrid.scrollLeft = left < lowerBound ? left + segmentWidth : left - segmentWidth;
    requestAnimationFrame(() => {
      isAdjustingScroll = false;
    });
  }
}

function enterPlayerMode(videoSrc, posterSrc) {
  isPlayerMode = true;
  outsideCloseGuardUntil = performance.now() + 260;
  document.body.classList.add("world-player-mode");
  worldsGrid.classList.add("is-hidden");

  playerView.hidden = false;
  playerView.classList.add("is-visible");
  playerVideo.pause();
  playerVideo.src = videoSrc;
  playerVideo.poster = posterSrc || "";
  playerVideo.currentTime = 0;
  playerVideo.play().catch(() => {});
}

function exitPlayerMode() {
  if (!isPlayerMode) return;

  isPlayerMode = false;
  playerVideo.pause();
  playerVideo.removeAttribute("src");
  playerVideo.load();

  playerView.classList.remove("is-visible");
  playerView.hidden = true;

  document.body.classList.remove("world-player-mode");
  worldsGrid.classList.remove("is-hidden");
  normalizeScrollLoop();
}

function moveLane(direction) {
  if (isPlayerMode) return;
  const step = Math.max(260, Math.min(520, getStep() * 1.35));
  worldsGrid.scrollBy({ left: direction * step, behavior: "smooth" });
}

function shouldAutoScroll() {
  if (isPlayerMode) return false;
  if (isTouchDevice) return !isWorldsVisible;
  return !isPointerInsideWorlds;
}

function autoScrollTick(timestamp) {
  if (lastAutoScrollTime === 0) {
    lastAutoScrollTime = timestamp;
  }

  const deltaMs = Math.min(34, timestamp - lastAutoScrollTime);
  lastAutoScrollTime = timestamp;

  if (shouldAutoScroll()) {
    worldsGrid.scrollLeft += AUTO_SCROLL_SPEED * deltaMs;
    normalizeScrollLoop();
  }

  requestAnimationFrame(autoScrollTick);
}

worldsGrid.addEventListener("click", (event) => {
  if (isPlayerMode) return;

  const bubble = event.target.closest(".world-bubble");
  if (!bubble) return;

  const card = bubble.closest(".world-card");
  const source = card?.querySelector("video source");
  const image = card?.querySelector("img");
  if (!source?.src) return;

  enterPlayerMode(source.src, image?.src);
});

worldsGrid.addEventListener(
  "wheel",
  (event) => {
    if (isPlayerMode) return;
    const horizontalDelta =
      Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
    if (horizontalDelta === 0) return;
    event.preventDefault();
    worldsGrid.scrollLeft += horizontalDelta * USER_SCROLL_GAIN;
    normalizeScrollLoop();
  },
  { passive: false }
);

worldsGrid.addEventListener("scroll", normalizeScrollLoop);
prevBtn.addEventListener("click", () => moveLane(-1));
nextBtn.addEventListener("click", () => moveLane(1));
worldsRail.addEventListener("mouseenter", () => {
  isPointerInsideWorlds = true;
});
worldsRail.addEventListener("mouseleave", () => {
  isPointerInsideWorlds = false;
});
worldsRail.addEventListener("touchstart", () => {
  isPointerInsideWorlds = true;
});
worldsRail.addEventListener("touchend", () => {
  isPointerInsideWorlds = false;
});

playerVideo.addEventListener("ended", exitPlayerMode);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    exitPlayerMode();
  }
});

document.addEventListener("pointerdown", (event) => {
  if (!isPlayerMode) return;
  if (performance.now() < outsideCloseGuardUntil) return;
  if (!playerView.contains(event.target)) {
    exitPlayerMode();
  }
});

window.addEventListener("resize", () => {
  isTouchDevice = window.matchMedia("(hover: none), (pointer: coarse)").matches;
  if (isPlayerMode) return;
  setMiddleTrackPosition();
});

const visibilityObserver = new IntersectionObserver(
  (entries) => {
    const [entry] = entries;
    isWorldsVisible = entry.isIntersecting && entry.intersectionRatio > 0.25;
  },
  { threshold: [0, 0.25, 0.5, 0.75, 1] }
);

visibilityObserver.observe(worldsSection);

buildLoopTrack();
requestAnimationFrame(() => {
  setMiddleTrackPosition();
});
requestAnimationFrame(autoScrollTick);
