(function initKeyHandler() {
    console.log(`[KeyHandler] launch(js=) executing, readyState=${document.readyState}`);

    const MAX_WAIT = 15000;
    let waited = 0;
    const POLL = 200;

    // 通过 data-key 属性查找按键元素（绕过 Gradio shadow DOM 限制）
    function findKeyElement(key) {
        // 1. 直接通过 data-key 属性查找
        let el = document.querySelector(`[data-key="${key}"]`);
        if (el) return el;

        // 2. 遍历所有 key-btn 元素，匹配文本内容
        const btns = document.querySelectorAll('.key-btn');
        for (const btn of btns) {
            if (btn.textContent.trim() === key) {
                return btn;
            }
        }

        // 3. 查找所有元素，匹配 data-key 属性（包括 shadow DOM）
        const allElements = document.querySelectorAll('*');
        for (const elem of allElements) {
            if (elem.getAttribute && elem.getAttribute('data-key') === key) {
                return elem;
            }
        }

        return null;
    }

    // 检查 HUD 是否存在
    function hasHUD() {
        const wasd = document.querySelector('[data-key-group="wasd"]') ||
                     document.querySelector('#abotworld-key-hud-wasd');
        const ijkl = document.querySelector('[data-key-group="ijkl"]') ||
                     document.querySelector('#abotworld-key-hud-ijkl');
        return !!(wasd && ijkl);
    }

    function tryInit() {
        const container = document.getElementById('key-state-input');
        const tb = container ? container.querySelector('textarea') : null;
        const hudReady = hasHUD();
        const hasW = !!findKeyElement('W');
        const hasI = !!findKeyElement('I');

        console.log(`[KeyHandler] tryInit waited=${waited}ms container=${!!container} tb=${!!tb} hud=${hudReady} W=${hasW} I=${hasI}`);

        if (!tb || !hudReady || !hasW || !hasI) {
            waited += POLL;
            if (waited < MAX_WAIT) {
                setTimeout(tryInit, POLL);
            } else {
                console.error(`[KeyHandler] DOM not ready after ${MAX_WAIT}ms, giving up.`);
                // 即使 HUD 没找到，也尝试设置键盘监听
                if (tb) {
                    console.log('[KeyHandler] Forcing setup without HUD...');
                    setup(tb);
                }
            }
            return;
        }
        console.log(`[KeyHandler] DOM ready after ${waited}ms, setting up...`);
        setup(tb);
    }

    function setup(tb) {
        // 统一使用大写，与后端 pipeline.set_act() 接口保持一致
        const KEYS = new Set(['W', 'A', 'S', 'D', 'I', 'J', 'K', 'L', 'Ctrl', 'Shift', 'Space', 'Alt']);
        const BODY_KEYS = {CONTROL: 'Ctrl', SHIFT: 'Shift', ' ': 'Space', ALT: 'Alt'};
        const MOUSE_KEYS = {0: 'MouseLeft', 2: 'MouseRight'};

        // 方向键 → IJKL 映射
        const ARROW_MAP = {
            'ARROWUP': 'I',
            'ARROWDOWN': 'K',
            'ARROWLEFT': 'J',
            'ARROWRIGHT': 'L',
        };

        // 冲突组：每组内的键互斥，同时按住时保留最后按下的
        const CONFLICT_GROUPS = [
            ['W', 'S'],  // 前/后
            ['A', 'D'],  // 左/右
            ['I', 'K'],  // 视角上/下
            ['J', 'L'],  // 视角左/右
        ];

        const pressedKeys = new Set();    // 当前物理按住的键
        const activatedKeys = new Set();  // 自上次发送起曾激活的键（未消费）
        const pressTime = {};             // 记录每个键最后一次按下的时间戳（ms）
        let throttleTimer = null;

        // Use native setter to bypass Svelte's reactive DOM interception
        const nativeSetter = Object.getOwnPropertyDescriptor(
            window.HTMLTextAreaElement.prototype, 'value'
        ).set;

        console.log(`[KeyHandler] nativeSetter type=${typeof nativeSetter}`);

        // 对 pressedKeys 做冲突消解：每个冲突组内只保留最后按下的键
        const resolveConflicts = (keys) => {
            const resolved = new Set(keys);
            CONFLICT_GROUPS.forEach((group) => {
                const held = group.filter((k) => resolved.has(k));
                if (held.length < 2) return;
                // 找出按下时间最晚的键，移除其余
                const winner = held.reduce((a, b) => (pressTime[a] || 0) >= (pressTime[b] || 0) ? a : b);
                held.forEach((k) => {
                    if (k !== winner) resolved.delete(k);
                });
            });
            return resolved;
        };

        const updateHUD = () => {
            KEYS.forEach((k) => {
                const el = findKeyElement(k);
                if (!el) return;
                if (pressedKeys.has(k)) {
                    el.classList.remove('releasing');
                    el.classList.add('active');
                } else {
                    if (el.classList.contains('active')) {
                        el.classList.remove('active');
                        el.classList.add('releasing');
                        setTimeout(() => { el.classList.remove('releasing'); }, 200);
                    }
                }
            });
        };

        const sendKeys = () => {
            // 冲突消解：pressed 和 activated 都要过滤
            const resolvedPressed = resolveConflicts(pressedKeys);
            const resolvedActivated = resolveConflicts(activatedKeys);

            const payload = JSON.stringify({
                pressed: Array.from(resolvedPressed),
                activated: Array.from(resolvedActivated)
            });
            console.log(`[KeyHandler] sendKeys payload=${payload}${resolvedPressed.size < pressedKeys.size ? ` (conflict resolved: raw pressed=${JSON.stringify(Array.from(pressedKeys))})` : ''}`);
            activatedKeys.clear();
            nativeSetter.call(tb, payload);
            tb.dispatchEvent(new Event('input', { bubbles: true }));
        };

        const triggerSend = () => {
            if (throttleTimer) return;
            throttleTimer = setTimeout(() => {
                throttleTimer = null;
                sendKeys();
            }, 50);
        };

        const clearAllTrackedKeys = (reason) => {
            if (pressedKeys.size === 0 && activatedKeys.size === 0) return;
            console.log(`[KeyHandler] clearing tracked keys, reason=${reason}`);
            pressedKeys.clear();
            activatedKeys.clear();
            updateHUD();
            triggerSend();
        };

        /** 焦点在此类元素上时不劫持 WASD/IJKL；#key-state-input 内例外（隐藏同步框） */
        function isGameKeysBlockedTarget(el) {
            if (!el || el.nodeType !== Node.ELEMENT_NODE) return false;
            if (el.closest && el.closest('#key-state-input')) return false;
            if (el.isContentEditable) return true;
            const tag = el.tagName;
            if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
            if (tag === 'INPUT') {
                const type = (el.type || '').toLowerCase();
                const nonText = new Set([
                    'button', 'checkbox', 'radio', 'submit', 'reset',
                    'file', 'hidden', 'color', 'range', 'image',
                ]);
                return !nonText.has(type);
            }
            return false;
        }

        document.addEventListener('focusin', (e) => {
            if (!isGameKeysBlockedTarget(e.target)) return;
            if (pressedKeys.size === 0) return;
            pressedKeys.clear();
            activatedKeys.clear();
            updateHUD();
            triggerSend();
        });

        document.addEventListener('keydown', (e) => {
            if (isGameKeysBlockedTarget(e.target)) return;
            // toUpperCase 消除 CapsLock / Shift 导致的大小写歧义；方向键映射到 IJKL
            const raw = e.key.toUpperCase();
            const key = ARROW_MAP[raw] || BODY_KEYS[raw] || raw;
            const tracked = KEYS.has(key);
            console.log(`[KeyHandler] keydown key=${key} repeat=${e.repeat} tracked=${tracked}`);
            if (e.metaKey || (e.ctrlKey && e.altKey && raw === 'D')) {
                if (tracked) {
                    pressedKeys.delete(key);
                    activatedKeys.delete(key);
                    updateHUD();
                    triggerSend();
                }
                return;
            }
            if (!tracked || e.repeat) return;
            e.preventDefault();
            pressTime[key] = Date.now();
            pressedKeys.add(key);
            activatedKeys.add(key);
            updateHUD();
            triggerSend();
        });

        document.addEventListener('keyup', (e) => {
            if (isGameKeysBlockedTarget(e.target)) return;
            const raw = e.key.toUpperCase();
            const key = ARROW_MAP[raw] || BODY_KEYS[raw] || raw;
            if (!KEYS.has(key)) return;
            console.log(`[KeyHandler] keyup key=${key}`);
            pressedKeys.delete(key);
            // pressTime 保留，keyup 后对方键仍然有效（松开后不影响胜者判断）
            updateHUD();
            triggerSend();
        });

        document.addEventListener('mousedown', (e) => {
            const key = MOUSE_KEYS[e.button];
            if (!key || !e.target.closest?.('.fe-video-fixed')) return;
            if (pressedKeys.has(key)) return;
            pressedKeys.add(key);
            activatedKeys.add(key);
            triggerSend();
        });
        document.addEventListener('mouseup', (e) => {
            if (pressedKeys.delete(MOUSE_KEYS[e.button])) triggerSend();
        });
        document.addEventListener('contextmenu', (e) => {
            if (e.target.closest?.('.fe-video-fixed')) e.preventDefault();
        });

        // 浏览器失焦/切后台时，保证 HUD 与状态不残留
        window.addEventListener('blur', () => clearAllTrackedKeys('window_blur'));
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState !== 'visible') clearAllTrackedKeys('visibility_hidden');
        });

        console.log('[KeyHandler] movement, camera and action listeners attached');
    }

    setTimeout(tryInit, POLL);
})();

/** Debug 区：Ctrl+Alt+D 切换 body.fe-debug-visible */
(function initDebugPanelToggle() {
    document.addEventListener(
        'keydown',
        (e) => {
            if (!e.ctrlKey || !e.altKey || e.metaKey) return;
            const k = e.key;
            if (k !== 'd' && k !== 'D') return;
            e.preventDefault();
            document.body.classList.toggle('fe-debug-visible');
        },
        true
    );
})();

/** 「探索平行宇宙」画廊：自定义横向滚动进度条
    浏览器原生滚动条在 Ubuntu overlay 模式下自动隐藏，无法通过 CSS 强制永久显示。
    因此用 JS 创建独立的滚动指示器，完全绕过原生滚动条 API。 */
(function initGalleryScrollbar() {
    const GALLERY_SELECTOR = '.fe-ref-gallery-hscroll';
    const SCROLL_CONTAINER_SELECTOR = '.grid-wrap';
    const BAR_ID = 'fe-gallery-scrollbar';
    const POLL_INTERVAL = 500;
    const MAX_WAIT = 20000;
    let waited = 0;

    function createScrollbar(gallery, scrollContainer) {
        // 防止重复创建
        if (document.getElementById(BAR_ID)) return;

        // 创建进度条容器
        const bar = document.createElement('div');
        bar.id = BAR_ID;
        bar.className = 'fe-gallery-scrollbar';

        // 创建轨道
        const track = document.createElement('div');
        track.className = 'fe-gallery-sb-track';

        // 创建滑块
        const thumb = document.createElement('div');
        thumb.className = 'fe-gallery-sb-thumb';

        track.appendChild(thumb);
        bar.appendChild(track);

        // 插入到 gallery 容器末尾
        gallery.appendChild(bar);

        // ── 更新滑块位置和宽度 ──
        function updateThumb() {
            const { scrollLeft, scrollWidth, clientWidth } = scrollContainer;
            const scrollable = scrollWidth - clientWidth;
            if (scrollable <= 0) {
                bar.style.display = 'none';
                return;
            }
            bar.style.display = '';
            const ratio = scrollLeft / scrollable;
            const thumbWidthRatio = clientWidth / scrollWidth;
            const thumbWidth = Math.max(thumbWidthRatio * 100, 15); // 最小 15%
            const thumbLeft = ratio * (100 - thumbWidth);
            thumb.style.width = thumbWidth + '%';
            thumb.style.transform = `translateX(${thumbLeft / thumbWidth * 100}%)`;
        }

        // ── 监听滚动事件 ──
        scrollContainer.addEventListener('scroll', updateThumb, { passive: true });

        // ── 监听内容变化（画廊图片动态加载） ──
        const gridContainer = scrollContainer.querySelector('.grid-container');
        if (gridContainer) {
            const observer = new MutationObserver(() => {
                requestAnimationFrame(updateThumb);
            });
            observer.observe(gridContainer, { childList: true, subtree: true });
        }

        // ── 监听容器尺寸变化（窗口调整、Gradio 布局重排） ──
        if (typeof ResizeObserver !== 'undefined') {
            const resizeObs = new ResizeObserver(() => requestAnimationFrame(updateThumb));
            resizeObs.observe(scrollContainer);
        }

        // ── 监听容器尺寸变化（窗口调整、Gradio 布局重排） ──
        if (typeof ResizeObserver !== 'undefined') {
            const resizeObs = new ResizeObserver(() => requestAnimationFrame(updateThumb));
            resizeObs.observe(scrollContainer);
        }

        // ── 点击轨道跳转 ──
        track.addEventListener('mousedown', (e) => {
            if (e.target === thumb) return; // 拖拽由 thumb 处理
            const rect = track.getBoundingClientRect();
            const clickRatio = (e.clientX - rect.left) / rect.width;
            const { scrollWidth, clientWidth } = scrollContainer;
            const scrollable = scrollWidth - clientWidth;
            scrollContainer.scrollTo({
                left: clickRatio * scrollable,
                behavior: 'smooth'
            });
        });

        // ── 拖拽滑块 ──
        let dragging = false;
        let dragStartX = 0;
        let dragStartScrollLeft = 0;

        thumb.addEventListener('mousedown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dragging = true;
            dragStartX = e.clientX;
            dragStartScrollLeft = scrollContainer.scrollLeft;
            document.body.style.cursor = 'grabbing';
            document.body.style.userSelect = 'none';
        });

        document.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            const dx = e.clientX - dragStartX;
            const { scrollWidth, clientWidth } = scrollContainer;
            const trackWidth = track.getBoundingClientRect().width;
            // 拖拽距离与滚动距离的映射
            const scrollRatio = dx / trackWidth;
            const scrollable = scrollWidth - clientWidth;
            scrollContainer.scrollLeft = dragStartScrollLeft + scrollRatio * scrollable;
        });

        document.addEventListener('mouseup', () => {
            if (!dragging) return;
            dragging = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        });

        // 初始更新
        updateThumb();
        // 延迟再次更新（等待图片加载导致布局变化）
        setTimeout(updateThumb, 1000);
        setTimeout(updateThumb, 3000);

        console.log('[GalleryScrollbar] custom scrollbar created');
    }

    function tryInit() {
        const gallery = document.querySelector(GALLERY_SELECTOR);
        if (!gallery) {
            waited += POLL_INTERVAL;
            if (waited < MAX_WAIT) {
                setTimeout(tryInit, POLL_INTERVAL);
            } else {
                console.warn('[GalleryScrollbar] gallery element not found after waiting');
            }
            return;
        }
        const scrollContainer = gallery.querySelector(SCROLL_CONTAINER_SELECTOR);
        if (!scrollContainer) {
            waited += POLL_INTERVAL;
            if (waited < MAX_WAIT) {
                setTimeout(tryInit, POLL_INTERVAL);
            } else {
                console.warn('[GalleryScrollbar] .grid-wrap not found after waiting');
            }
            return;
        }
        // 检查是否有可滚动内容
        const scrollable = scrollContainer.scrollWidth - scrollContainer.clientWidth;
        if (scrollable <= 0) {
            // 内容未溢出，可能还在加载，继续等待
            waited += POLL_INTERVAL;
            if (waited < MAX_WAIT) {
                setTimeout(tryInit, POLL_INTERVAL);
            }
            return;
        }
        createScrollbar(gallery, scrollContainer);
    }

    // 等待 DOM 就绪后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => setTimeout(tryInit, POLL_INTERVAL));
    } else {
        setTimeout(tryInit, POLL_INTERVAL);
    }
})();
