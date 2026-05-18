window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    function wireCopyButton(buttonSelector, sourceElementId, copiedLabel) {
        $(buttonSelector).on('click', function() {
            var el = document.getElementById(sourceElementId);
            if (!el) return;
            var text = el.textContent || '';
            var btn = this;
            navigator.clipboard.writeText(text).then(function() {
                var $btn = $(btn);
                var orig = $btn.html();
                $btn.html('<span class="icon is-small"><i class="fas fa-check"></i></span><span>' + copiedLabel + '</span>');
                setTimeout(function() { $btn.html(orig); }, 2000);
            }).catch(function() {
                window.prompt('Copy citation:', text);
            });
        });
    }

    wireCopyButton('#copy-bibtex', 'bibtex-code', 'Copied');
    wireCopyButton('#copy-apa', 'apa-citation-text', 'Copied');

    document.querySelectorAll('.code-window').forEach(function (win) {
        var btn = win.querySelector('.code-copy-btn');
        var code = win.querySelector('code');
        if (!btn || !code) return;
        btn.addEventListener('click', function () {
            var text = code.textContent || '';
            var label = btn.querySelector('span');
            navigator.clipboard.writeText(text).then(function () {
                btn.classList.add('is-copied');
                if (label) label.textContent = 'Copied';
                setTimeout(function () {
                    btn.classList.remove('is-copied');
                    if (label) label.textContent = 'Copy';
                }, 2000);
            }).catch(function () {
                window.prompt('Copy code:', text);
            });
        });
    });

    (function sectionNavStickyTop() {
        var nav = document.querySelector('.section-nav');
        if (!nav) return;

        function updateNavTop() {
            if (window.innerWidth < 1100) {
                nav.style.top = '';
                return;
            }
            var h = nav.offsetHeight;
            nav.style.top = 'calc(50vh - ' + (h / 2) + 'px)';
        }

        updateNavTop();
        window.addEventListener('resize', updateNavTop, { passive: true });
        if (typeof ResizeObserver !== 'undefined') {
            var ro = new ResizeObserver(updateNavTop);
            ro.observe(nav);
        }
    })();

    (function sectionNavHighlight() {
        var nav = document.querySelector('.section-nav');
        if (!nav) return;
        var links = nav.querySelectorAll('a[href^="#"]');
        if (!links.length) return;
        var sections = [];
        links.forEach(function (link) {
            var id = link.getAttribute('href').slice(1);
            var el = document.getElementById(id);
            if (el) sections.push({ id: id, el: el, link: link });
        });
        if (!sections.length) return;

        function setActive(id) {
            links.forEach(function (link) {
                link.classList.toggle('is-active', link.getAttribute('href') === '#' + id);
            });
        }

        function updateActiveFromScroll() {
            var marker = window.scrollY + Math.min(window.innerHeight * 0.28, 180);
            var current = sections[0].id;
            sections.forEach(function (s) {
                if (s.el.offsetTop <= marker) current = s.id;
            });
            var last = sections[sections.length - 1];
            if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) {
                current = last.id;
            }
            setActive(current);
        }

        var scrollTimer;
        window.addEventListener('scroll', function () {
            if (scrollTimer) cancelAnimationFrame(scrollTimer);
            scrollTimer = requestAnimationFrame(updateActiveFromScroll);
        }, { passive: true });
        window.addEventListener('resize', updateActiveFromScroll, { passive: true });
        updateActiveFromScroll();

        links.forEach(function (link) {
            link.addEventListener('click', function () {
                var id = link.getAttribute('href').slice(1);
                setActive(id);
            });
        });
    })();

    (function imageLightbox() {
        var modal = document.getElementById('image-lightbox');
        if (!modal) return;

        var backdrop = modal.querySelector('.image-lightbox-backdrop');
        var viewport = modal.querySelector('.image-lightbox-viewport');
        var img = modal.querySelector('.image-lightbox-image');
        var zoomLabel = modal.querySelector('.image-lightbox-zoom-label');
        var btnIn = modal.querySelector('.image-lightbox-zoom-in');
        var btnOut = modal.querySelector('.image-lightbox-zoom-out');
        var btnReset = modal.querySelector('.image-lightbox-reset');
        var btnClose = modal.querySelector('.image-lightbox-close-btn');

        var userScale = 1;
        var translateX = 0;
        var translateY = 0;
        var dragging = false;
        var dragStartX = 0;
        var dragStartY = 0;
        var dragOriginX = 0;
        var dragOriginY = 0;
        var isPinching = false;
        var pinchStartDistance = 0;
        var pinchStartScale = 1;
        var pinchStartTranslateX = 0;
        var pinchStartTranslateY = 0;
        var pinchStartCenterX = 0;
        var pinchStartCenterY = 0;

        function isDefaultView() {
            return userScale === 1 && translateX === 0 && translateY === 0;
        }

        function lockDisplayedSizeBeforeZoom() {
            if (img.style.maxWidth !== '100%') return;
            var rect = img.getBoundingClientRect();
            img.style.width = rect.width + 'px';
            img.style.height = rect.height + 'px';
            img.style.maxWidth = 'none';
            img.style.maxHeight = 'none';
        }

        function updateTransform() {
            if (isDefaultView()) {
                img.style.transform = '';
                img.style.width = '';
                img.style.height = '';
                img.style.maxWidth = '100%';
                img.style.maxHeight = '100%';
            } else {
                img.style.transform = 'translate(' + translateX + 'px, ' + translateY + 'px) scale(' + userScale + ')';
            }
            if (zoomLabel) zoomLabel.textContent = Math.round(userScale * 100) + '%';
        }

        function resetView() {
            userScale = 1;
            translateX = 0;
            translateY = 0;
            updateTransform();
        }

        var objectUrl = null;

        function revokeObjectUrl() {
            if (objectUrl) {
                URL.revokeObjectURL(objectUrl);
                objectUrl = null;
            }
        }

        function svgElementToObjectUrl(svgEl) {
            var clone = svgEl.cloneNode(true);
            if (!clone.getAttribute('xmlns')) {
                clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
            }
            var svgString = new XMLSerializer().serializeToString(clone);
            return URL.createObjectURL(new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' }));
        }

        function openLightbox(sourceEl) {
            revokeObjectUrl();
            var src;
            var alt = sourceEl.getAttribute('alt') || sourceEl.getAttribute('aria-label') || 'figure';
            if (sourceEl.tagName === 'IMG') {
                src = sourceEl.currentSrc || sourceEl.src;
            } else if (sourceEl.tagName.toLowerCase() === 'svg') {
                objectUrl = svgElementToObjectUrl(sourceEl);
                src = objectUrl;
            } else {
                return;
            }
            img.src = src;
            img.alt = alt;
            userScale = 1;
            translateX = 0;
            translateY = 0;
            modal.hidden = false;
            modal.classList.add('is-open');
            document.body.classList.add('is-lightbox-open');
            document.body.style.overflow = 'hidden';
            img.onload = function () {
                img.onload = null;
                resetView();
            };
            if (img.complete) {
                img.onload = null;
                resetView();
            }
        }

        function closeLightbox() {
            endPinch();
            dragging = false;
            modal.classList.remove('is-open');
            modal.classList.remove('is-dragging');
            modal.hidden = true;
            document.body.classList.remove('is-lightbox-open');
            document.body.style.overflow = '';
            img.removeAttribute('src');
            revokeObjectUrl();
        }

        function zoomBy(delta) {
            var next = Math.min(4, Math.max(0.5, userScale + delta));
            if (userScale === 1 && next !== 1) lockDisplayedSizeBeforeZoom();
            userScale = next;
            updateTransform();
        }

        function touchDistance(touches) {
            var dx = touches[0].clientX - touches[1].clientX;
            var dy = touches[0].clientY - touches[1].clientY;
            return Math.hypot(dx, dy);
        }

        function touchCenter(touches) {
            return {
                x: (touches[0].clientX + touches[1].clientX) / 2,
                y: (touches[0].clientY + touches[1].clientY) / 2
            };
        }

        function beginPinch(touches) {
            isPinching = true;
            dragging = false;
            viewport.classList.remove('is-dragging');
            modal.classList.remove('is-dragging');
            pinchStartDistance = touchDistance(touches);
            if (pinchStartDistance < 1) pinchStartDistance = 1;
            pinchStartScale = userScale;
            pinchStartTranslateX = translateX;
            pinchStartTranslateY = translateY;
            var center = touchCenter(touches);
            pinchStartCenterX = center.x;
            pinchStartCenterY = center.y;
            if (userScale === 1) lockDisplayedSizeBeforeZoom();
        }

        function updatePinch(touches) {
            if (pinchStartDistance < 1) return;
            var distance = touchDistance(touches);
            userScale = Math.min(4, Math.max(0.5, pinchStartScale * (distance / pinchStartDistance)));
            var center = touchCenter(touches);
            translateX = pinchStartTranslateX + (center.x - pinchStartCenterX);
            translateY = pinchStartTranslateY + (center.y - pinchStartCenterY);
            updateTransform();
        }

        function endPinch() {
            isPinching = false;
        }

        window.addEventListener('resize', function () {
            if (!modal.classList.contains('is-open')) return;
            if (userScale === 1 && translateX === 0 && translateY === 0) resetView();
        }, { passive: true });

        function wireFigureZoom(sourceEl) {
            if (sourceEl.closest('.figure-zoom-trigger')) return;

            var label = sourceEl.getAttribute('alt') || sourceEl.getAttribute('aria-label') || 'figure';
            var trigger = document.createElement('button');
            trigger.type = 'button';
            trigger.className = 'figure-zoom-trigger';
            trigger.setAttribute('aria-label', 'View larger: ' + label);

            var hint = document.createElement('span');
            hint.className = 'figure-zoom-hint';
            hint.setAttribute('aria-hidden', 'true');
            hint.innerHTML = '<i class="fas fa-search-plus"></i>';

            sourceEl.parentNode.insertBefore(trigger, sourceEl);
            trigger.appendChild(sourceEl);
            trigger.appendChild(hint);

            trigger.addEventListener('click', function () {
                openLightbox(sourceEl);
            });
        }

        document.querySelectorAll('.figure-container img').forEach(wireFigureZoom);
        document.querySelectorAll('.figure-container > svg').forEach(wireFigureZoom);

        if (backdrop) backdrop.addEventListener('click', closeLightbox);
        if (btnClose) btnClose.addEventListener('click', closeLightbox);
        if (btnIn) btnIn.addEventListener('click', function () { zoomBy(0.2); });
        if (btnOut) btnOut.addEventListener('click', function () { zoomBy(-0.2); });
        if (btnReset) btnReset.addEventListener('click', resetView);

        modal.addEventListener('click', function (ev) {
            if (ev.target === modal || ev.target === backdrop) {
                closeLightbox();
            }
        });

        img.addEventListener('click', function (ev) {
            ev.stopPropagation();
        });

        modal.addEventListener('wheel', function (ev) {
            if (!modal.classList.contains('is-open')) return;
            ev.preventDefault();
            zoomBy(ev.deltaY < 0 ? 0.12 : -0.12);
        }, { passive: false });

        img.addEventListener('touchstart', function (ev) {
            if (!modal.classList.contains('is-open')) return;
            if (ev.touches.length === 2) {
                ev.preventDefault();
                beginPinch(ev.touches);
            }
        }, { passive: false });

        img.addEventListener('touchmove', function (ev) {
            if (!modal.classList.contains('is-open')) return;
            if (!isPinching || ev.touches.length !== 2) return;
            ev.preventDefault();
            updatePinch(ev.touches);
        }, { passive: false });

        img.addEventListener('touchend', function (ev) {
            if (ev.touches.length < 2) endPinch();
        });

        img.addEventListener('touchcancel', endPinch);

        img.addEventListener('pointerdown', function (ev) {
            if (ev.button !== 0) return;
            if (isDefaultView() || isPinching) return;
            dragging = true;
            viewport.classList.add('is-dragging');
            modal.classList.add('is-dragging');
            dragStartX = ev.clientX;
            dragStartY = ev.clientY;
            dragOriginX = translateX;
            dragOriginY = translateY;
            img.setPointerCapture(ev.pointerId);
        });

        img.addEventListener('pointermove', function (ev) {
            if (!dragging || isPinching) return;
            translateX = dragOriginX + (ev.clientX - dragStartX);
            translateY = dragOriginY + (ev.clientY - dragStartY);
            updateTransform();
        });

        function endDrag(ev) {
            if (!dragging) return;
            dragging = false;
            viewport.classList.remove('is-dragging');
            modal.classList.remove('is-dragging');
            try { img.releasePointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
        }
        img.addEventListener('pointerup', endDrag);
        img.addEventListener('pointercancel', endDrag);

        document.addEventListener('keydown', function (ev) {
            if (!modal.classList.contains('is-open')) return;
            if (ev.key === 'Escape') closeLightbox();
            if (ev.key === '+' || ev.key === '=') zoomBy(0.2);
            if (ev.key === '-') zoomBy(-0.2);
        });
    })();

    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

    if (document.querySelectorAll('.carousel').length > 0 && typeof bulmaCarousel !== 'undefined') {
        var carousels = bulmaCarousel.attach('.carousel', options);
        for(var i = 0; i < carousels.length; i++) {
            carousels[i].on('before:show', function(state) {
                console.log(state);
            });
        }
    }

    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    var interpSlider = $('#interpolation-slider');
    if (interpSlider.length) {
        interpSlider.on('input', function(event) {
          setInterpolationImage(this.value);
        });
        setInterpolationImage(0);
        interpSlider.prop('max', NUM_INTERP_FRAMES - 1);
    }

    if (typeof bulmaSlider !== 'undefined') {
        bulmaSlider.attach();
    }

    /* Desiderata matrix: fixed tooltips (escape overflow, no gap under inline-flex badge) */
    (function desiderataMatrixTooltips() {
        var wrap = document.querySelector('.desiderata-matrix-wrap');
        if (!wrap) return;

        var scrollEl = wrap.querySelector('.desiderata-matrix-scroll');
        var activeTip = null;
        var hideTimer = null;
        var margin = 10;
        var maxWpx = 304;

        function clearHideTimer() {
            if (hideTimer) {
                clearTimeout(hideTimer);
                hideTimer = null;
            }
        }

        function scheduleClose(tip) {
            clearHideTimer();
            hideTimer = setTimeout(function () {
                closeTip(tip);
            }, 200);
        }

        function layoutBubble(tip, bubble) {
            var trigger = tip.querySelector('.d-cell');
            if (!trigger || !bubble) return;

            var vw = window.innerWidth;
            var vh = window.innerHeight;
            var r = trigger.getBoundingClientRect();
            var capW = Math.min(maxWpx, vw - 2 * margin);

            bubble.style.position = 'fixed';
            bubble.style.transform = 'none';
            bubble.style.width = capW + 'px';
            bubble.style.left = '-9999px';
            bubble.style.top = '0';
            bubble.style.visibility = 'hidden';
            bubble.style.opacity = '0';

            var br = bubble.getBoundingClientRect();
            var bw = br.width;
            var bh = br.height;

            var cx = r.left + r.width / 2;
            var left = cx - bw / 2;
            if (left < margin) left = margin;
            if (left + bw > vw - margin) left = Math.max(margin, vw - margin - bw);

            var spaceBelow = vh - r.bottom - margin;
            var spaceAbove = r.top - margin;
            var top;
            if (spaceBelow >= bh + margin || spaceBelow >= spaceAbove) {
                top = r.bottom + margin;
                if (top + bh > vh - margin) top = Math.max(margin, vh - margin - bh);
            } else {
                top = r.top - margin - bh;
                if (top < margin) top = margin;
            }

            bubble.style.left = Math.round(left) + 'px';
            bubble.style.top = Math.round(top) + 'px';
            bubble.style.width = '';
            bubble.style.visibility = '';
            bubble.style.opacity = '';
        }

        function closeTip(tip) {
            if (!tip) return;
            clearHideTimer();
            var bubble = tip._dBubble;
            if (bubble) {
                bubble.classList.remove('is-open');
            }
            if (bubble && bubble._homeParent) {
                bubble._homeParent.appendChild(bubble);
                bubble.style.left = '';
                bubble.style.top = '';
                bubble.style.width = '';
                bubble.style.position = '';
                bubble.style.transform = '';
                bubble.style.visibility = '';
                bubble.style.opacity = '';
            }
            if (activeTip === tip) activeTip = null;
        }

        function openTip(tip) {
            clearHideTimer();
            if (activeTip && activeTip !== tip) closeTip(activeTip);

            var bubble = tip._dBubble || tip.querySelector('.d-tip-bubble');
            if (!bubble) return;
            tip._dBubble = bubble;
            if (!bubble._homeParent) bubble._homeParent = tip;

            document.body.appendChild(bubble);
            activeTip = tip;
            layoutBubble(tip, bubble);
            requestAnimationFrame(function () {
                bubble.classList.add('is-open');
            });
        }

        wrap.querySelectorAll('.d-tip').forEach(function (tip) {
            var bubble = tip.querySelector('.d-tip-bubble');
            if (bubble && !bubble._wired) {
                bubble._wired = true;
                bubble.addEventListener('mouseenter', clearHideTimer);
                bubble.addEventListener('mouseleave', function () {
                    scheduleClose(tip);
                });
            }

            tip.addEventListener('mouseenter', function () {
                openTip(tip);
            });
            tip.addEventListener('mouseleave', function () {
                scheduleClose(tip);
            });
            tip.addEventListener('focusin', function () {
                openTip(tip);
            });
            tip.addEventListener('focusout', function (ev) {
                if (!tip.contains(ev.relatedTarget)) scheduleClose(tip);
            });
        });

        if (scrollEl) {
            scrollEl.addEventListener(
                'scroll',
                function () {
                    if (activeTip) closeTip(activeTip);
                },
                { passive: true }
            );
        }

        window.addEventListener(
            'scroll',
            function () {
                if (activeTip && activeTip._dBubble) layoutBubble(activeTip, activeTip._dBubble);
            },
            true
        );

        window.addEventListener('resize', function () {
            if (activeTip && activeTip._dBubble) layoutBubble(activeTip, activeTip._dBubble);
        });

        document.addEventListener('keydown', function (ev) {
            if (ev.key === 'Escape' && activeTip) closeTip(activeTip);
        });
    })();

    /* Author list: same floating tooltip system as desiderata matrix */
    (function authorListTooltips() {
        var wrap = document.querySelector('.publication-authors');
        if (!wrap) return;

        var activeTip = null;
        var hideTimer = null;
        var margin = 10;
        var maxWpx = 304;

        function clearHideTimer() {
            if (hideTimer) {
                clearTimeout(hideTimer);
                hideTimer = null;
            }
        }

        function scheduleClose(tip) {
            clearHideTimer();
            hideTimer = setTimeout(function () {
                closeTip(tip);
            }, 200);
        }

        function layoutBubble(tip, bubble) {
            var trigger = tip.querySelector('.author-name') || tip;
            if (!trigger || !bubble) return;

            var vw = window.innerWidth;
            var vh = window.innerHeight;
            var r = trigger.getBoundingClientRect();
            var capW = Math.min(maxWpx, vw - 2 * margin);

            bubble.style.position = 'fixed';
            bubble.style.transform = 'none';
            bubble.style.width = capW + 'px';
            bubble.style.left = '-9999px';
            bubble.style.top = '0';
            bubble.style.visibility = 'hidden';
            bubble.style.opacity = '0';

            var br = bubble.getBoundingClientRect();
            var bw = br.width;
            var bh = br.height;

            var cx = r.left + r.width / 2;
            var left = cx - bw / 2;
            if (left < margin) left = margin;
            if (left + bw > vw - margin) left = Math.max(margin, vw - margin - bw);

            var spaceBelow = vh - r.bottom - margin;
            var spaceAbove = r.top - margin;
            var top;
            if (spaceBelow >= bh + margin || spaceBelow >= spaceAbove) {
                top = r.bottom + margin;
                if (top + bh > vh - margin) top = Math.max(margin, vh - margin - bh);
            } else {
                top = r.top - margin - bh;
                if (top < margin) top = margin;
            }

            bubble.style.left = Math.round(left) + 'px';
            bubble.style.top = Math.round(top) + 'px';
            bubble.style.width = '';
            bubble.style.visibility = '';
            bubble.style.opacity = '';
        }

        function closeTip(tip) {
            if (!tip) return;
            clearHideTimer();
            var bubble = tip._dBubble;
            if (bubble) {
                bubble.classList.remove('is-open');
            }
            if (bubble && bubble._homeParent) {
                bubble._homeParent.appendChild(bubble);
                bubble.style.left = '';
                bubble.style.top = '';
                bubble.style.width = '';
                bubble.style.position = '';
                bubble.style.transform = '';
                bubble.style.visibility = '';
                bubble.style.opacity = '';
            }
            if (activeTip === tip) activeTip = null;
        }

        function openTip(tip) {
            clearHideTimer();
            if (activeTip && activeTip !== tip) closeTip(activeTip);

            var bubble = tip._dBubble || tip.querySelector('.d-tip-bubble');
            if (!bubble) return;
            tip._dBubble = bubble;
            if (!bubble._homeParent) bubble._homeParent = tip;

            document.body.appendChild(bubble);
            activeTip = tip;
            layoutBubble(tip, bubble);
            requestAnimationFrame(function () {
                bubble.classList.add('is-open');
            });
        }

        wrap.querySelectorAll('.d-tip').forEach(function (tip) {
            var bubble = tip.querySelector('.d-tip-bubble');
            if (bubble && !bubble._wired) {
                bubble._wired = true;
                bubble.addEventListener('mouseenter', clearHideTimer);
                bubble.addEventListener('mouseleave', function () {
                    scheduleClose(tip);
                });
            }

            tip.addEventListener('mouseenter', function () {
                openTip(tip);
            });
            tip.addEventListener('mouseleave', function () {
                scheduleClose(tip);
            });
            tip.addEventListener('focusin', function () {
                openTip(tip);
            });
            tip.addEventListener('focusout', function (ev) {
                if (!tip.contains(ev.relatedTarget)) scheduleClose(tip);
            });
        });

        window.addEventListener(
            'scroll',
            function () {
                if (activeTip && activeTip._dBubble) layoutBubble(activeTip, activeTip._dBubble);
            },
            true
        );

        window.addEventListener('resize', function () {
            if (activeTip && activeTip._dBubble) layoutBubble(activeTip, activeTip._dBubble);
        });

        document.addEventListener('keydown', function (ev) {
            if (ev.key === 'Escape' && activeTip) closeTip(activeTip);
        });
    })();

})
