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
