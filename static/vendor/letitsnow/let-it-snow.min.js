! function (a) {
	a.letItSnow = function (b, c) {
		var d = this;
		d.$el = a(b), d.el = b, d.$el.data("LetItSnow", d), d.init = function () {
			if (d.options = a.extend({}, a.letItSnow.defaultOptions, c), 1 == d.options.makeFlakes)
				for (var b = 0; 100 > b; b++) d.$el.prepend('<span class="lis-flake"></span>');
			1 == d.options.sticky && (d.makeFlakes(), setInterval(function () {
				d.makeFlakes()
			}, 7e3))
		}, d.makeFlakes = function () {
			a(".lis-flake--js").remove();
			for (var b = 0; 50 > b; b++) d.$el.append('<span class="' + d.options.stickyFlakes + '"></span>');
			var c = a("." + d.options.stickyFlakes);
			d.animateFlakes(c)
		};
		var e = d.$el.find("[collectsnow]");
		if (e.length) var f = e.offset().top,
			g = e.offset().left,
			h = a(window).width() - (g + e.outerWidth());
		d.animateFlakes = function (b) {
			var c = function (a, b) {
					return Math.floor(Math.random() * (b - a + 1)) + a
				},
				d = 0;
			b.each(function () {
				function b(b) {
					i.animate({
						marginTop: "2500px"
					}, {
						duration: b,
						step: function (b) {
							if (e.length) {
								var c = Math.round(b),
									d = a(window).width() - (i.offset().left + i.outerWidth());
								j + c >= f - 10 && f + 10 >= j + c && i.css("left").replace(/[^-\d\.]/g, "") >= g && d >= h && i.stop().attr("class", "lis-flake--stuck").css({
									width: "3px",
									height: "3px",
									top: j + 6 + "px"
								})
							}
						}
					})
				}
				var i = a(this),
					j = 10 * c(0, 100) - 1e3;
				d++;
				var k = {
					left: c(0, 100) + "%",
					top: j + "px",
					width: c(0, 5) + "px",
					height: c(0, 5) + "px",
					transform: "rotate" + (c(0, 100) + "deg")
				};
				i.css(k), i.hasClass("lis-flake--stuck") || (16 >= d ? b(1e4) : d > 17 && 33 >= d ? setTimeout(function () {
					b(8e3)
				}, 4e3) : d > 34 && 50 > d && setTimeout(function () {
					b(8e3)
				}, 6e3))
			})
		}, d.init()
	}, a.letItSnow.defaultOptions = {
		stickyFlakes: "lis-flake--js",
		makeFlakes: !0,
		sticky: !0
	}, a.fn.letItSnow = function (b) {
		return this.each(function () {
			new a.letItSnow(this, b)
		})
	}
}(jQuery);
