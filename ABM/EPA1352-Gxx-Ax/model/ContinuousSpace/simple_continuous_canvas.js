var ContinuousVisualization = function(height, width, context) {
	var height = height;
	var width = width;
	var context = context;

	this.draw = function(objects) {
		for (var i in objects) {
			var p = objects[i];
			if (p.Shape == "rect")
				this.drawRectangle(p.x, p.y, p.w, p.h, p.Color, p.Filled, p.Text, p.Text_color);
			if (p.Shape == "circle")
				this.drawCircle(p.x, p.y, p.r, p.Color, p.Filled, p.Text, p.Text_color);
		};
	};

	this.drawCircle = function(x, y, radius, color, fill, text, text_color) {

		var cx = x * width;
		var cy = y * height;
		var r = radius;

		context.beginPath();
		context.arc(cx, cy, r, 0, Math.PI * 2, false);
		context.closePath();

		context.strokeStyle = color;
		context.stroke();

		if (fill) {
			context.fillStyle = color;
			context.fill();
		}

		// This part draws the text inside the Circle
        if (text !== undefined) {
                context.fillStyle = text_color;
                context.textAlign = 'center';
                context.textBaseline= 'middle';
                context.fillText(text, cx, cy);
        }

	};

	this.drawRectangle = function(x, y, w, h, color, fill, text, text_color) {

		var cx = x * width;
		var cy = y * height;
		var dx = w;
		var dy = h;

		// Keep the drawing centered:
		var x0 = cx - dx * 0.5;
		var y0 = cy - dy * 0.5;

		context.beginPath();
		context.strokeStyle = color;
		context.fillStyle = color;

		if (fill)
			context.fillRect(x0, y0, dx, dy);
		else
			context.strokeRect(x0, y0, dx, dy);

		// This part draws the text inside the Rectangle
        if (text !== undefined) {
                context.fillStyle = text_color;
                context.textAlign = 'center';
                context.textBaseline= 'middle';
                context.fillText(text, cx, cy);
        }
	};

	this.resetCanvas = function() {
		context.clearRect(0, 0, height, width);
		context.beginPath();
	};
};

var Simple_Continuous_Module = function(canvas_width, canvas_height) {
	// Create the element
	// ------------------

	// Create the tag:
	var canvas_tag = "<canvas width='" + canvas_width + "' height='" + canvas_height + "' ";
	canvas_tag += "style='border:1px dotted'></canvas>";
	
	var parent_div_tag = '<div style="height:' + canvas_height + 'px;" class="world-grid-parent" target="_sim"></div>'
	
	// Append it to body:
	var canvas = $(canvas_tag)[0];
	
	var parent = $(parent_div_tag)[0];
	
	//$("body").append(canvas);
	$("#elements").append(parent);
	parent.append(canvas);

	// Create the context and the drawing controller:
	var context = canvas.getContext("2d");
	var canvasDraw = new ContinuousVisualization(canvas_width, canvas_height, context);

	this.render = function(data) {
		canvasDraw.resetCanvas();
		canvasDraw.draw(data);
	};

	this.reset = function() {
		canvasDraw.resetCanvas();
	};

};
