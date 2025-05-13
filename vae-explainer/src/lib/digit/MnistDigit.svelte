<!-- https://github.com/xnought/introduction-to-autoencoders/blob/main/src/MNIST/components/mnistDigit/MnistDigit.svelte -->
<script>
	import { drawPixels } from "./mnistDigit";
	import { interpolateGreys, scaleLinear } from "d3";
	import { onMount } from "svelte";
	import * as d3 from "d3";

  export let style = "";

	// take in the digit data
	// export let digit: IData;
	// const { data, label } = digit;
	function cpyArray(a) {
		const b = new Float32Array(a.length);
		for (let i = 0; i < a.length; i++) {
			b[i] = a[i];
		}
		return b;
	}
	export let enableDrawing = false;
	export let data;
	let cpy;
	$: if (data) cpy = cpyArray(data);
	$: if (data && ctx) redraw(data);

	// configuration
	export let onChange = (x) => {};
	export let square = 250;
	export let dim = 28;
	export let width = square;
	export let height = square;
	let canvasPtr; // points to the canvas in the DOM
	let ctx; // the context assigned onMount
	export let boldness = 2; // 1, 2, or 3
	export let maxVal = 255;

	// create greyscale function 0 -> 1 output shade
	export let range = [1, 0];
	let colorScale = scaleLinear().domain([0, maxVal]).range(range);
	const greyScale = (value) => interpolateGreys(colorScale(value));

	// on creation paint the digit
	const redraw = (data) => {
		ctx.clearRect(0, 0, square, square); // clear the canvas
		drawPixels(ctx, width, dim, greyScale, data); // draw the digit on canvas
	};

	// on creation make sure to store the canvas context
	onMount(() => {
		ctx = canvasPtr.getContext("2d");
	});
	// $: if (data && ctx) redraw();
	let drawing = false;
	let mousePos = [0, 0];

  function drawBrush(cpy, i, j, dim, pixelSize, boldness) {
	const brushL = [
		[0.00, 0.00, 0.0, 0.00, 0.00],
		[0.00, 0.0, 0.1, 0.0, 0.00],
		[0.0, 0.1, 0.2, 0.1, 0.0],
		[0.00, 0.0, 0.1, 0.0, 0.00],
		[0.00, 0.00, 0.0, 0.00, 0.00]
	];

	const brushR = [
		[0.1, 0.2, 0.4, 0.2, 0.1],
		[0.2, 0.4, 0.8, 0.4, 0.2],
		[0.4, 0.8, 1.0, 0.8, 0.4],
		[0.2, 0.4, 0.8, 0.4, 0.2],
		[0.1, 0.2, 0.4, 0.2, 0.1]
	];

	const brushLocations = [
		[[-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2]],
		[[-2, -1], [-1, -1], [0, -1], [1, -1], [2, -1]],
		[[-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0]],
		[[-2, 1], [-1, 1], [0, 1], [1, 1], [2, 1]],
		[[-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2]]
	];
    // basically draws a pixel around the given i, j pixel given the brush
    for(let bi = 0; bi < brushL.length; bi++) {
      inner: for(let bj = 0; bj < brushL[0].length; bj++) {
        const o = brushL[bi][bj] + brushR[bi][bj] * boldness;
        const [di, dj] = brushLocations[bi][bj];
        const ic = i + di;
        const jc = j + dj;

        // if out of bounds, don't draw this pixel!
        if(ic < 0 || jc < 0 || jc >= dim || ic >= dim) continue inner;

				const x = ic * pixelSize;
				const y = jc * pixelSize;
        ctx.fillRect(x, y, pixelSize, pixelSize);
        ctx.fillStyle = d3.color(`hsla(0, 0%, 255%, ${o})`);
        cpy[ic * dim + jc] += o;
        cpy[ic * dim + jc] = Math.min(1, cpy[ic * dim + jc]); // clamp
      }
    }
    onChange(cpy);
}
	function drawBigPixel(ctx, width, dim, mousePos, boldness = 2) {
		const canvasBigPixelSize = width/dim;
		// Take an x,y point and
		for (let i = 0; i < dim; i++) {
			for (let j = 0; j < dim; j++) {
				const x = j * canvasBigPixelSize;
				const y = i * canvasBigPixelSize;
				const withinX =
					mousePos[0] >= x && mousePos[0] <= x + canvasBigPixelSize;
				const withinY =
					mousePos[1] >= y && mousePos[1] <= y + canvasBigPixelSize;
				const withinPixel = withinX && withinY;
				if (withinPixel) {
          drawBrush(cpy, i, j, dim, canvasBigPixelSize, boldness);
					return;
				}
			}
		}
	}
</script>

<div {style}>
	<canvas
		bind:this={canvasPtr}
		{width}
		{height}
		on:mousedown={() => {
			drawing = true;
		}}
		on:mouseup={() => {
			drawing = false;
		}}
		on:mousemove={({ offsetX, offsetY }) => {
			if (drawing && enableDrawing) {
				mousePos = [offsetX, offsetY];
				// cover the block where the pixel is with full data white (1.0), then rerender the entire thing;
				drawBigPixel(ctx, width, dim, mousePos, boldness);
			}
		}}
	/>
	<!-- some buttom to switch boldness -->
	<!-- <div class="flex gap-2">
		<button
			class="btn btn-xs btn-primary"
			on:click={() => {
				boldness = 1;
			}}
		>
			thin
		</button>
		<button
			class="btn btn-xs btn-primary"
			on:click={() => {
				boldness = 2;
			}}
		>
			medium	
		</button>
		<button
			class="btn btn-xs btn-primary"
			on:click={() => {
				boldness = 3;
			}}
		>
			heavy	
		</button>
	</div> -->
</div>

<style>
</style>
