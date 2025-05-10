<script>
	import { onDestroy, onMount } from "svelte";
	import { loadImage, loadModels, sample } from "./lib/load";
	import { Button, Spinner } from "flowbite-svelte";
	import {
		ExpandOutline,
		MinimizeOutline,
		TrashBinOutline,
	} from "flowbite-svelte-icons";
	import * as tf from "@tensorflow/tfjs";
	import MnistDigit from "./lib/digit/MnistDigit.svelte";
	import NormalCurve from "./lib/NormalCurve.svelte";
	import LatentScatter from "./lib/LatentScatter.svelte";
	import Header from "./lib/Header.svelte";
	import ImageSelector from "./lib/ImageSelector.svelte";
	import Trapezoid from "./lib/Trapezoid.svelte";
	import Popover from "./lib/Popover.svelte";
	import Sankey from "./lib/Sankey.svelte";
	import Katex from "./lib/Katex.svelte";

	import { tweened } from "svelte/motion";
	import { cubicOut } from "svelte/easing";
	import {
		stddevs,
		means,
		randomSample,
		popoverWidth,
		popoverEncHeight,
		popoverEncY,
		popoverDecY,
		popoverDecHeight
	} from "./lib/stores";
	import { writable, derived, get } from "svelte/store";
	import Code from "./lib/Code.svelte";

	function toGrey(d) {
		const result = new Uint8ClampedArray(d.length / 4);
		for (let i = 0, j = 0; i < d.length; i += 4, j++) {
			result[j] = d[i];
		}
		return result;
	}

	function showMemory() {
		console.table(tf.memory());
	}

	async function loadImageFull(url) {
		const img = await loadImage(url);
		const canvas = document.createElement("canvas");
		const ctx = canvas.getContext("2d");
		ctx.drawImage(img, 0, 0);
		const d = ctx.getImageData(0, 0, img.width, img.height).data;
		img.remove();
		canvas.remove();
		return d;
	}

	const compGraph = true;
	const inputOutputCanvasSize = 300;
	const scatterSquare = 200;
	const trapWidth = 100;
	const images = [1, 2, 3, 4, 5, 7].map((d) => `images/${d}.png`);
	let rawImages;
	let selectedImage = "images/1.png";
	const latentDims = 20;
	let inDisp = Array(784).fill(0);
	let outDisp = Array(784).fill(0);
	//let stddevs = Array(latentDims).fill(1);
	//let means = Array(latentDims).fill(0);
	const zs = writable(Array(latentDims).fill(0));

	let zz = Array(latentDims).fill(0);

	async function fetchAllImages(urls) {
		let result = {};
		for (let i = 0; i < urls.length; i++) {
			const url = urls[i];
			const d = await loadImageFull(url);
			const g = toGrey(d);
			const f32 = new Float32Array(g.length).map((_, i) => g[i] / 255);
			result[url] = f32;
		}
		return result;
	}

	$: {
        if (outDisp) {
            console.log('outDisp updated:', outDisp);
        }
    }

	// 删除以下代码
	// let xs = Array(latentDims * 2).fill(0);

	// 修改 forward 函数，删除与二维高斯映射相关的逻辑
	function forward(img) {
		tf.tidy(() => {
			const x = tf.tensor(img, [1, 28, 28, 1]);
			inDisp = img;

			const code = enc.predict(x);
			// 删除以下代码
			// xs = code.arraySync()[0];

			const [z, logvar, mean, eps] = sample(code);
			// $randomSample = eps.arraySync()[0];
			// $stddevs = tf.exp(logvar.mul(0.5)).arraySync()[0];
			// $means = mean.arraySync()[0];
			$zs = z.arraySync()[0];

			// 修改为直接使用 zs
			// const z = tf.tensor($zs, [1, latentDims]);
			console.log("z", z);
			const xHat = dec.predict(z).reshape([-1, 784]);
			outDisp = xHat.arraySync()[0];
		});
}

	$: modelsExist = enc && dec;
	$: if (modelsExist && rawImages && selectedImage)
		forward(rawImages[selectedImage]);

	let enc, dec;
	onMount(async () => {
		[enc, dec] = await loadModels();
		rawImages = await fetchAllImages(images);
		rawImages["clear"] = new Float32Array(784).fill(0);
	});
	onDestroy(() => {
		enc.dispose();
		dec.dispose();
	});

	const width = 1200;
	const height = 500;

	let expanded = false;
	const expandedSize = 275;
	const minimizedSize = 20;
	const cExpansion = tweened(expanded ? expandedSize : minimizedSize, {
		duration: 1000,
		easing: cubicOut,
	});
	$: expansion = $cExpansion;
	$: xDigit1 = 0;
	$: padding = 100;
	$: trapPadding = 10;
	$: xLatent = xDigit1 + inputOutputCanvasSize + padding + expansion;
	$: yLatent = inputOutputCanvasSize / 2 - scatterSquare / 2;
	$: xDigit2 = xLatent + scatterSquare + padding + expansion;
	$: xTrap1 = xDigit1 + inputOutputCanvasSize + trapPadding;
	$: xTrap2 = xDigit2 - trapWidth - trapPadding;
	$: yTrap2 = xLatent;
	$: xMid = xLatent + scatterSquare / 2;
	$: popoverX = xMid - $popoverWidth / 2;
	$: popoverY = 300;
	$: xTrap1Out = xTrap1 + trapWidth;
	$: yTrap1Out = yLatent;
	$: fullyExpanded = expansion == expandedSize;
	$: fullyMinimized = expansion == minimizedSize;
</script>

<Header></Header>

<main>
	<div class="mb-2 flex gap-2 items-center">
		<ImageSelector imageUrls={images} bind:selectedUrl={selectedImage} />
	</div>

	<div class="latent-slider-container">
		{#each Array.from({ length: latentDims }, (_, i) => i) as index}
			<div class="latent-slider">
				<label for="latent-slider-{index}">Latent Dim {index}</label>
				<input
					id="latent-slider-{index}"
					type="range"
					min="-5"
					max="5"
					step="0.1"
					bind:value={zz[index]}
					on:input={() => {
						console.log("changing zs", zs);
						tf.tidy(() => {
							// const z = tf.tensor($zs, [1, latentDims]);
							// z.data().then(data => {
							// 	console.log('Data:', data);
							// });

							// // 使用 array() 方法获取数值
							// z.array().then(array => {
							// 	console.log('Array:', array);
							// });

							// // 使用 buffer() 方法获取数值
							// z.buffer().then(buffer => {
							// 	console.log('Buffer:', buffer.values);
							// });
							const z = tf.tensor(zz, [1, latentDims]);
							console.log("z", z);
							const xHat = dec.predict(z).reshape([-1, 784]);
							console.log("xHat", xHat);
							outDisp = xHat.arraySync()[0];
						});
					}}
				/>
			</div>
		{/each}
	</div>
	

	<svg
		width={xDigit2 + inputOutputCanvasSize + 100}
		height={popoverY + 600}
		style="overflow: visible;"
	>
	
		<foreignObject
			x={xDigit1}
			y={0}
			width={inputOutputCanvasSize}
			height={inputOutputCanvasSize}
			style="overflow: visible;"
		>
			<MnistDigit
				style="outline: 2px solid var(--pink); cursor: crosshair;"
				enableDrawing
				data={inDisp}
				square={inputOutputCanvasSize}
				maxVal={1}
				onChange={(d) => {
					forward(d);
				}}
			></MnistDigit>
			<Button
				class="mt-2"
				size="xs"
				color="alternative"
				on:click={() => {
					selectedImage = "clear";
					rawImages = rawImages; // weirdly needed for UI to update;
				}}><TrashBinOutline class="mr-1" size="sm" /> Clear</Button
			>
		</foreignObject>

		<Trapezoid
			label="Encoder"
			fill="--pink"
			fill2="--purple"
			x={xTrap1}
			y={0}
			width={trapWidth}
			height={inputOutputCanvasSize}
			trapHeights={[inputOutputCanvasSize, scatterSquare]}
		/>

		<foreignObject
			x={xLatent}
			y={yLatent + scatterSquare + 30}
			width={200}
			height={50}
		>
			<Button
				size="xs"
				color="light"
				style="width: 200px;"
				on:click={() => {
					if (expanded) {
						$cExpansion = minimizedSize;
					} else {
						$cExpansion = expandedSize;
					}
					expanded = !expanded;
				}}
			>
				{#if expanded}
					<MinimizeOutline class="mr-1" size="sm" /> Minimize Details
				{:else}
					<ExpandOutline class="mr-1" size="sm" /> Explain VAE Details
				{/if}
			</Button>
		</foreignObject>


		<Trapezoid
			label="Decoder"
			fill="--light-blue"
			fill2="--green"
			x={xTrap2}
			y={0}
			width={trapWidth}
			height={inputOutputCanvasSize}
			trapHeights={[scatterSquare, inputOutputCanvasSize]}
		/>
		
		


		<foreignObject
			x={xDigit2}
			y={0}
			width={inputOutputCanvasSize}
			height={inputOutputCanvasSize}
			style="overflow: visible;"
		>
			<MnistDigit
				data={outDisp}
				square={inputOutputCanvasSize}
				maxVal={1}
				style="outline: 2px solid var(--green);"
			></MnistDigit>
		</foreignObject>

	</svg>

	
</main>

<!--
<div style="position: absolute; bottom: 5px; right: 5px;">
	<Button color="alternative" on:click={() => showMemory()}
		>debug mode: tf.memory()</Button
	>
</div>
-->

<style>
	main {
		padding: 20px;
	}
	#tool {
		display: flex;
		gap: 5px;
	}
	#innards {
		display: flex;
		gap: 5px;
		align-items: center;
	}
	foreignObject {
		overflow: visible;
	}
</style>
