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
    import { lab } from "d3";

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
	const inputOutputCanvasSize = 370;
	const scatterSquare = 270;
	const latentWidth = 270;
	const latentHeight = 450;
	const trapWidth = 100;
	const images = [1, 2, 3, 4, 5, 7].map((d) => `images/${d}.png`);
	let rawImages;
	let selectedImage = "images/1.png";
	const latentDims = 20;
	let inDisp = Array(784).fill(0);
	let outDisp = Array(784).fill(0);
	let drawImage = Array(784).fill(0);
	let boldness = 0.2;
	//let stddevs = Array(latentDims).fill(1);
	//let means = Array(latentDims).fill(0);
	const zs = writable(Array(latentDims).fill(0));

	let zz = Array(latentDims).fill(0);

	let cur_enc_drate = 0;
	let cur_dec_drate = 0;

	let enc_lbl = 0;
	let dec_lbl = 0;

	const enc_drate_list = [0.0, 0.25, 0.5, 0.75, 1.0];
	const dec_drate_list = [0.0, 0.25, 0.5, 0.75, 1.0];
	const enc_lbl_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
	const dec_lbl_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

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

	function forward(img) {
		tf.tidy(() => {
			const x = tf.tensor(img, [1, 28, 28, 1]);
			// const input_enc_lbl = tf.tensor([enc_lbl]);
			//  input_enc_label should be int32, and one_hot encoded
			// const input_enc_lbl = tf.tensor([enc_lbl], [1], 'int32');
			const input_enc_lbl = tf.oneHot(tf.tensor([enc_lbl], [1], "int32"), 11);
			// const input_enc_lbl = tf.oneHot(tf.tensor([enc_lbl],[1], "int32"), 11);
			console.log(x.arraySync());
			console.log(input_enc_lbl.arraySync());

			inDisp = img;
			const [enc, dec] = models[`${cur_enc_drate},${cur_dec_drate}`];
			const [mu, logstd] = enc.predict([x, input_enc_lbl]);
			
			// xs = mu.concat(logstd, 1).arraySync()[0];
			
			// const [z, logvar, mean, eps] = sample(code);
			const [z, logvar, mean, eps] = sample(mu, logstd);
			// $randomSample = eps.arraySync()[0];
			// $stddevs = tf.exp(logvar.mul(0.5)).arraySync()[0];
			// $means = mean.arraySync()[0];
			$zs = z.arraySync()[0];
			zz =  z.arraySync()[0];
			
			// const input_dec_lbl = tf.tensor([dec_lbl], [1], 'int32');
			const input_dec_lbl = tf.oneHot(tf.tensor([dec_lbl], [1], "int32"), 11);
			const xHat = dec.predict([input_dec_lbl, z]).reshape([-1, 784]);
			outDisp = xHat.arraySync()[0];
		});
	}

	// $: modelsExist = enc && dec;
	$: modelsExist = Object.keys(models).length === enc_drate_list.length * dec_drate_list.length;
	$: if (cur_enc_drate && cur_dec_drate && enc_lbl && dec_lbl) 
		forward(inDisp);
	

	$: if (modelsExist && rawImages && selectedImage)
		forward(rawImages[selectedImage]);

	// let enc, dec;
	// onMount(async () => {
	// 	[enc, dec] = await loadModels();
	// 	rawImages = await fetchAllImages(images);
	// 	rawImages["clear"] = new Float32Array(784).fill(0);
	// });
	// onDestroy(() => {
	// 	enc.dispose();
	// 	dec.dispose();
	// });

	let models = {}; // 用于存储模型字典

	onMount(async () => {
		for (let enc_drate of enc_drate_list) {
			for (let dec_drate of dec_drate_list) {
				const [enc, dec] = await loadModels(enc_drate, dec_drate);
				models[`${enc_drate},${dec_drate}`] = [enc, dec]; // 将模型存储到字典中
				console.log(`Loaded model with enc_drate: ${enc_drate}, dec_drate: ${dec_drate}`);
			}
		}
		rawImages = await fetchAllImages(images);
		rawImages["clear"] = new Float32Array(784).fill(0);
	});
	onDestroy(() => {
		for (const key in models) {
			const [enc, dec] = models[key];
			enc.dispose();
			dec.dispose();
		}
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
	<!-- <div class="mb-2 flex gap-2 items-center">
		<ImageSelector imageUrls={images} bind:selectedUrl={selectedImage} />
	</div> -->

	<!-- <div class="mb-2 flex gap-2 items-center">
		<label for="enc-drate-selector" class="font-large text-gray-300">Encoder Dropout Rate:</label>
		<select id="enc-drate-selector" bind:value={cur_enc_drate} class="border border-gray-200 rounded-md p-2 text-gray-300 bg-black w-20" on:change={forward(inDisp)}>
			{#each enc_drate_list as drate}
				<option 
					value={drate} class="text-white bg-black">{drate}
				</option>
			{/each}
		</select>
		<label for="dec-drate-selector" class="font-large text-gray-300">Decoder Dropout Rate:</label>
		<select id="dec-drate-selector" bind:value={cur_dec_drate} class="border border-gray-200 rounded-md p-2 text-gray-300 bg-black w-20" on:change={forward(inDisp)}>
			{#each dec_drate_list as drate}
				<option value={drate} class="text-white bg-black">{drate}</option>
			{/each}
		</select>
		<label for="enc-lbl-selector" class="font-large text-gray-300">Encoder Label :</label>
		<select id="enc-lbl-selector" bind:value={enc_lbl} class="border border-gray-200 rounded-md p-2 text-gray-300 bg-black w-20" on:change={forward(inDisp)}>
			{#each enc_lbl_list as lbl}
				<option value={lbl} class="text-white bg-black">{lbl}</option>
			{/each}
		</select>
		<label for="dec-lbl-selector" class="font-large text-gray-300">Decoder Label :</label>
		<select id="dec-lbl-selector" bind:value={dec_lbl} class="border border-gray-200 rounded-md p-2 text-gray-300 bg-black w-20" on:change={forward(inDisp)}>
			{#each dec_lbl_list as lbl}
				<option value={lbl} class="text-white bg-black">{lbl}</option>
			{/each}
		</select>
	</div> -->

	<div class="mb-4 p-4 rounded-xl bg-gray-900 shadow-lg border border-gray-700 flex flex-wrap gap-4 items-center justify-start">
  <!-- Encoder Dropout Rate -->
  <div class="flex flex-col items-start">
    <label for="enc-drate-selector" class="text-sm font-medium text-gray-400 mb-1">Encoder Dropout</label>
    <select
      id="enc-drate-selector"
      bind:value={cur_enc_drate}
      on:change={forward(inDisp)}
      class="bg-gray-800 text-white rounded-md px-3 py-2 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-pink-400 transition-all duration-200 w-32"
    >
      {#each enc_drate_list as drate}
        <option value={drate} class="bg-gray-800 text-white">{drate}</option>
      {/each}
    </select>
  </div>

  <!-- Decoder Dropout Rate -->
  <div class="flex flex-col items-start">
    <label for="dec-drate-selector" class="text-sm font-medium text-gray-400 mb-1">Decoder Dropout</label>
    <select
      id="dec-drate-selector"
      bind:value={cur_dec_drate}
      on:change={forward(inDisp)}
      class="bg-gray-800 text-white rounded-md px-3 py-2 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-purple-400 transition-all duration-200 w-32"
    >
      {#each dec_drate_list as drate}
        <option value={drate} class="bg-gray-800 text-white">{drate}</option>
      {/each}
    </select>
  </div>

  <!-- Encoder Label -->
  <div class="flex flex-col items-start">
    <label for="enc-lbl-selector" class="text-sm font-medium text-gray-400 mb-1">Encoder Label</label>
    <select
      id="enc-lbl-selector"
      bind:value={enc_lbl}
      on:change={forward(inDisp)}
      class="bg-gray-800 text-white rounded-md px-3 py-2 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-400 transition-all duration-200 w-32"
    >
      {#each enc_lbl_list as lbl}
        <option value={lbl} class="bg-gray-800 text-white">{lbl}</option>
      {/each}
    </select>
  </div>

  <!-- Decoder Label -->
  <div class="flex flex-col items-start">
    <label for="dec-lbl-selector" class="text-sm font-medium text-gray-400 mb-1">Decoder Label</label>
    <select
      id="dec-lbl-selector"
      bind:value={dec_lbl}
      on:change={forward(inDisp)}
      class="bg-gray-800 text-white rounded-md px-3 py-2 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-green-400 transition-all duration-200 w-32"
    >
      {#each dec_lbl_list as lbl}
        <option value={lbl} class="bg-gray-800 text-white">{lbl}</option>
      {/each}
    </select>
  </div>
  <!-- Image Selector -->
  <div class="flex flex-col items-start">
    <label class="text-sm font-medium text-gray-400 mb-1">Input Image</label>
    <div class="flex gap-2 flex-wrap">
      {#each images as src}
        <img
          src={src}
          alt="sample"
          width={40}
          class="rounded-md transition-all cursor-pointer opacity-40 hover:opacity-70"
          class:outlined={selectedImage === src}
          on:click={() => {
            selectedImage = src;
            forward(inDisp); // Automatically update output on selection
          }}
        />
      {/each}
      <!-- Clear Button -->
      <div
        class="w-[40px] h-[40px] bg-black opacity-40 cursor-pointer rounded-md transition-all"
        on:click={() => {
          selectedImage = 'clear';
          forward(inDisp);
        }}
        class:selected-outline={selectedImage === 'clear'}
      ></div>
    </div>
  </div>
</div>
	

	<svg
		width={xDigit2 + inputOutputCanvasSize + 100}
		height={popoverY + 600}
		style="overflow: visible;"
	>
		<foreignObject
			x={xDigit1}
			y={40}
			width={inputOutputCanvasSize}
			height={inputOutputCanvasSize}
			style="overflow: visible;"
		>
			<MnistDigit
				style="outline: 2px solid var(--pink); cursor: crosshair;"
				enableDrawing
				boldness = {boldness}
				data={inDisp}
				square={inputOutputCanvasSize}
				maxVal={1}
				onChange={(d) => {
					drawImage = d;
					forward(d);
				}}
			></MnistDigit>
			<div class="boldness-slider">
				<div class="slider-label">Boldness</div>
				<input
					type="range"
					min="0"
					max="1"
					step="0.01"
					bind:value={boldness}
					on:input={() => {
						console.log(`Boldness: ${boldness}`);
					}}
				/>
				<div class="slider-value">{boldness.toFixed(2)}</div>
				
			<style>
				.boldness-slider {
					display: flex;
					align-items: center;
					justify-content: space-between;
					gap: 10px;
					padding: 10px;
					border: 1px solid #ccc;
					border-radius: 5px;
				}
				.slider-label {
					font-size: 1em;
					font-weight: bold;
					color: #999;
				}
				.slider-value {
					font-size: 0.9em;
					color: #666;
				}
				input[type="range"] {
					width: 100%;
					height: 10px;
					background: #ddd;
					accent-color: #4caf50;
				}
			</style>
			</div>
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
			y={40}
			width={trapWidth}
			height={inputOutputCanvasSize}
			trapHeights={[inputOutputCanvasSize, scatterSquare]}
		/>

		

		<Trapezoid
			label="Decoder"
			fill="--light-blue"
			fill2="--green"
			x={xTrap2}
			y={40}
			width={trapWidth}
			height={inputOutputCanvasSize}
			trapHeights={[scatterSquare, inputOutputCanvasSize]}
		/>

		<!-- <foreignObject
			x={xLatent}
			y={20}
			width={latentWidth}
			height={latentHeight}
			style="overflow: visible;outline: 2px solid var(--purple);\
			padding-left: 10px; padding-right: 5px; padding-top: 10px; \
			padding-bottom: 10px; "
		>
		<div class="latent-slider-container">
			{#each Array.from({ length: latentDims }, (_, i) => i) as index}
				<div class="latent-slider" style="display: flex; align-items: center; justify-content: space-between;">
				<div class="latent-slider-label" style="display: flex; align-items: center;">
					<label for="latent-slider-{index}" style="padding-left: 10px;width: 30px; margin-right: 5px; font-size: 0.9em;">
						{index + 1}
					</label>
					<div style="width: 1px; height: 15px; background-color: #ccc; margin: 0 2px;"></div>
						<code style="font-size: 0.8em; color: #666;">
							{zz[index].toFixed(2)}
						</code>
					</div>
					<input
						id="latent-slider-{index}"
						type="range"
						min="-5"
						max="5"
						step="0.1"
						
						bind:value={zz[index]}
						on:input={() => {
							tf.tidy(() => {
								const z = tf.tensor(zz, [1, latentDims]);
								// console.log("z", z);
								// const xHat = dec.predict(z).reshape([-1, 784]);
								const input_dec_lbl = tf.oneHot(tf.tensor([dec_lbl], [1], "int32"), 11);
								const dec = models[`${cur_enc_drate},${cur_dec_drate}`][1];
								const xHat = dec.predict([input_dec_lbl, z]).reshape([-1, 784]);

								// console.log("xHat", xHat);
								outDisp = xHat.arraySync()[0];
							});
						}}
						style="flex: 1; height: 2px; margin-left: 10px;"
					/>
				</div>
			{/each}
		</div>
		</foreignObject> -->
		<foreignObject
  x={xLatent-20}
  y={20}
  width={latentWidth+40}
  height={latentHeight}
  style="
    overflow: visible;
    padding: 16px;
  "
>
  <!-- Card-style container -->
  <div class="bg-gray-900 rounded-xl shadow-lg border border-purple-600 p-4 space-y-3 w-full h-full">
    <!-- Title or Header (optional) -->
    <h3 class="text-sm text-purple-300 font-semibold mb-2">Latent Space Sliders</h3>

    <!-- Slider list -->
    <div class="space-y-3 overflow-y-auto max-h-[340px] pr-2">
      {#each Array.from({ length: latentDims }, (_, i) => i) as index}
        <div class="flex items-center gap-3">
          <!-- Label and value -->
          <div class="flex items-center gap-2 w-24 shrink-0">
            <label
              for="latent-slider-{index}"
              class="text-gray-300 text-sm font-mono w-6 text-right"
            >
              {index + 1}
            </label>
            <div class="w-px h-4 bg-gray-600"></div>
            <code class="text-gray-400 text-xs">{zz[index].toFixed(2)}</code>
          </div>

          <!-- Slider -->
          <input
            id="latent-slider-{index}"
            type="range"
            min="-5"
            max="5"
            step="0.01"
            bind:value={zz[index]}
            on:input={() => {
              tf.tidy(() => {
                const z = tf.tensor(zz, [1, latentDims]);
                const input_dec_lbl = tf.oneHot(tf.tensor([dec_lbl], [1], "int32"), 11);
                const dec = models[`${cur_enc_drate},${cur_dec_drate}`][1];
                const xHat = dec.predict([input_dec_lbl, z]).reshape([-1, 784]);
                outDisp = xHat.arraySync()[0];
              });
            }}
            class="flex-1 h-2 appearance-none bg-gray-700 rounded-md accent-purple-500 cursor-pointer transition-all"
          />
        </div>
      {/each}
    </div>
  </div>
</foreignObject>




		<foreignObject
			x={xDigit2}
			y={40}
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

		{#if expanded && $stddevs && $means}
			<Popover x={popoverX} y={popoverY} />
		{/if}
	</svg>

	<div class="bg-gray-900 text-gray-200 p-6 rounded-xl shadow-md border border-gray-700 leading-relaxed space-y-4 text-sm sm:text-base transform -translate-y-80">
  <p>
    Our <span class="font-semibold text-blue-400">Conditional VAE Explainer</span> is structured as follows:
  </p>
  <p>
    The <span class="font-semibold">Encoder</span> receives an input image and an <code class="bg-gray-800 px-1 py-0.5 rounded text-purple-300">encoder label</code> (ranging from <code>0</code> to <code>9</code> for digits, or <code>10</code> to indicate no label). It encodes the image into a 20-dimensional latent space, outputting <code class="bg-gray-800 px-1 py-0.5 rounded text-cyan-300">mean</code> and <code class="bg-gray-800 px-1 py-0.5 rounded text-cyan-300">var</code>.
    You can manipulate the sampled latent vector <code class="text-purple-300">z</code> through the 20 sliders in the middle panel.
  </p>
  <p>
    This vector <code class="text-purple-300">z</code>, along with a <code class="bg-gray-800 px-1 py-0.5 rounded text-green-300">decoder label</code> (also <code>0</code> to <code>9</code> or <code>10</code>), is passed into the <span class="font-semibold">Decoder</span> to reconstruct a digit image.
  </p>
  <p>
    <span class="font-semibold text-purple-400">Encoder Dropout Rate</span>: the probability that the encoder label is replaced with <code>10</code>. A value of <code>0</code> means label information is always preserved; a value of <code>1</code> means it is always removed, making the encoder function like that of an unconditional VAE.
  </p>
  <p>
    <span class="font-semibold text-green-400">Decoder Dropout Rate</span>: similarly, the probability that the decoder label is replaced with <code>10</code>.
  </p>
  <p>
    You can draw freely on the input canvas to visualize real-time decoded outputs, or choose from predefined digit samples to feed into the encoder. Feel free to explore!
  </p>
  <p>
    <span class="font-semibold text-yellow-300">Interesting observations:</span> Several sliders correspond to interpretable features like vertical/horizontal translation, small rotations, stroke thickness, scale changes, or digit deformation. Others may have little or no noticeable effect.
  </p>
</div>


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
