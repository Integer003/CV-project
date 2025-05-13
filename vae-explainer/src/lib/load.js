import * as tf from "@tensorflow/tfjs";

const filepath = "tfjs_models_10D";




export function sample(mu, logvar) {
const zStdDev = tf.exp(logvar.mul(0.5)); // e^(1/2 sigma^2) = sigma

	const eps = tf.randomNormal(mu.shape);
	const z = tf.add(mu, tf.mul(eps, zStdDev)); // \mu + \sigma * \epsilon    or stddev*N(0, I) + mean

	return [z, logvar, mu, eps];
}

export async function loadModels(enc_drate, dec_drate) {
	if ( enc_drate == 0)
		enc_drate = "0.0"
	if ( dec_drate == 0)
		dec_drate = "0.0"
	if ( enc_drate == 1)
		enc_drate = "1.0"
	if ( dec_drate == 1)
		dec_drate = "1.0"
	

	const encoder = await tf.loadGraphModel(`${filepath}/encoder-${enc_drate}-${dec_drate}-big/model.json`);
	const decoder = await tf.loadGraphModel(`${filepath}/decoder-${enc_drate}-${dec_drate}-big/model.json`);
	return [encoder, decoder];
}

// export async function loadModels() {
// 	const encoder = await tf.loadGraphModel(`${filepath}/encoder-big/model.json`);
// 	const decoder = await tf.loadGraphModel(`${filepath}/decoder-big/model.json`);
// 	return [encoder, decoder];
// }

export async function loadLatents() {
  const d = await (await fetch(`${filepath}/latents-big.json`)).json();
  return d;
}

// export async function loadLatents(enc_drate, dec_drate) {
// 	const d = await (await fetch(`${filepath}/latents-${enc_drate}-${dec_drate}-big.json`)).json();
// 	return d;
//   }



export function loadImage(url) {
	const img = new Image();
	return new Promise((res, rej) => {
		img.src = url;
		img.onload = () => res(img);
		img.onerror = rej;
	});
}
