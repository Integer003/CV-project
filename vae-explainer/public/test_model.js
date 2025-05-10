import * as tf from '@tensorflow/tfjs';

// 加载模型
async function loadModel(modelUrl) {
  const model = await tf.loadGraphModel(modelUrl);
  console.log('模型加载完成');
  return model;
}

// 验证输入输出
async function testModel(model) {
  // 创建输入数据
  const inputLabel = tf.tensor1d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'int32'); // 示例标签
  const inputLatent = tf.randomNormal([11, 20]); // 随机生成一个形状为 [11, 20] 的浮点张量

  // 执行推理
  const output = model.execute({ input_label: inputLabel, input_latent: inputLatent });

  // 输出结果
  output.print();
  console.log('输出形状:', output.shape);
}

// 主函数
async function main() {
  const modelUrl = 'tfjs_models/encoder-0.0-0.0-big/model.json'; // 替换为你的模型路径
  const encoder = await tf.loadGraphModel(modelUrl);
  await testModel(encoder);
}

main();