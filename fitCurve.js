require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')

// Train a simple model:
const model = tf.sequential()
model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]}))
model.add(tf.layers.dense({units: 1, activation: 'linear'}))
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'})

module.exports = {
  model: model,
  a: tf.variable(tf.scalar(Math.random())),
  b: tf.variable(tf.scalar(Math.random())),
  c: tf.variable(tf.scalar(Math.random())),
  d: tf.variable(tf.scalar(Math.random()))
}
