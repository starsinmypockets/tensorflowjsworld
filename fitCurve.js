const tf = require('@tensorflow/tfjs')

// Load the binding:
require('@tensorflow/tfjs-node')

// Train a simple model:
const model = tf.sequential()
model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]}))
model.add(tf.layers.dense({units: 1, activation: 'linear'}))
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'})

module.exports = {
  model: model,
  xs: tf.randomNormal([100, 10]),
  ys: tf.randomNormal([100, 1])
}
