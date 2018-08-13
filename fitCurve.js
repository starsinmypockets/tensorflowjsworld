require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const defaults = {
  a: tf.variable(tf.scalar(Math.random())),
  b: tf.variable(tf.scalar(Math.random())),
  c: tf.variable(tf.scalar(Math.random())),
  d: tf.variable(tf.scalar(Math.random())),
  x: tf.variable(tf.scalar(Math.random())),
  numIterations: 75,
  learningRate: 0.5,
  trainingMethod: 'sgd',
  test: 'default'
}

class FitCurve {
  constructor(options) {
    Object.assign(this, defaults, options)
    this.a.print() 
    this.b.print()
    this.c.print()
    this.d.print()
    this.x.print()
  }
  
  optimizer(learningRate = .5) {
    tf.train.sgd(learningRate)
  }

  test() {
    return this.test
  }
  
  predict(x) {
    const {a, b, c, d} = this

    return tf.tidy(() => {
      // ax^3 + bx^2 + cx + d
      return a.mul(x.pow(tf.scalar(3)))
        .add(b.mul(x.square()))
        .add(c.mul(x))
        .add(d)
    })
  }

  // our loss function - mean squared error
  loss(predicted, ys) {
    return predicted.sub(ys).square().mean()
  }
  
  // train with data
  train(xs, ys, n) {
    for (let i = 0; i < n; i++) {
      const pred = this.predict(xs[i])
      return loss(pred, ys)
    }
  } 
  
  generateData(numPoints, coeff, sigma = 0.04) {
		// https://github.com/tensorflow/tfjs-examples/blob/master/polynomial-regression-core/data.js
		return tf.tidy(() => {
			const [a, b, c, d] = [
				tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
				tf.scalar(coeff.d)
			];

			const xs = tf.randomUniform([numPoints], -1, 1);

			// Generate polynomial data
			const three = tf.scalar(3, 'int32');
			const ys = a.mul(xs.pow(three))
				.add(b.mul(xs.square()))
				.add(c.mul(xs))
				.add(d)
				// Add random noise to the generated data
				// to make the problem a bit more interesting
				.add(tf.randomNormal([numPoints], 0, sigma));

			// Normalize the y values to the range 0 to 1.
			const ymin = ys.min();
			const ymax = ys.max();
			const yrange = ymax.sub(ymin);
			const ysNormalized = ys.sub(ymin).div(yrange);

			return {
				xs, 
				ys: ysNormalized
			};
		}) 
  }
}

module.exports = FitCurve
