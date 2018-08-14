const isNode = (typeof module !== 'undefined')

if (isNode) {
  var tf = require('@tensorflow/tfjs')
}

const defaults = {
  a: tf.variable(tf.scalar(Math.random())),
  b: tf.variable(tf.scalar(Math.random())),
  c: tf.variable(tf.scalar(Math.random())),
  d: tf.variable(tf.scalar(Math.random())),
  x: tf.variable(tf.scalar(Math.random())),
  numIterations: 75,
  learningRate: 0.5,
  trainingMethod: 'sgd',
}

class FitCurve {
  constructor(options) {
    Object.assign(this, defaults, options)
    this.a.print() 
    this.b.print()
    this.c.print()
    this.d.print()
    this.x.print()
    this.optimizer = tf.train.sgd(.5)
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
  async train(xs, ys, n) {
    for (let i = 0; i < n; i++) {
      this.optimizer.minimize(() => {
        const pred = this.predict(xs)
        return this.loss(pred, ys)
      })

      await tf.nextFrame()
    }
  } 
  
  generateData(numPoints, coeff, sigma = 0.04, noise=true) {
		// https://github.com/tensorflow/tfjs-examples/blob/master/polynomial-regression-core/data.js
		return tf.tidy(() => {
      const [a, b, c, d] = [
				tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
				tf.scalar(coeff.d)
			];

			const xs = noise ? tf.randomUniform([numPoints], -1, 1) : tf.tensor([...Array(21).keys()].map(n => (n - 10)/10))

			// Generate polynomial data
			const three = tf.scalar(3, 'int32')
			const ys = noise ? 
        a.mul(xs.pow(three))
				.add(b.mul(xs.square()))
				.add(c.mul(xs))
				.add(d)
				// Add random noise to the generated data
				// to make the problem a bit more interesting
        .add(tf.randomNormal([numPoints], 0, sigma))
          :
        a.mul(xs.pow(three))
				.add(b.mul(xs.square()))
				.add(c.mul(xs))
				.add(d)

			// Normalize the y values to the range 0 to 1.
			const ymin = ys.min()
			const ymax = ys.max()
			const yrange = ymax.sub(ymin)
			const ysNormalized = ys.sub(ymin).div(yrange)

			return {
				xs, 
				ys: ysNormalized
			}
		}) 
  }
}

async function getCoefficients() {
  const fc = new FitCurve()
  const testData = await fc.generateData(100, {a: -.8, b: -.2, c: .9, d: .5})
  const trained = await fc.train(testData.xs, testData.ys, 100)
}

if (isNode) module.exports = FitCurve
