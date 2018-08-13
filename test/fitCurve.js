const FitCurve = require('../fitCurve.js')
const assert = require('chai').assert
const fc = new FitCurve()
const tf = require('@tensorflow/tfjs')

describe('Tests run', () => {
  it('Should be true', () => {
    assert(true)
	})

  it('Check initial data', async () => {
    const res = await Promise.all([
      fc.a.data(),
      fc.b.data(),
      fc.c.data(),
      fc.d.data()
    ])

    assert(res.length === 4)
    assert(typeof res[0][0] === 'number')
  })

  it('Generate random data', async () => {
    const coeff = {a: -.8, b: -.2, c: .9, d: .5}
    const n = 100
    const data = await fc.generateData(n, coeff)
    const xs = await data.xs.data()
    const ys = await data.ys.data()
    
    assert(xs.length === n)
    assert(typeof xs[50] === 'number')
    assert(ys.length === n)
    assert(typeof ys[50] === 'number')
  })

  it('Check our polynomial function', async () => {
    const x = await tf.variable(tf.scalar(Math.random()))
    const poly = await fc.predict(x).data()
    console.log(poly)
    assert('For given x, return valid y', typeof poly[0] === 'number')
  })
  
  it('Check loss function returns valid result', async () => {
    const trainingData = await fc.generateData(100, {a: -.8, b: -.2, c: .9, d: .5})
    const x = await tf.variable(tf.scalar(Math.random()))
    const l = await fc.loss(x, trainingData.ys)
    const lossResult = await l.data()
    
    assert(lossResult > 0 && lossResult < 1)
  })

  it('Throw it some training data', async () => {
    const testData = await fc.generateData(100, {a: -.8, b: -.2, c: .9, d: .5})

    console.log(await testData.xs.data(), testData.ys.data())

    const trained = await fc.train(testData.xs, testData.ys, 100)
    assert(true)
  })
})
