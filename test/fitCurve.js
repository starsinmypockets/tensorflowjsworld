const fitCurve = require('../fitCurve.js')
const assert = require('chai').assert

describe('Tests run', () => {
  it('Should be true', () => {
    assert(true)
	})

  it('Check data', () => {
    assert(true)
    Promise.all([
      fitCurve.a.data(),
      fitCurve.b.data(),
      fitCurve.c.data(),
      fitCurve.d.data()
    ]).then(res => {
      assert(res.length === 4, 'four scalars')
      assert(typeof res[0][0] === 'number', 'scalar value is number')
    })
  })

  it('Check our polynomial function', async () => {
    const {a, b, c, d, x, predict} = fitCurve
    const poly = await predict(a, b, c, d, x).data()
    console.log(poly)
    assert('For given x, return valid y', typeof poly[0] === 'number')
  })
})
