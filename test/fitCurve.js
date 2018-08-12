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
})
