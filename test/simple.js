const simple = require('../simple.js')
const assert = require('chai').assert
let simpleScore1, simpleScore2

function fitSimpleModel(model, cb) {
    model.fit(simple.xs,simple.ys, {
    epochs: 100,
    callbacks: {
      onEpochEnd: (epoch, log) => {
        console.log(`Epoch ${epoch}: loss = ${log.loss}`)
      },
      onTrainEnd: (a,b,c,d,e,f,g) => {
        cb()
      }
    }
  })
}

describe('Tests run', function () {
  it('Should be true', function () {
    assert(true)
	})
})

describe('Simple model behaves', () => {
  it('App model should be a thing', () => {
    assert(typeof simple.model === 'object')
    assert(simple.model.model.loss === 'meanSquaredError')
  })

  it('Evaluate before training', async () => {
    const t = await simple.model.evaluate(simple.xs, simple.ys, {verbose: true})
    t.data().then(res => {
      // evaluate loss before training
      simpleScore1 = res[0]
      assert(res[0])
      assert(typeof res[0] === 'number')
    })
  })

  it('Evaluate model', () => {
    fitSimpleModel(simple.model, async res => {
      const t = await simple.model.evaluate(simple.xs, simple.ys, {verbose: true})
      t.data().then(res => {
        // evaluate loss after training
        simpleScore2 = res[0]
        assert(res[0])
        assert(typeof res[0] === 'number')
        assert(simpleScore1 > simpleScore2, 'loss after fitting should be lower')
      })
    })
  })
})
