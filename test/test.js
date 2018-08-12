const app = require('../app.js')
const assert = require('chai').assert
let simpleScore1, simpleScore2

function fitSimpleModel(model, cb) {
    model.fit(app.xs, app.ys, {
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
    assert(typeof app.simpleModel === 'object')
    assert(app.simpleModel.model.loss === 'meanSquaredError')
  })

  it('Evaluate before training', async () => {
    const t = await app.simpleModel.evaluate(app.xs, app.ys, {verbose: true})
    t.data().then(res => {
      // evaluate loss before training
      simpleScore1 = res[0]
      assert(res[0])
      assert(typeof res[0] === 'number')
    })
  })

  it('Evaluate model', () => {
    fitSimpleModel(app.simpleModel, async res => {
      const t = await app.simpleModel.evaluate(app.xs, app.ys, {verbose: true})
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
