'use strict';

async function getCoefficients() {
  function randCoefficients() {
    return { a: rand(), b: rand(), c: rand(), d: rand() }

    function rand() {
      let n = Math.random()
      return (Math.random() >= .5) ? -n : n
    }
  }

  const fc = new FitCurve()
  let frame = 1
  
  // generate initial data series
  const testData = await fc.generateData(100, {a: 2.8, b: -.2, c: 1.9, d: .5}, true)
  console.log(testData)
  const underlyingModel= await fc.generateData(10, {a: 2.8, b: -.2, c: 1.9, d: .5}, 0, false)
  const initialGuess = await fc.generateData(10, randCoefficients(), 0, false)

  const xs = await testData.xs.data()
  
  // prediction series
  const p0 = await fc.predict(testData.xs).data()
  await fc.train(testData.xs, testData.ys, 1)
  const p1 = await fc.predict(testData.xs).data()
  await fc.train(testData.xs, testData.ys, 5)
  const p2 = await fc.predict(testData.xs).data()
  await fc.train(testData.xs, testData.ys, 100)
  const p3 = await fc.predict(testData.xs).data()
  
  const div =  document.getElementById('tester')
  const xs1 = await testData.xs.data()
  
  /** TODO add New Model button **/
  /** TODO add play pause button **/
  /** TODO - animate: **/
  // WHILE "PLAY":
  // 0. Switch PLAY to PAUSE
  // 1. Show training data and model
  // 2. Hide Training data
  // 3. Show first guess
  // 4. Show second guess
  // 6. ...
  // 7. Repeat

  const allSeries = [
    [ 
      {
        x: xs1,
        y: await testData.ys.data(),
        name: 'Data',
        mode: 'markers',
        type: 'scatter'
      },
    ],
    [
      {
        x: await underlyingModel.xs.data(),
        y: await underlyingModel.ys.data(),
        name: 'Model',
        mode: 'lines',
        type: 'scatter'
      },
    ],
    [
      {
        x: await initialGuess.xs.data(),
        y: await initialGuess.ys.data(),
        name: 'Guess 1',
        mode: 'markers',
        type: 'scatter'
      }, 
    ],
    [
      {
        x: xs1,
        y: p0,
        name: 'Guess 2',
        mode: 'markers',
        type: 'scatter'
      },
    ],
    [
      {
        x: xs1,
        y: p1,
        name: 'Guess 3',
        mode: 'markers',
        type: 'scatter'
      },
    ],
    [
      {
        x: xs1,
        y: p2,
        name: 'Guess 4',
        mode: 'markers',
        type: 'scatter'
      },
    ],
    [
      {
        x: xs1,
        y: p3,
        name: 'Guess 5',
        mode: 'markers',
        type: 'scatter'
      },
    ]
  ]

  const layout = {
    xaxis: {
      range: [ -1, 1 ]
    },
    yaxis: {
      range: [-1, 2]
    },
      title:'Learning a polynomial curve'
  }

  setInterval(() => {
    if (frame == 1) {
      const initData = allSeries[0].concat(allSeries[1])
      Plotly.newPlot(div, initData, layout)
    } else if (frame === allSeries.length) {
      frame = 0 // will increment before next loop
      Plotly.deleteTraces(div, [0,1,2,3,4,5,6])
    } else {
      Plotly.addTraces(div, allSeries[frame])
    }
    frame++
  }, 1500)
}
