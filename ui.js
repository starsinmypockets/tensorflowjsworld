async function getCoefficients() {
  function randCoefficients() {
    return { a: rand(), b: rand(), c: rand(), d: rand() }

    function rand() {
      let n = Math.random()
      return (Math.random() >= .5) ? -n : n
    }
  }

  const fc = new FitCurve()
  
  // generate initial data series
  const testData = await fc.generateData(100, {a: 2.8, b: -.2, c: 1.9, d: .5}, noise=true)
  console.log(testData)
  const underlyingModel= await fc.generateData(10, {a: 2.8, b: -.2, c: 1.9, d: .5}, 0, noise = false)
  const initialGuess = await fc.generateData(10, randCoefficients(), 0, noise = false)

  const xs = await testData.xs.data()
  
  // prediction series
  const p0 = await fc.predict(testData.xs).data()
  await fc.train(testData.xs, testData.ys, 1)
  const p1 = await fc.predict(testData.xs).data()
  await fc.train(testData.xs, testData.ys, 5)
  const p2 = await fc.predict(testData.xs).data()
  await fc.train(testData.xs, testData.ys, 100)
  const p3 = await fc.predict(testData.xs).data()
  await fc.train(testData.xs, testData.ys, 400)
  const p4 = await fc.predict(testData.xs).data()
  
  
  const div =  document.getElementById('tester')
  const xs1 = await testData.xs.data()

  Plotly.plot(div, [
    {
      x: xs1,
      y: await testData.ys.data(),
      name: 'Data',
      mode: 'markers',
      type: 'scatter'
    },
    {
      x: await underlyingModel.xs.data(),
      y: await underlyingModel.ys.data(),
      name: 'Model',
      mode: 'lines',
      type: 'scatter'
    },
    {
      x: await initialGuess.xs.data(),
      y: await initialGuess.ys.data(),
      name: 'Guess 1',
      mode: 'markers',
      type: 'scatter'
    }, 
    {
      x: xs1,
      y: p0,
      name: 'Guess 2',
      mode: 'markers',
      type: 'scatter'
    }, 
    {
      x: xs1,
      y: p1,
      name: 'Guess 3',
      mode: 'markers',
      type: 'scatter'
    },
    {
      x: xs1,
      y: p2,
      name: 'Guess 4',
      mode: 'markers',
      type: 'scatter'
    },
    {
      x: xs1,
      y: p3,
      name: 'Guess 5',
      mode: 'markers',
      type: 'scatter'
    },
    {
      x: xs1,
      y: p4,
      name: 'Guess 6',
      mode: 'markers',
      type: 'scatter'
    }
  ])
}

function drawPlot(series) {
  const div =  document.getElementById('tester');

  Plotly.plot(div, [
  ], { margin: { t: 0 } } );

  console.log( Plotly.BUILD );
}
