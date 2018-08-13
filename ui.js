async function getCoefficients() {
  const fc = new FitCurve()
  const testData = await fc.generateData(100, {a: 2.8, b: -.2, c: 1.9, d: .5}, noise=true)
  const underlyingModel= await fc.generateData(10, {a: 2.8, b: -.2, c: 1.9, d: .5}, 0, noise = false)
  drawPlot(
    await testData.xs.data(),
    await testData.ys.data(),
    await underlyingModel.xs.data(),
    await underlyingModel.ys.data(),
  )
  const trained = await fc.train(testData.xs, testData.ys, 100)
}

function drawPlot(xs, ys, xs1, ys1) {
  const div =  document.getElementById('tester');

  Plotly.plot(div, [
    {
      x: xs,
      y: ys,
      mode: 'markers',
      type: 'scatter'
    },
    {
      x: xs1,
      y: ys1,
      mode: 'lines+markers',
      type: 'scatter'
      
    }
  ], { margin: { t: 0 } } );

  console.log( Plotly.BUILD );
}
