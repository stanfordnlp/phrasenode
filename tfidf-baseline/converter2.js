const fs = require('fs');
const { DIR, QUERY_OUTPUT } = require('./files')

const INPUT_FILE = QUERY_OUTPUT


var dataset = fs.readFileSync(INPUT_FILE).toString().split('\n')
  .map(s => {
    try {
      return JSON.parse(s)
    } catch (err) {
      return null
    }
  })


var currentSite = null;
var currArr = []
for(var i = 0; i < dataset.length; i++) {
  var datum = dataset[i]
  if (currentSite == null) {
    currentSite = datum.webpage
    currArr = []
  }
  if (currentSite == datum.webpage) {
    currArr.push({xid: datum.xid, answers: [
      {phrase: datum.phrase},
      {type:'prediction',xid:datum.prediction}
    ]})
  } else {
    fs.writeFileSync('answers/ans-'+currentSite+'.json', JSON.stringify(currArr))
    currentSite = datum.webpage
    currArr = []
  }
}


var results = dataset.map(o => {
  for(var x = 0; x < o.predictions.length; x++) {
    var pred = o.predictions[x]
    var xid = parseInt(pred.xid)
    if(pred.score > 0 && xid >= o.xid-2 && xid <= o.xid+2) {
      return true
    }
  }
  return false
})

var correct = results.reduce((s,n)=> s+=n,0)
var total = results.length
console.log('correct:', correct, '\ntotal:',total, '\naccuracy', correct/total);
