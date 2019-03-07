const fs = require('fs');

const INPUT_FILE = 'dataset/results.jsonl'
const OUTPUT_FILE = 'dataset/converted.jsonl'

var dataset = fs.readFileSync(INPUT_FILE).toString().split('\n')
  .map(s => {
    try {
      return JSON.parse(s)
    } catch (err) {
      return null
    }
  })

var currentSite = null;
var processed = []
for(var i = 0; i < dataset.length; i++) {
  var datum = dataset[i]
  if (currentSite == null) {
    currentSite = {
      webpage: datum.webpage,
      answers: [],
      special: []
    }
  }
  if (currentSite.webpage == datum.webpage) {
    currentSite.answers.push(datum)
    currentSite.special.push({type:'prediction',xid:datum.prediction})
  } else {
    processed.push(currentSite)
    currentSite = {
      webpage: datum.webpage,
      answers: [],
      special: []
    }
  }
}


fs.writeFileSync(OUTPUT_FILE, processed.map(o=>JSON.stringify(o)).join('\n'))
