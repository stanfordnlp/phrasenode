const natural = require('natural');
const fs = require('fs')
var tokenize = require('./utils').tokenize
const { DIR, DATA_FILE, QUERY_OUTPUT } = require('./files')



var TfIdf = natural.TfIdf;
var s = fs.readFileSync(DIR+'tfidf.json').toString()
var tfidf = new TfIdf(JSON.parse(s));

var tfidfMap = JSON.parse(fs.readFileSync(DIR+'tfidf-map.json').toString())


function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

const mapIdx = (array) => array.map((x, i) => [x, i])


var baseline = (o) => {
  var query = tokenize(o.phrase)
  var scores = mapIdx(tfidf.tfidfs(query))

  var results = scores
    .map(s=>[s[0], s[1], tfidfMap[''+s[1]]])
    .filter(arr=>{
      var [score, idx, rank] = arr
      return (rank && rank.file===o.webpage)
    })
    .sort((a,b)=>b[0]-a[0])
    .splice(0,5)
    .map(arr=>{
      var [score, idx, rank] = arr
      return { xid: rank.xid, score }
    })
  console.log(query)
  console.log(results)

  // return results[0].xid

  // return results[0][0]
  return results
}




var dataset = fs.readFileSync(DATA_FILE).toString().split('\n')
  .map(s=>{
    try {
      return JSON.parse(s)
    } catch (err) {
      return null
    }
  })
  .filter(s=>!!s)
// .filter(s=>s.webpage === 'about.com')


var results = dataset
  .map(o=>{
    var predictions = baseline(o)
    if(predictions[0]) {
      var match = predictions[0].xid === (''+o.xid)
      console.log(`${match}\t${predictions[0].xid}\t${o.xid}`)
    }
    o.predictions = predictions
    return o
  })

var check = (actual, predicted) => {
  return predicted === actual
  // return (predicted >= actual-2 && predicted <= actual+2)
}

var correct = results.reduce((s, n) => s+=check(n.xid, parseInt(n.predictions[0].xid)), 0)
var total = results.length

console.log('correct:', correct, '\ntotal:',total, '\naccuracy', correct/total);

fs.writeFileSync(QUERY_OUTPUT, results.map(o=>JSON.stringify(o)).join('\n'))


