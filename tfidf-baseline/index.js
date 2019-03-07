const cheerio = require('cheerio');
const fs = require('fs');
const stemmer = require('porter-stemmer').stemmer

const DATA_FILE = 'dataset/data/v3.jsonl'
const OUTPUT_FILE = 'dataset/results.jsonl'
const PAGE_PATH = p => 'dataset/pages/v3/'+p+'.html'


var baseline = function(text, html) {
  // TODO check for containment

  // https://stackoverflow.com/questions/5582228/insert-space-before-capital-letters
  // s.replace(/([A-Z])/g, ' $1').trim().split(/[^\w]/g)
  var tokenize = s => s.replace(/([A-Z])/g, ' $1').trim()
    .toLowerCase().split(/[^\wA-Z]/g).reduce((o,e) => {
      if (e === '') return o
      e = stemmer(e)
      if (e in o) { o[e]+=1 }
      else { o[e]=1 }
      return o
    }, {})
  var nkeys = obj => Object.keys(obj).length

  var rankelement = o => {
    var doc = tokenize(o.text + ',' + o.attrs)
    var query = tokenize(text)
    var score = 0
    for (var k in query) {
      if (k in doc) {
        score += Math.min(query[k], doc[k])
      }
    }
    o.score = score;
    return o;
  }

  var weightedrankelement = o => {
    var query = tokenize(text)
    var score = 0

    var doc = tokenize(o.text)
    for (var k in query) {
      if (k in doc) {
        score += 5 * Math.min(query[k], doc[k])
      }
    }
    // var doc = tokenize(o.heavyattrs)
    // for (var k in query) {
    //   if (k in doc) {
    //     score += 5 * Math.min(query[k], doc[k])
    //   }
    // }
    var doc = tokenize(o.attrs)
    for (var k in query) {
      if (k in doc) {
        score += 1 * Math.min(query[k], doc[k])
      }
    }
    o.score = score;
    return o;
  }

  var whitelistElements = ['a', 'span', 'button']
  var blacklistElements = ['p'/*, 'div'*/]
  var $ = cheerio.load(html)
  return $('body :not(script)')
    .filter((i, elem) => {
      var tag = elem.tagName.toLowerCase();
      return (blacklistElements.indexOf(tag) < 0) && (whitelistElements.indexOf(tag) >= 0 || elem.children.length === 0)
    })
    .map((i, el) => {
      var attrs = []
      var heavyattrs = []
      for (var nm in el.attribs) {
        // if(nm.includes('title')
        //   || nm.includes('tooltip')
        //   || nm.includes('label')
        //   || nm === ('class')) {
        //   heavyattrs.push(el.attribs[nm]);
        // }
        attrs.push(el.attribs[nm])
      }
      el = $(el)
      return {
        attrs: attrs.join(','),
        // heavyattrs: heavyattrs.join(','),
        text: el.text().trim(),
        xid: el.attr('data-xid'),
        el: el
      }
    })
    .get()
  //.map(rankelement)
    .map(weightedrankelement)
    .filter(o=>o.score>0)
    .sort((a, b) => b.score - a.score)
    .map(o => o.xid)
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


var results = dataset
  .map(o=>{
    var file = fs.readFileSync(PAGE_PATH(o.webpage))
    // var match = baseline(o.phrase, file).slice(0,5).indexOf(''+o.xid) >= 0
    var prediction = baseline(o.phrase, file)[0]
    var match = prediction === (''+o.xid)
    console.log(match)
    o.prediction = prediction
    return o
  })

var correct = results.reduce((s, n) => s+=(''+n.xid === n.prediction), 0)
var total = results.length

console.log('correct:', correct, '\ntotal:',total, '\naccuracy', correct/total);

fs.writeFileSync(OUTPUT_FILE, results.map(o=>JSON.stringify(o)).join('\n'))

