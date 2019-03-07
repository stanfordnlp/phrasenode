const cheerio = require('cheerio');
const fs = require('fs');
const zlib = require('zlib')
const { DIR, DATA_FILE, PAGE_PATH, UNSTEMMED_PREPROCESS_OUTPUT } = require('./files')


const Tokenizer = require('tokenize-text');
var tokenizer = new Tokenizer();
var tokenize = s => tokenizer.words()(s).map(m => m.value.toLowerCase()).join(' ')




var dataset = fs.readFileSync(DATA_FILE).toString().split('\n')
  .map(s => {
    try {
      return JSON.parse(s)
    } catch (err) {
      return null
    }
  })
  .filter(s=>!!s)
// .filter(s=>s.webpage === 'about.com')


var files = dataset.reduce((o, d) => {
  if(d) o[d.webpage] = d.version
  return o
}, {})


var renderMap = function(visibility) {
  visibility = visibility.info
  return visibility.reduce((o, m) => {
    if (m.attributes) {
      var xid = m.attributes['data-xid']
      if(xid) {
        o[xid] = m
      }
    }
    return o
  }, {})
}


var yuckimperative = []
for(var webpage in files) {
  const version = files[webpage]
  yuckimperative.push([webpage, version])
}

var cache = {}

var documents =
  yuckimperative.map(result => {
    var [webpage, version] = result
    var key = webpage+version
    if(key in cache) {
      var html = cache[key]
      var $ = cache[key+'$']
      var renderInfo = cache[key+'renderInfo']
    } else {
      cache = {}
      var html = fs.readFileSync(PAGE_PATH(version, webpage)).toString()
      var $ = cheerio.load(html)
      console.log(version, webpage, cache)
      var visibility = JSON.parse(zlib.gunzipSync(fs.readFileSync(DIR+'infos/'+version+'/info-'+webpage+'.gz')).toString())
      var renderInfo = renderMap(visibility)

      cache[key] = html
      cache[key+'$'] = $
      cache[key+'renderInfo'] = renderInfo
    }

    var whitelistElements = ['a', 'span', 'button']
    var blacklistElements = ['p', 'style', 'script', 'code', 'pre', 'small', 'center']
    return $('body :not(script)')
      .filter((i, elem) => {
        var tag = elem.tagName.toLowerCase();
        var xid = $(elem).attr('data-xid')
        var render = renderInfo[xid]
        var rendered = (render && 'hidden' in render // && 'topLevel' in render
        && 'width' in render && 'height' in render) ?
          // render.width fails if width is 0. same for height
          (render.width && render.height) &&
          (render.hidden === false /*&& render.topLevel === true*/) :
          true;

        var keep = rendered && (blacklistElements.indexOf(tag) < 0) && (whitelistElements.indexOf(tag) >= 0 || $(elem).children().length === 0)

        return keep
      })
      .map((i, el) => {
        var attrs = []
        for (var nm in el.attribs) {
          if(nm &&
            nm==='class' ||
            nm==='id' ||
            nm==='value' ||
            nm==='placeholder' ||
            nm==='name' ||
            nm.includes('aria') ||
            nm.includes('label') ||
            nm.includes('tooltip') ||
            nm.includes('src') ||
            nm.includes('href')) {
              attrs.push(el.attribs[nm])
            }
        }
        var tagName = el.tagName.toLowerCase()
        el = $(el)
        return {
          attrs: tokenize(attrs.join(',')),
          // text: (',' +el.text().trim()).repeat(3),
          text: tokenize(el.text().trim()),
          xid: el.attr('data-xid'),
          tag: tagName,
          webpage: webpage
        }
      })
      .get()
  })



fs.writeFileSync(UNSTEMMED_PREPROCESS_OUTPUT,
  documents.map(d=>JSON.stringify(d)).join('\n'))


