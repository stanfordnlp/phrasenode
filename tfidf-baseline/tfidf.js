const natural = require('natural');
const fs = require('fs')
const tokenize = require('./utils').tokenize
const { DIR, DATA_FILE } = require('./files')



var documents = fs.readFileSync(DIR+'all-nodes.jsonl')
  .toString().split('\n').map(s=>JSON.parse(s))


var dataset = fs.readFileSync(DATA_FILE).toString().split('\n')
  .map(s => {
    try {
      return JSON.parse(s)
    } catch (err) {
      return null
    }
  })
  .filter(s=>!!s)
//.filter(s=>s.webpage === 'about.com')


var stopwords = [
  /*'about', 'above', 'after', 'again', 'all', 'also', 'am', 'an', 'and', 'another',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'but', 'by', 'came', 'can', 'cannot', 'come', 'could', 'did',
    'do', 'does', 'doing', 'during', 'each', 'few', 'for', 'from', 'further', 'get',
    'got', 'has', 'had', 'he', 'have', 'her', 'here', 'him', 'himself', 'his', 'how',
    'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'like', 'make', 'many', 'me',
    'might', 'more', 'most', 'much', 'must', 'my', 'myself', 'never', 'now', 'of', 'on',
    'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'said', 'same', 'see', 'should', 'since', 'so', 'some', 'still', 'such', 'take', 'than',
    'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
    'way', 'we', 'well', 'were', 'what', 'where', 'when', 'which', 'while', 'who',
    'whom', 'with', 'would', 'why', 'you', 'your', 'yours', 'yourself',*/
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '$', '1',
    '2', '3', '4', '5', '6', '7', '8', '9', '0', '_'];

var TfIdf = natural.TfIdf;
// TfIdf.setStopwords(stopwords)
var tfidf = new TfIdf();


var tfidfMap = {}
var counter = 0

for(var i = 0; i < documents.length; i++) {
  var doc = documents[i];
  doc.map(d => {
    //var words = d.attrs + ',' + d.text
    var words = d.text
    tfidf.addDocument(words)
    tfidfMap[counter] = {
      xid: d.xid,
      file: d.webpage
    }
    counter += 1
  })
}




for(var i = 0; i < dataset.length; i++) {
  var data = dataset[i];
  if(data) {
    var words = tokenize((data.phrase+' ').repeat(3) + data.attrs)
    tfidf.addDocument(words)
  }
}


fs.writeFileSync(DIR+'tfidf.json', JSON.stringify(tfidf))
fs.writeFileSync(DIR+'tfidf-map.json', JSON.stringify(tfidfMap))

