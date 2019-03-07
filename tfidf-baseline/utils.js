const Tokenizer = require('tokenize-text');
const stemmer = require('porter-stemmer').stemmer;


var tokenizer = new Tokenizer();
var tokenize = s => tokenizer.words()(s).map(m => m.value.toLowerCase()).map(s=>stemmer(s)).join(' ')
// https://stackoverflow.com/questions/18769010/regular-expression-to-split-uppercase-uppercase-lowercase-pattern-in-groovy
// var tokenize = s => s.replace(/([A-Z])/g, ' $1').trim()
//   .toLowerCase().split(/[^\wA-Z]/g).filter(e => e.length > 1)
//   .map(s=>stemmer(s))


exports.tokenize = tokenize
