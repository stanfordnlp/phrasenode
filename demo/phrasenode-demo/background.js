chrome.browserAction.onClicked.addListener(function(tab) {
  console.log(tab.url);
  chrome.tabs.executeScript({
    code: 'runPhraseNode()'
  });
});

chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  $.post('http://localhost:6006/pred', request, function (response) {
    sendResponse(response);
  });
  return true;
});
