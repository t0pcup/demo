function resetCanvas() {
  // remove all drawed objects
  source.clear();
  map.setLayers([styles[styleSelector.value]]);


  // // remove all imported objects
  // // zIndex = 0 for sat imagery preview
  // // zIndex = 1 for drag&drop events
  // // zIndex = 2 for ready orders' sat imagery
  // // zIndex = 3 for ready orders' predict shape
  // var lst = [];
  //   for (let i = 0, ii = map.getLayers().array_.length; i < ii; ++i) {
  //       if (map.getLayers().array_[i].values_['zIndex'] !== 1 && map.getLayers().array_[i].values_['zIndex'] !== 3) {
  //           lst.push(map.getLayers().array_[i])
  //       }
  //   }
  //   map.setLayers([styles[styleSelector.value]]);
  //   for (let i = 1; i < lst.length; i++) {
  //       map.addLayer(lst[i]);
  //   }
}