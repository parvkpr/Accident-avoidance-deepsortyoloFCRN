var nem = require("nem-sdk").default;
var bodyparser=require('body-parser');
var express = require('express'),

  app = express(),
  port = process.env.PORT || 3000;

app.listen(port);
app.use(bodyparser.json());
app.post('/data',function(req,res){
  console.log(req.body)
var w=req.body.data
// Include the library

function Hello(w) {
    
try{
// Create an NIS endpoint object
var endpoint = nem.model.objects.create("endpoint")(nem.model.nodes.defaultMijin, nem.model.nodes.mijinPort);


// Create a common object holding key
var common = nem.model.objects.create("common")("", "45c854da62576ea70a7aed7884863b78a027beaacf65991b1c7430bf079d8720");

// Create an un-prepared transfer transaction object
var transferTransaction = nem.model.objects.create("transferTransaction")("MAI2LG3IO4A6ZGFHDP64IDSJXO7P7BCQRLFCG6ES", 0, w);

// Prepare the transfer transaction object
var transactionEntity = nem.model.transactions.prepare("transferTransaction")(common, transferTransaction, nem.model.network.data.mijin.id);

// Serialize transfer transaction and announce
var he= nem.model.transactions.send(common, transactionEntity, endpoint);
console.log(he)
}
catch(err) {
    console.log(err)}
}
Hello(w)
res.send({
"message":"done"
})
})
console.log('todo list RESTful API server started on: ' + port);

