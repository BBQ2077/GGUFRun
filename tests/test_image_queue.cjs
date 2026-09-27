const fs=require('node:fs');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const html=fs.readFileSync('assets/image-web.html','utf8');
const script=html.match(/\/\*QUEUE_CORE_START\*\/([\s\S]*?)\/\*QUEUE_CORE_END\*\//);
assert.ok(script,'queue core must be embedded in served HTML');
const context={Date};vm.createContext(context);vm.runInContext(script[1],context);
const Queue=context.QueueController;
function store(){const data=new Map();return{getItem:k=>data.get(k)||null,setItem:(k,v)=>data.set(k,v)}}
function tick(){return new Promise(r=>setImmediate(r))}
(async()=>{
 const memory=store(),events=[],gates=[];
 const sender=body=>{events.push(body.prompt);return new Promise(resolve=>gates.push(resolve))};
 const q=new Queue(memory,sender,()=>{});
 const draft={prompt:'第一張',width:512,steps:15};q.add(draft);draft.prompt='改掉';draft.width=1024;
 q.add({prompt:'第二張',width:768});q.add({prompt:'第三張'});
 assert.equal(q.tasks[0].body.prompt,'第一張','parameters snapshotted when added');
 const run=q.start();await tick();assert.deepEqual(events,['第一張'],'only one request at a time');
 q.add({prompt:'第四張'});q.pause();gates.shift()({saved:true});await run;
 assert.equal(q.tasks.length,3,'completed row disappears immediately');
 assert.equal(q.tasks[0].status,'pending');assert.equal(q.tasks[0].body.prompt,'第二張');
 assert.ok(q.lastDurationMs>=0,'last completion duration remains available to UI');
 const resumed=q.start();await tick();gates.shift()({saved:true});await tick();gates.shift()({saved:true});await tick();gates.shift()({saved:true});await resumed;
 assert.deepEqual(events,['第一張','第二張','第三張','第四張']);
 assert.equal(q.tasks.length,0,'all finished records are removed');
 const afterReload=new Queue(memory,()=>{},()=>{});assert.equal(afterReload.add({prompt:'新一輪'}).id,1,'counter resets when queue empty');
 const old=store();old.setItem('ggufrun.image.queue.v1',JSON.stringify([{id:11,body:{prompt:'old done'},status:'done'},{id:12,body:{prompt:'next'},status:'pending'}]));
 const migrated=new Queue(old,()=>{},()=>{});assert.equal(migrated.tasks.length,1);assert.equal(migrated.tasks[0].body.prompt,'next');
 assert.equal(migrated.tasks[0].status,'pending','unfinished work survives refresh');
 const bad=new Queue(store(),async body=>{if(body.prompt==='fail')throw Error('bad request');return{saved:true}},()=>{});
 bad.add({prompt:'fail'});bad.add({prompt:'after'});await bad.start();
 assert.equal(bad.tasks[0].status,'failed');assert.equal(bad.tasks[1].status,'pending');bad.retry(bad.tasks[0].id);assert.equal(bad.tasks[0].status,'pending');
 const notSaved=new Queue(store(),async()=>({images:['ok']}),()=>{});notSaved.add({prompt:'must save'});await notSaved.start();
 assert.equal(notSaved.tasks[0].status,'failed','generation must not be done if save fails');
 const interrupted=store();interrupted.setItem('ggufrun.image.queue.v1',JSON.stringify([{id:10,body:{prompt:'was running'},status:'running'}]));
 const q2=new Queue(interrupted,()=>{},()=>{});assert.equal(q2.tasks[0].status,'interrupted');
 console.log('queue tests passed: sequential, pause/resume, completed hide, reset counter, migration, persistence, retry');
})().catch(e=>{console.error(e);process.exitCode=1});
