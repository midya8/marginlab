'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn}=require('node:child_process');
const {waitForDebugger}=require('./chrome_support.js');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
test('real Chrome exact certificates: all four source matches, verification, binding control, clipboard/download and mobile', {timeout:90000}, async()=>{
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'marginlab-browser-'));
 const browser=spawn(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-extensions','--disable-dev-shm-usage','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});let ws,startupError,diagnostics='';browser.on('error',e=>startupError=e);browser.stderr.on('data',d=>diagnostics=(diagnostics+d).slice(-4000));
 try{
  for(let i=0;i<1200&&!startupError&&browser.exitCode===null&&!fs.existsSync(path.join(profile,'DevToolsActivePort'));i++)await delay(25);
  assert.ok(fs.existsSync(path.join(profile,'DevToolsActivePort')),`Chrome debugger did not start: ${startupError?.message||diagnostics||browser.exitCode}`);const port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];
  const tab=await waitForDebugger(`http://127.0.0.1:${port}/json/list`,{alive:()=>browser.exitCode===null&&!startupError});ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((res,rej)=>{ws.onopen=res;ws.onerror=rej;});
  let seq=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const {res,rej}=pending.get(m.id);pending.delete(m.id);m.error?rej(new Error(JSON.stringify(m.error))):res(m.result);}};
  const call=(method,params={})=>new Promise((res,rej)=>{const id=++seq;pending.set(id,{res,rej});ws.send(JSON.stringify({id,method,params}));});
  const js=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  await call('Page.enable');await call('Page.navigate',{url:process.env.MARGIN_CERT_URL||'file://'+path.resolve(__dirname,'../docs/certificate.html')});
  for(let i=0;i<250;i++){if(await js(`Boolean(globalThis.MARGIN_CERT_APP?.getProof())`))break;await delay(20);}
  assert.equal(await js(`document.getElementById('proof-error').textContent`),'');
  for(const id of ['linear-0.01','linear-0.1','linear-1','linear-10']){
    await js(`document.getElementById('proof-model').value='${id}';document.getElementById('proof-model').dispatchEvent(new Event('change'))`);
    for(let i=0;i<250;i++){if(await js(`Boolean(MARGIN_CERT_APP.getProof())`))break;await delay(20);}
    assert.equal(await js(`document.getElementById('proof-error').textContent`),'');assert.match(await js(`document.getElementById('proof-status').textContent`),/Verified with exact/);assert.equal(await js(`MARGIN_CERT_APP.getProof().problem.x.length`),341);
  }
  assert.equal(await js(`(()=>{const p=structuredClone(MARGIN_CERTIFICATES['linear-1']);p.problem.x[0][0]=42;return MarginCertificate.check(p).valid&&!MARGIN_CERT_APP.matches(p,MARGIN_DATA.models.find(m=>m.id==='linear-1'));})()`),true);
  await js(`document.getElementById('proof-model').focus()`);await call('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowUp',code:'ArrowUp',windowsVirtualKeyCode:38});await call('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowUp',code:'ArrowUp',windowsVirtualKeyCode:38});
  for(let i=0;i<250;i++){if(await js(`Boolean(MARGIN_CERT_APP.getProof())`))break;await delay(20);}assert.equal(await js(`document.getElementById('proof-model').value`),'linear-1');
  await js(`globalThis.originalProof=structuredClone(MARGIN_CERTIFICATES['linear-1']);MARGIN_CERTIFICATES['linear-1'].problem.x[0][0]=42;MARGIN_CERT_APP.update()`);await delay(500);assert.equal(await js(`MARGIN_CERT_APP.getProof()`),null);assert.match(await js(`document.getElementById('proof-error').textContent`),/does not match/);assert.equal(await js(`document.getElementById('proof-gap').textContent`),'-');await js(`document.getElementById('proof-copy').click()`);assert.match(await js(`document.getElementById('proof-copy-status').textContent`),/Verify the proof/);
  await js(`MARGIN_CERTIFICATES['linear-1']=globalThis.originalProof;MARGIN_CERT_APP.update()`);for(let i=0;i<250;i++){if(await js(`Boolean(MARGIN_CERT_APP.getProof())`))break;await delay(20);}assert.equal(await js(`document.getElementById('proof-error').textContent`),'');
  await js(`Object.defineProperty(navigator,'clipboard',{value:{writeText:async()=>{throw Error('denied')}},configurable:true});document.getElementById('proof-copy').click()`);await delay(10);assert.match(await js(`document.getElementById('proof-copy-status').textContent`),/Copy was blocked/);
  await js(`Object.defineProperty(navigator,'clipboard',{value:{writeText:async s=>globalThis.copied=s},configurable:true});document.getElementById('proof-copy').click()`);await delay(10);assert.equal(await js(`JSON.parse(globalThis.copied).version`),1);
  await js(`HTMLAnchorElement.prototype.click=function(){globalThis.downloaded={name:this.download,url:this.href}};document.getElementById('proof-download').click()`);assert.equal(await js(`globalThis.downloaded.name`),'marginlab-linear-proof.json');assert.equal(await js(`fetch(globalThis.downloaded.url).then(r=>r.json()).then(p=>MarginCertificate.check(p).valid)`),true);
  for(const width of [320,390,1280]){await js(`for(const e of document.querySelectorAll('details'))e.open=true`);await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});await delay(20);assert.ok(await js(`document.documentElement.scrollWidth<=innerWidth`),'overflow at '+width);assert.ok(await js(`Array.from(document.querySelectorAll('button,select,summary')).every(e=>e.getBoundingClientRect().height>=44)`),'touch at '+width);}
  await js(`for(const e of document.querySelectorAll('details'))e.open=false;window.scrollTo(0,0)`);
  if(process.env.MARGIN_CERT_SCREENSHOT){const width=Number(process.env.MARGIN_CERT_SCREENSHOT_WIDTH||1280);await call('Emulation.setDeviceMetricsOverride',{width,height:1200,deviceScaleFactor:1,mobile:width<700});await delay(30);const metrics=await call('Page.getLayoutMetrics'),image=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width,height:metrics.cssContentSize.height,scale:1}});fs.writeFileSync(process.env.MARGIN_CERT_SCREENSHOT,Buffer.from(image.data,'base64'));}
 }finally{if(ws)ws.close();if(browser.pid&&browser.exitCode===null){browser.kill('SIGTERM');await new Promise(resolve=>browser.once('exit',resolve));}await fs.promises.rm(profile,{recursive:true,force:true,maxRetries:10,retryDelay:200});}
});
