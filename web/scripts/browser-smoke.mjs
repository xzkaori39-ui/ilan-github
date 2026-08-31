import { spawn } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const chrome = process.env.CHROME_BIN || (process.platform === "darwin"
  ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
  : "/usr/bin/google-chrome");
const base = process.env.BASE_URL || "http://localhost:8080";
const roles = [
  { key: "student", username: "student", password: "student123", expected: "制度问答工作台" },
  { key: "department", username: "jwc_admin", password: "admin123", expected: "部门治理空间" },
  { key: "super", username: "admin", password: "admin123", expected: "超级管理空间" },
];

const delay = (ms) => new Promise(r => setTimeout(r, ms));
async function connect(url) {
  const ws = new WebSocket(url); await new Promise((ok, fail) => { ws.onopen = ok; ws.onerror = fail; });
  let seq = 0; const pending = new Map();
  ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { const {ok,fail}=pending.get(m.id); pending.delete(m.id); m.error?fail(new Error(m.error.message)):ok(m.result); } };
  const call = (method, params={}) => new Promise((ok,fail)=>{const id=++seq;pending.set(id,{ok,fail});ws.send(JSON.stringify({id,method,params}))});
  return { ws, call };
}
async function waitFor(call, expression, timeout=15000) {
  const start=Date.now();
  while(Date.now()-start<timeout){const r=await call("Runtime.evaluate",{expression,returnByValue:true,awaitPromise:true});if(r.result.value)return r.result.value;await delay(250)}
  throw new Error(`timeout: ${expression}`);
}
async function run(role, port) {
  const profile=await mkdtemp(join(tmpdir(),`wenshu-${role.key}-`));
  const proc=spawn(chrome,["--headless=new","--no-sandbox","--disable-gpu","--hide-scrollbars","--no-first-run","--no-default-browser-check",`--remote-debugging-port=${port}`,`--user-data-dir=${profile}`,"--window-size=1440,1000",base],{stdio:"ignore"});
  try {
    let pages; for(let i=0;i<40;i++){try{pages=await fetch(`http://127.0.0.1:${port}/json`).then(r=>r.json());if(pages.length)break}catch{}await delay(250)}
    const page=pages.find(p=>p.type==="page"); const {ws,call}=await connect(page.webSocketDebuggerUrl);
    await call("Page.enable"); await call("Runtime.enable");
    await waitFor(call,`document.querySelector('input[placeholder=\"请输入账号\"]') !== null`);
    const loginPage=(await call("Runtime.evaluate",{expression:"({text:document.body.innerText,title:document.title})",returnByValue:true})).result.value;
    if(!loginPage.text.includes("i兰") || loginPage.title !== "i兰 · 校园知识服务助手") throw new Error("i兰品牌未显示在登录页");
    await call("Runtime.evaluate",{expression:`(()=>{const set=(el,v)=>{const s=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;s.call(el,v);el.dispatchEvent(new Event('input',{bubbles:true}))};const a=document.querySelector('input[placeholder=\"请输入账号\"]');const p=document.querySelector('input[type=password]');set(a,${JSON.stringify(role.username)});set(p,${JSON.stringify(role.password)});document.querySelector('form').requestSubmit();})()`});
    await waitFor(call,`document.body.innerText.includes(${JSON.stringify(role.expected)})`,20000);
    const errors=await call("Runtime.evaluate",{expression:`({title:document.title,text:document.body.innerText.slice(0,5000),w:document.documentElement.scrollWidth,vw:document.documentElement.clientWidth})`,returnByValue:true});
    if(errors.result.value.w>errors.result.value.vw+2) throw new Error(`horizontal overflow ${errors.result.value.w}/${errors.result.value.vw}`);
    const checked=[];
    if(role.key!=="student"){
      const nav=[
        ["态势总览","Loop 渐进退出"],["知识资产","部门与制度事实治理"],
        ["可信审核","审核中心"],["进化 Loop","反馈如何改变下一轮行为"],
        ["记忆与实验","制度事实平面"],["Agent 网络","部门子 Agent 可视化"],
        ["RAG 评测","离线回放"],
      ];
      for(const [label,expected] of nav){
        await call("Runtime.evaluate",{expression:`[...document.querySelectorAll('button')].find(b=>b.innerText.includes(${JSON.stringify(label)}))?.click()`});
        await waitFor(call,`document.body.innerText.includes(${JSON.stringify(expected)})`);checked.push(label);
        if(role.key==="super"){const tabShot=await call("Page.captureScreenshot",{format:"png",captureBeyondViewport:false});await writeFile(`/tmp/wenshu-super-${label.replace(/\s/g,'-')}.png`,Buffer.from(tabShot.data,"base64"));}
      }
      const text=(await call("Runtime.evaluate",{expression:`document.body.innerText`,returnByValue:true})).result.value;
      await call("Runtime.evaluate",{expression:`[...document.querySelectorAll('button')].find(b=>b.innerText.includes('知识资产'))?.click()`});await delay(300);
      const assetText=(await call("Runtime.evaluate",{expression:`document.body.innerText`,returnByValue:true})).result.value;
      await call("Runtime.evaluate",{expression:`[...document.querySelectorAll('button')].find(b=>b.innerText.includes('进化 Loop'))?.click()`});await delay(300);
      const loopText=(await call("Runtime.evaluate",{expression:`document.body.innerText`,returnByValue:true})).result.value;
      if(role.key==="department" && (assetText.includes("新建部门") || loopText.includes("触发新一轮 Loop"))) throw new Error("department admin sees global mutations");
      if(role.key==="super" && (!assetText.includes("新建部门") || !loopText.includes("触发新一轮 Loop"))) throw new Error("super admin missing global mutations");
      await call("Runtime.evaluate",{expression:`[...document.querySelectorAll('button')].find(b=>b.innerText.includes('记忆与实验'))?.click()`}); await delay(1000);
      await waitFor(call,`document.body.innerText.includes('制度事实平面')`);
      await call("Runtime.evaluate",{expression:`[...document.querySelectorAll('button')].find(b=>b.innerText.includes('RAG 评测'))?.click()`}); await delay(500);
      await waitFor(call,`document.body.innerText.includes('Faithfulness 需 Judge，首期未启用') && document.body.innerText.includes('配置快照')`);
      if(role.key==="super") {
        await waitFor(call,`document.body.innerText.includes('Recall@5') && document.body.innerText.includes('上游未提供')`);
        await call("Runtime.evaluate",{expression:`[...document.querySelectorAll('button')].find(b=>b.innerText.includes('知识图谱'))?.click()`});
        await waitFor(call,`Boolean(document.body.innerText.includes('局部关系网络') && document.querySelector('svg[aria-label="可缩放知识图谱关系网络"]'))`,20000);
        const initialTransform=(await call("Runtime.evaluate",{expression:`document.querySelector('svg g[transform]')?.getAttribute('transform')`,returnByValue:true})).result.value;
        await call("Runtime.evaluate",{expression:`[...document.querySelectorAll('button')].find(b=>b.innerText==='＋')?.click()`});
        await delay(200);
        const graphState=(await call("Runtime.evaluate",{expression:`({controls:document.querySelectorAll('button').length,graphButtons:[...document.querySelectorAll('button')].filter(b=>['＋','－','重置视图'].includes(b.innerText)).map(b=>b.innerText),error:document.body.innerText.includes('图增强不可用')})`,returnByValue:true})).result.value;
        if(graphState.graphButtons.length!==3) throw new Error(`graph controls missing: ${JSON.stringify(graphState)}`);
        const zoomedTransform=(await call("Runtime.evaluate",{expression:`document.querySelector('svg g[transform]')?.getAttribute('transform')`,returnByValue:true})).result.value;
        if(!initialTransform || initialTransform===zoomedTransform) throw new Error(`graph zoom did not update: ${initialTransform} -> ${zoomedTransform}`);
        checked.push("知识图谱可视化");
      }
      if(role.key==="super" && process.env.RUN_EVALUATION==="1"){
        await call("Runtime.evaluate",{expression:`[...document.querySelectorAll('button')].find(b=>b.innerText==='运行离线回放')?.click()`});
        await waitFor(call,`document.body.innerText.includes('离线回放运行中')`);
        await waitFor(call,`document.body.innerText.includes('离线建议：')`,600000);
        checked.push("真实离线回放");
      }
    } else {
      const welcomeShot=await call("Page.captureScreenshot",{format:"png",captureBeyondViewport:false});
      await writeFile("/tmp/wenshu-student-welcome.png",Buffer.from(welcomeShot.data,"base64"));
      await call("Runtime.evaluate",{expression:`[...document.querySelectorAll('button')].find(b=>b.innerText.includes('我的长期记忆'))?.click()`}); await delay(500);
      await waitFor(call,`document.body.innerText.includes('我的长期记忆')`);
      await call("Runtime.evaluate",{expression:`[...document.querySelectorAll('section button')].find(b=>b.innerText==='×')?.click()`});
      const clicked=(await call("Runtime.evaluate",{expression:`(()=>{const x=document.querySelector('div[class*=sessionItem]');if(x){x.click();return true}return false})()`,returnByValue:true})).result.value;
      if(clicked){await waitFor(call,`document.body.innerText.includes('查看 ') && document.body.innerText.includes('条制度依据')`,15000);checked.push("历史会话与引用");}
    }
    const shot=await call("Page.captureScreenshot",{format:"png",captureBeyondViewport:false});await writeFile(`/tmp/wenshu-${role.key}.png`,Buffer.from(shot.data,"base64"));
    ws.close(); return {role:role.key, title:errors.result.value.title, overflow:false, checked, screenshot:`/tmp/wenshu-${role.key}.png`};
  } finally { proc.kill("SIGTERM"); await delay(300); await rm(profile,{recursive:true,force:true}); }
}
const selectedRoles = process.env.SMOKE_ROLE ? roles.filter(role => role.key === process.env.SMOKE_ROLE) : roles;
if (selectedRoles.length === 0) throw new Error(`unknown SMOKE_ROLE: ${process.env.SMOKE_ROLE}`);
const results=[];for(let i=0;i<selectedRoles.length;i++)results.push(await run(selectedRoles[i],9333+i));console.log(JSON.stringify(results,null,2));
