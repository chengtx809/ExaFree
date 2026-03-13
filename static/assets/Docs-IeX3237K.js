import{d as l,c as r,a as o,F as u,r as i,y as d,i as x,e as s,n as c,t as p}from"./index-BN3xPRAS.js";const m={class:"space-y-6"},b={class:"rounded-3xl border border-border bg-card p-6"},f={class:"mt-6 flex rounded-full border border-border bg-muted/30 p-1 text-xs"},q=["onClick"],v={key:0,class:"mt-6 space-y-4 text-sm text-foreground"},g={key:1,class:"mt-6 space-y-3 text-xs text-muted-foreground leading-relaxed"},k=l({__name:"Docs",setup(_){const a=x("api"),n=[{id:"api",label:"API 文档"},{id:"disclaimer",label:"使用声明"}];return(y,e)=>(s(),r("div",m,[o("section",b,[e[2]||(e[2]=o("div",{class:"space-y-1"},[o("p",{class:"text-base font-semibold text-foreground"},"帮助中心"),o("p",{class:"text-xs text-muted-foreground"},"Exa API 池化与常见问题")],-1)),o("div",f,[(s(),r(u,null,i(n,t=>o("button",{key:t.id,class:c(["flex-1 rounded-full px-4 py-2 font-medium transition-colors",a.value===t.id?"bg-foreground text-background":"text-muted-foreground hover:text-foreground"]),onClick:h=>a.value=t.id},p(t.label),11,q)),64))]),a.value==="api"?(s(),r("div",v,[...e[0]||(e[0]=[d(`<div class="space-y-2"><p class="font-semibold">\`/search\`（推荐入口）</p><pre class="overflow-x-auto whitespace-pre-wrap rounded-2xl border border-border bg-card px-4 py-3 text-xs font-mono scrollbar-slim">curl -X POST &quot;http://localhost:7860/search&quot; \\
  -H &quot;Content-Type: application/json&quot; \\
  -H &quot;Authorization: Bearer YOUR_API_KEY&quot; \\
  -d &#39;{
    &quot;query&quot;: &quot;latest exa ai funding news&quot;,
    &quot;numResults&quot;: 5
  }&#39;</pre></div><div class="space-y-2"><p class="font-semibold">原生 Exa 反代入口</p><pre class="overflow-x-auto whitespace-pre-wrap rounded-2xl border border-border bg-card px-4 py-3 text-xs font-mono scrollbar-slim">POST /answer
POST /search
POST /contents
POST /findSimilar
POST /research/v1
GET  /research/v1
GET  /research/v1/{researchId}</pre><p class="text-xs text-muted-foreground">上述接口均走账号池轮询，自动切换失效 key。</p></div><div class="space-y-2"><p class="font-semibold">账户配置格式（简化）</p><pre class="overflow-x-auto whitespace-pre-wrap rounded-2xl border border-border bg-card px-4 py-3 text-xs font-mono scrollbar-slim">[
  {
    &quot;id&quot;: &quot;account_1&quot;,
    &quot;exa_api_key&quot;: &quot;xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx&quot;,
    &quot;mail_provider&quot;: &quot;duckmail&quot;,
    &quot;mail_address&quot;: &quot;user@example.com&quot;,
    &quot;mail_password&quot;: &quot;******&quot;
  }
]</pre></div>`,3)])])):(s(),r("div",g,[...e[1]||(e[1]=[d('<div class="rounded-2xl border border-amber-200 bg-amber-50 p-4"><p class="font-medium text-amber-700">仅限学习与研究用途，禁止滥用。</p></div><div class="rounded-2xl border border-border bg-muted/30 p-4"><p class="font-medium text-foreground">你需要自行承担以下责任：</p><ul class="mt-2 space-y-1 pl-4"><li>• 遵守 Exa 及邮箱服务商条款</li><li>• 遵守所在地区法律法规</li><li>• 对账号封禁、数据丢失等后果自行负责</li></ul></div>',2)])]))])]))}});export{k as default};
