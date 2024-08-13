addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

const AUTH_TOKEN = 'momo';  // 自定义的Token

async function handleRequest(request) {
  const url = new URL(request.url);
  const pathSegments = url.pathname.split('/').filter(Boolean);

  // 通过路径获取Token
  const token = pathSegments[pathSegments.length - 1];

  // 检查请求中的Token是否与预定义的Token匹配
  if (token !== AUTH_TOKEN) {
      return new Response('Unauthorized', { status: 401 });
  }

  // 定义订阅地址列表
  const subscriptions = [
      'https://prob.xn--l9qyaz082a.cn/api/v1/client/subscribe?token=039dc1df64ab829e22419b97e84aaeb3',
      'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3QvcyFBa2p5UUFsYW9PS1AzU2F0OEx1MW1VU3BfM251/root/content',
      'https://workers.170011.xyz/fe5b82c9-80ab-4c9f-acd4-0f838f413173/pty',
      'https://oiiccdn.yydsii.com/api/v1/client/subscribe?token=9bcb31ae19a8bfeddfa1aee10fb8378e',
      'https://api.ssrsub.com/api/v1/client/subscribe?token=56b203fbc1a0ac712e1113af2acab49b',
      'https://link01.pikachucloud.site/api/v1/client/subscribe?token=745d6a1b15291768c0383f2796a80d24',
  ];

  // 直接提供的单个节点链接
  const singleNodes = [
      'vmess://ew0KICAidiI6ICIyIiwNCiAgInBzIjogIkdQVHx2bWVzc1x1MDAyQndzIiwNCiAgImFkZCI6ICIxMDcuMTc1LjQ0LjE5MiIsDQogICJwb3J0IjogIjU0NTc4IiwNCiAgImlkIjogImI3OGM0N2RjLTFkMjQtNGIxYy05MTA0LWY4NWEzNWJiNTRmZCIsDQogICJhaWQiOiAiMCIsDQogICJzY3kiOiAiYXV0byIsDQogICJuZXQiOiAid3MiLA0KICAidHlwZSI6ICJub25lIiwNCiAgImhvc3QiOiAiIiwNCiAgInBhdGgiOiAiL2I3OGM0N2RjIiwNCiAgInRscyI6ICIiLA0KICAic25pIjogIiIsDQogICJhbHBuIjogIiIsDQogICJmcCI6ICIiDQp9',
      'vless://d53b906d-7e26-470d-e122-abc8210c9882@107-175-44-192.nip.io:35760?encryption=none&flow=xtls-rprx-vision&security=tls&sni=107-175-44-192.nip.io&fp=chrome&type=tcp&headerType=none#GPT%7Cvless%2Bvision',
      'vless://269ac665-ade0-4f46-f9f5-7e10734dccea@107.175.44.192:443?encryption=none&flow=xtls-rprx-vision&security=reality&sni=blog.api.www.cloudflare.com&fp=chrome&pbk=9rx7JwMO-KRZZEM9TQBO19BOAmmGjJyjN86ll2J7uVc&type=tcp&headerType=none#GPT%7Cvless%2Bvision%2Breality',
  ];

  let aggregatedNodes = [...singleNodes];

  for (const baseUrl of subscriptions) {
      try {
          const response = await fetch(baseUrl);
          if (response.ok) {
              const text = await response.text();
              const nodes = decodeBase64(text).split('\n');
              aggregatedNodes = aggregatedNodes.concat(nodes);
          }
      } catch (error) {
          console.log(`Failed to fetch subscription from ${baseUrl}: ${error}`);
      }
  }

  // 过滤空行并去重
  aggregatedNodes = aggregatedNodes.filter(node => node.trim() !== '').filter((value, index, self) => self.indexOf(value) === index);

  const aggregatedContent = encodeBase64(aggregatedNodes.join('\n'));

  return new Response(aggregatedContent, {
      headers: {
          'content-type': 'text/plain',
          'Cache-Control': 'no-store'
      }
  });
}

// Base64解码
function decodeBase64(encoded) {
  try {
      return atob(encoded);
  } catch (e) {
      console.log('Failed to decode Base64 string:', e);
      return '';
  }
}

// Base64编码
function encodeBase64(str) {
  return btoa(str);
}
