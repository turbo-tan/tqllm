// pm2 app for the tqLLM serving PAW-27B-X3.1 on GPU 2, 256K context, port 8087.
//   pm2 start  tqllm/scripts/tqllama.config.js
//   pm2 stop   tqllama
//   pm2 restart tqllama
//   pm2 logs   tqllama
module.exports = {
  apps: [{
    name: "tqllama",
    script: __dirname + "/tqllama.sh",
    interpreter: "bash",
    autorestart: true,
    max_restarts: 3,
    min_uptime: "60s",
    kill_timeout: 20000,
  }],
};
