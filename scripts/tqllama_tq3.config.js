// pm2 app for the tqLLM serving PAW-27B-X3.1 on GPU 2, 256K context, port 8087.
//   pm2 start  tqllm/scripts/tqllama_tq3.config.js
//   pm2 stop   tqllama-tq3
//   pm2 restart tqllama-tq3
//   pm2 logs   tqllama-tq3
module.exports = {
  apps: [{
    name: "tqllama-tq3",
    script: __dirname + "/tqllama_tq3.sh",
    interpreter: "bash",
    autorestart: true,
    max_restarts: 3,
    min_uptime: "60s",
    kill_timeout: 20000,
  }],
};
