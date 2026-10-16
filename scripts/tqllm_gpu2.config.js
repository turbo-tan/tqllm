// pm2 app for the tqLLM server on GPU 2 (RTX 3090), 256K context.
//   pm2 start  tqllm/scripts/tqllm_gpu2.config.js
//   pm2 stop   tqllm-gpu2
//   pm2 restart tqllm-gpu2
//   pm2 logs   tqllm-gpu2
module.exports = {
  apps: [{
    name: "tqllm-gpu2",
    script: __dirname + "/tqllm_gpu2.sh",
    interpreter: "bash",
    autorestart: true,
    max_restarts: 3,
    min_uptime: "60s",
    kill_timeout: 20000,
  }],
};
