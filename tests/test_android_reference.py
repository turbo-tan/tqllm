"""Independent NumPy FP32 oracle for a tiny dense hybrid qwen35 GGUF.

Run with --runner BUILD/tqandroid-run --state BUILD/test-android-state.
No external model, network, llama.cpp, CUDA, MLX or phone is required.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import numpy as np


def fixture(path):
    rng = np.random.default_rng(871)
    h, f, heads, kh, vh, d, hd, vocab = 32, 64, 4, 2, 4, 4, 8, 64
    weights = {}
    def matrix(name, rows, cols):
        weights[name] = rng.normal(0, .12, (rows, cols)).astype('<f4')
    def norm(name, n):
        weights[name] = rng.uniform(.8, 1.2, n).astype('<f4')
    matrix('token_embd.weight', vocab, h)
    norm('output_norm.weight', h)
    for l in range(4):
        p = f'blk.{l}.'
        norm(p+'attn_norm.weight', h)
        norm(p+'post_attention_norm.weight', h)
        matrix(p+'ffn_gate.weight', f, h)
        matrix(p+'ffn_up.weight', f, h)
        matrix(p+'ffn_down.weight', h, f)
        if l == 3:
            matrix(p+'attn_q.weight', 2*heads*hd, h)
            matrix(p+'attn_k.weight', 2*hd, h)
            matrix(p+'attn_v.weight', 2*hd, h)
            matrix(p+'attn_output.weight', h, heads*hd)
            norm(p+'attn_q_norm.weight', hd)
            norm(p+'attn_k_norm.weight', hd)
        else:
            width = 2*kh*d+vh*d
            matrix(p+'attn_qkv.weight', width, h)
            matrix(p+'attn_gate.weight', vh*d, h)
            matrix(p+'ssm_alpha.weight', vh, h)
            matrix(p+'ssm_beta.weight', vh, h)
            matrix(p+'ssm_out.weight', h, vh*d)
            matrix(p+'ssm_conv1d.weight', width, 4)
            weights[p+'ssm_a'] = -rng.uniform(.2, 1., vh).astype('<f4')
            weights[p+'ssm_dt.bias'] = rng.normal(0, .1, vh).astype('<f4')
            norm(p+'ssm_norm.weight', d)
    meta = {'general.architecture':'qwen35', 'tokenizer.ggml.eos_token_id':63}
    dims = {'block_count':4, 'embedding_length':h, 'feed_forward_length':f,
            'attention.head_count':heads, 'attention.head_count_kv':2, 'attention.key_length':hd,
            'rope.dimension_count':8, 'ssm.group_count':kh, 'ssm.time_step_rank':vh,
            'ssm.state_size':d, 'ssm.inner_size':vh*d, 'ssm.conv_kernel':4,
            'full_attention_interval':4, 'context_length':64,
            'attention.layer_norm_rms_epsilon':1e-6, 'rope.freq_base':10000.}
    meta.update({'qwen35.'+k:v for k,v in dims.items()})
    def string(s):
        b=s.encode(); return struct.pack('<Q',len(b))+b
    header = b'GGUF'+struct.pack('<IQQ',3,len(weights),len(meta))
    for k,v in meta.items():
        header += string(k)
        if isinstance(v,str): header += struct.pack('<I',8)+string(v)
        elif isinstance(v,int): header += struct.pack('<II',4,v)
        else: header += struct.pack('<If',6,v)
    payload=b''
    for name,w in weights.items():
        payload += b'\0'*((-len(payload))%32)
        header += string(name)+struct.pack('<I',w.ndim)
        header += struct.pack('<'+'Q'*w.ndim,*reversed(w.shape))+struct.pack('<IQ',0,len(payload))
        payload += w.tobytes()
    path.write_bytes(header+b'\0'*((-len(header))%32)+payload)
    return weights


def reference(w, tokens):
    def rms(x, weight): return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*weight
    def sigmoid(x): return 1/(1+np.exp(-x))
    def silu(x): return x*sigmoid(x)
    def rotation(x, pos):
        a=pos*np.power(10000.,-np.arange(4)*2/8)
        return np.concatenate((x[:,:4]*np.cos(a)-x[:,4:]*np.sin(a),
                               x[:,4:]*np.cos(a)+x[:,:4]*np.sin(a)),axis=-1).astype('f4')
    histories=[np.zeros((3,32),dtype='f4') for _ in range(3)]
    states=[np.zeros((4,4,4),dtype='f4') for _ in range(3)]
    keys=[]; values=[]; outputs=[]
    for pos,token in enumerate(tokens):
        x=w['token_embd.weight'][token].copy()
        for l in range(4):
            p=f'blk.{l}.'
            n=rms(x,w[p+'attn_norm.weight'])
            def mat(name,a=None): return w[p+name]@(n if a is None else a)
            if l==3:
                qg=mat('attn_q.weight').reshape(4,16)
                q=rotation(rms(qg[:,:8],w[p+'attn_q_norm.weight']),pos)
                k=rotation(rms(mat('attn_k.weight').reshape(2,8),w[p+'attn_k_norm.weight']),pos)
                v=mat('attn_v.weight').reshape(2,8)
                keys.append(k);values.append(v)
                out=np.empty((4,8),dtype='f4')
                for head in range(4):
                    scores=np.array([q[head]@a[head//2]/math.sqrt(8) for a in keys])
                    probs=np.exp(scores-scores.max());probs/=probs.sum()
                    out[head]=sum(probs[t]*values[t][head//2] for t in range(len(probs)))
                y=mat('attn_output.weight',(out*sigmoid(qg[:,8:])).flatten())
            else:
                qkv=mat('attn_qkv.weight')
                joined=np.vstack((histories[l],qkv))
                mixed=silu(np.sum(joined.T*w[p+'ssm_conv1d.weight'],axis=1))
                histories[l]=joined[1:]
                q=mixed[:8].reshape(2,4);k=mixed[8:16].reshape(2,4);v=mixed[16:].reshape(4,4)
                q=q/np.sqrt(np.sum(q*q,axis=1,keepdims=True)+1e-6)/2
                k=k/np.sqrt(np.sum(k*k,axis=1,keepdims=True)+1e-6)
                dt=mat('ssm_alpha.weight')+w[p+'ssm_dt.bias']
                decay=np.exp(np.logaddexp(0,dt)*w[p+'ssm_a'])
                beta=sigmoid(mat('ssm_beta.weight'))
                out=np.empty((4,4),dtype='f4')
                for head in range(4):
                    s=states[l][head]*decay[head]
                    delta=(v[head]-s@k[head%2])*beta[head]
                    s+=np.outer(delta,k[head%2]);states[l][head]=s
                    out[head]=s@q[head%2]
                out=rms(out,w[p+'ssm_norm.weight'])*silu(mat('attn_gate.weight').reshape(4,4))
                y=mat('ssm_out.weight',out.flatten())
            x=x+y
            n=rms(x,w[p+'post_attention_norm.weight'])
            x=x+mat('ffn_down.weight',silu(mat('ffn_gate.weight'))*mat('ffn_up.weight'))
        outputs.append(w['token_embd.weight']@rms(x,w['output_norm.weight']))
    return outputs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runner',type=Path,required=True);ap.add_argument('--state',type=Path,required=True)
    args=ap.parse_args(); runner=args.runner.resolve();state=args.state.resolve()
    with tempfile.TemporaryDirectory(prefix='tqandroid-oracle-') as tmp:
        d=Path(tmp);model=d/'tiny.gguf';w=fixture(model);tokens=[1,7,2,9,3]
        expected=reference(w,tokens)
        for length in range(1,len(tokens)+1):
            (d/'ids.txt').write_text(' '.join(map(str,tokens[:length])))
            subprocess.run([str(runner),'--model',str(model),'--token-ids-file',str(d/'ids.txt'),
                            '--context','16','--max-tokens','0','--threads','2','--logits',str(d/'logits')],
                           check=True,capture_output=True,text=True,timeout=30)
            actual=np.fromfile(d/'logits',dtype='<f4')
            np.testing.assert_allclose(actual,expected[length-1],atol=2e-5,rtol=2e-5)
        original=model.read_bytes()
        bad = subprocess.run([str(runner),'--model',str(model),'--token-ids-file',str(d/'ids.txt'),
                              '--max-tokens','0','--logits',str(model)],capture_output=True,text=True,timeout=30)
        assert bad.returncode != 0 and model.read_bytes()==original, 'output alias destroyed model'
        for option,value in [('--threads','0'),('--context','-1'),('--max-tokens','1junk')]:
            bad=subprocess.run([str(runner),option,value],capture_output=True,text=True,timeout=30)
            assert bad.returncode != 0, 'invalid numeric option accepted'
        subprocess.run([str(state),str(model)],check=True,timeout=30)
        print(json.dumps({'result':'PASS','prefixes':len(tokens),'vocab':64,'atol':2e-5,'rtol':2e-5,
                          'oracle':'independent NumPy qwen35 GDN+attention+FFN'}))
if __name__=='__main__': main()
