"""Run the metadata tool on small valid and malformed shard sets, without a model/GPU."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

BINARY = sys.argv.pop(1)


def text(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


def shard(path, index, names, metadata=False, tensor_type=46):
    fields = {'split.no': index, 'split.count': 2}
    if metadata:
        fields['general.architecture'] = 'qwen4exp'
        fields.update({'qwen4exp.' + k: v for k, v in {
            'block_count': 1, 'embedding_length': 32, 'expert_count': 2, 'expert_used_count': 1,
            'expert_feed_forward_length': 32, 'expert_shared_feed_forward_length': 32,
            'hyper_connection.count': 2, 'hyper_connection.low_rank': 16}.items()})
    header = b'GGUF' + struct.pack('<IQQ', 3, len(names), len(fields))
    for key, value in fields.items():
        header += text(key)
        header += struct.pack('<I', 8 if isinstance(value, str) else 4)
        header += text(value) if isinstance(value, str) else struct.pack('<I', value)
    payload = b''
    for name in names:
        header += text(name) + struct.pack('<IQQQIQ', 3, 32, 32, 2, tensor_type, len(payload))
        payload += bytes(1024)
    header += bytes((-len(header)) % 32)
    path.write_bytes(header + payload)


class ShardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.first = self.root / 'first.gguf'
        self.second = self.root / 'second.gguf'
        shard(self.first, 0, ['blk.0.ffn_gate_exps.weight'], metadata=True)
        shard(self.second, 1, ['blk.0.ffn_up_exps.weight', 'blk.0.ffn_down_exps.weight'])

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *paths):
        return subprocess.run([BINARY, *map(str, paths)], capture_output=True, text=True, timeout=10)

    def test_metadata_only_in_first_shard(self):
        result = self.run_tool(self.first, self.second)
        self.assertEqual(result.returncode, 0, result.stderr)
        info = json.loads(result.stdout)
        self.assertEqual(info['expert_bytes'], 3072)
        self.assertEqual(info['max_expert_triplet_bytes'], 1536)
        self.assertFalse(info['inference_ready'])

    def test_missing_shard(self):
        result = self.run_tool(self.first)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('incomplete shard set', result.stderr)

    def test_duplicate_shard(self):
        result = self.run_tool(self.first, self.first)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('duplicate shard index', result.stderr)

    def test_duplicate_tensor(self):
        shard(self.second, 1, ['blk.0.ffn_gate_exps.weight'])
        result = self.run_tool(self.first, self.second)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('duplicate tensor', result.stderr)

    def test_truncated_payload(self):
        self.second.write_bytes(self.second.read_bytes()[:-1])
        result = self.run_tool(self.first, self.second)
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, 'exceeds shard|out of bounds')

    def test_metadata_shard_must_be_first(self):
        self.assertNotEqual(self.run_tool(self.second, self.first).returncode, 0)

    def test_gsq_dialect_is_explicit(self):
        shard(self.first, 0, ['blk.0.ffn_gate_exps.weight'], metadata=True, tensor_type=42)
        self.assertNotEqual(self.run_tool(self.first, self.second).returncode, 0)
        result = self.run_tool('--gsq-rco', self.first, self.second)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['dialect'], 'gsq-rco')


if __name__ == '__main__':
    unittest.main()
