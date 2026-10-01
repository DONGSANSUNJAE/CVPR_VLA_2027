import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'app'), str(ROOT/'app/scripts'), str(ROOT/'tools'),
               str(ROOT/'app/openpi_snapshot/src'), str(ROOT/'app/openpi_snapshot/packages/openpi-client/src')]


class PortableTests(unittest.TestCase):
    def test_false_adapter_and_delta_roundtrip(self):
        import numpy as np
        from configs.train_config import build_config, DELTA_MASK
        from compute_norm_stats import transform_numeric
        from openpi import transforms
        cfg = build_config()
        self.assertFalse(cfg.data.adapt_to_pi)
        self.assertEqual(cfg.fsdp_devices, 4)
        self.assertEqual(cfg.batch_size, 64)
        self.assertEqual(cfg.num_train_steps, 60000)
        state = np.array([.2,-.3,.7,-.8,.1,.6,.75,-.2,.3,-.7,.8,-.1,-.6,.25], dtype=np.float32)
        actions = np.tile(state, (50,1)) + np.linspace(-.01,.01,50,dtype=np.float32)[:,None]
        s,a = transform_numeric(state[None], actions[None])
        np.testing.assert_array_equal(s[0], state)
        np.testing.assert_array_equal(a[0,:, [6,13]], actions[:,[6,13]].T)
        native = cfg.data.create(cfg.assets_dirs, cfg.model).data_transforms
        raw = dict(state=state.copy(), actions=actions.copy(), images={'cam_high':np.zeros((3,224,224),dtype=np.uint8)}, prompt='test')
        encoded = transforms.compose(native.inputs)(copy.deepcopy(raw))
        np.testing.assert_allclose(encoded['actions'], a[0], atol=1e-7)
        decoded = transforms.compose(native.outputs)(encoded)
        np.testing.assert_allclose(decoded['actions'], actions, atol=1e-7)
        np.testing.assert_array_equal(raw['state'], state)

    def test_zip_restore_resume_and_corruption(self):
        from package_payload import build_shard
        from restore_payload import restore
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); source=p/'source';source.write_bytes(b'hello robotwin\0'*100)
            shard=build_shard(p,0,[dict(source=str(source),path='data/example.bin')])
            manifest=p/'payload_manifest.json'
            manifest.write_text(json.dumps(dict(schema='pi05_portable_zip_v1',complete=True,episodes=27500,shards=[shard])))
            restore(manifest,p,p/'out')
            restore(manifest,p,p/'out')
            self.assertEqual((p/'out/data/example.bin').read_bytes(),source.read_bytes())
            (p/'payload-000.zip').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):restore(manifest,p,p/'other')


if __name__ == '__main__':
    unittest.main()
