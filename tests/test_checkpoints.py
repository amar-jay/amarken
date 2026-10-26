import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class CheckpointTests(unittest.TestCase):
    def test_storage(self):
        for package in ('amarken', 'example'):
            with self.subTest(package=package), tempfile.TemporaryDirectory() as directory:
                module = importlib.import_module(f'{package}.logger')
                root = Path(directory) / 'checkpoints'
                def save(state, path):
                    Path(path).write_text(repr(state))
                torch = Mock(save=save)
                with patch.object(module, 'CHECKPOINT_DIR', root), patch.dict(sys.modules, torch=torch):
                    logger = module.WandbLogger(enabled=False, checkpoints={'backend': 'local'})
                    result = logger.save_checkpoint({'step': 1}, 'run/latest.pt')
                    self.assertEqual(result, root / 'run/latest.pt')
                    self.assertEqual(result.read_text(), "{'step': 1}")
                    for invalid in ('../escape.pt', '/tmp/escape.pt', '.'):
                        with self.assertRaises(ValueError):
                            logger.save_checkpoint({}, invalid)
                    worker = module.WandbLogger(enabled=False, rank=1, checkpoints={'backend': 'local'})
                    self.assertIsNone(worker.save_checkpoint({}, 'worker.pt'))
                    self.assertFalse((root / 'worker.pt').exists())
                    bucket = module.WandbLogger(enabled=False, checkpoints={
                        'backend': 's3', 'bucket': 'test-bucket', 'prefix': '/run-1/'})
                    client = Mock()
                    bucket._storage_client = client
                    self.assertEqual(bucket.save_checkpoint({}, 'step.pt'), 's3://test-bucket/run-1/step.pt')
                    client.upload_file.assert_called_once_with(str(root / 'step.pt'), 'test-bucket', 'run-1/step.pt')
                    client.upload_file.side_effect = RuntimeError('upload failed')
                    with self.assertRaises(RuntimeError):
                        bucket.save_checkpoint({}, 'retained.pt')
                    self.assertTrue((root / 'retained.pt').exists())
                    with patch.object(torch, 'save', side_effect=RuntimeError('save failed')):
                        with self.assertRaises(RuntimeError):
                            logger.save_checkpoint({}, 'run/latest.pt')
                    self.assertEqual(result.read_text(), "{'step': 1}")
                    with self.assertRaises(ValueError):
                        module.CheckpointConfig(backend='s3')
                    with self.assertRaises(ValueError):
                        module.CheckpointConfig(backend='unknown')


if __name__ == '__main__':
    unittest.main()
