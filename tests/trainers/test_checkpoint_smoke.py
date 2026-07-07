"""Smoke test for trainer checkpoint save/load functionality with AMP and torch.compile."""
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from graph_signal_diffusion.trainers.trainer import DiffusionTrainer
from graph_signal_diffusion.diffusion.base import BaseDiffusion


class DummyModel(nn.Module):
    """Minimal model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.param = nn.Parameter(torch.randn(5, 5))
    
    def forward(self, x):
        return self.linear(x)


class DummyDiffusion(BaseDiffusion):
    """Minimal diffusion wrapper for testing."""
    def __init__(self, model):
        super().__init__(model)
        self.beta = nn.Parameter(torch.tensor(0.1))
    
    def training_loss(self, data):
        return torch.tensor(1.0, requires_grad=True)
    
    def sample(self, shape, device, data=None, use_amp=False, return_selector_sampling_diagnostics=False):
        return torch.randn(shape)
    
    def add_noise(self, x, t, noise=None):
        """Dummy implementation of abstract method."""
        return x + 0.1 * torch.randn_like(x) if noise is None else x + noise


def test_checkpoint_save_load_basic():
    """Test basic save and load functionality."""
    print("=" * 80)
    print("Test 1: Basic checkpoint save/load")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
        
        device = torch.device("cpu")
        model = DummyModel()
        diffusion = DummyDiffusion(model)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=0.001)
        
        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=tmpdir,
            use_amp=False
        )
        trainer.epoch = 42
        
        initial_weight = model.linear.weight.data.clone()
        
        print(f"✓ Created trainer (epoch={trainer.epoch})")
        
        trainer.save_checkpoint(checkpoint_path)
        print(f"✓ Saved checkpoint")
        
        model.linear.weight.data.fill_(999.0)
        print(f"✓ Modified weights")
        
        loaded_epoch = trainer.load_checkpoint(checkpoint_path)
        print(f"✓ Loaded checkpoint (epoch={loaded_epoch})")
        
        assert torch.allclose(model.linear.weight.data, initial_weight), "Weights not restored"
        assert loaded_epoch == 42, f"Epoch mismatch: {loaded_epoch} != 42"
        
        print("✅ Test 1 PASSED\n")
        return True


def test_checkpoint_with_amp():
    """Test that GradScaler state is saved and restored with AMP."""
    print("=" * 80)
    print("Test 2: Checkpoint with AMP (GradScaler)")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("⚠️  SKIPPED (CUDA not available)\n")
        return True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_amp_checkpoint.pt")
        
        device = torch.device("cuda")
        model = DummyModel().to(device)
        diffusion = DummyDiffusion(model)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=0.001)
        
        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=tmpdir,
            use_amp=True
        )
        trainer.epoch = 10
        
        # Simulate some scaler state changes
        if trainer.scaler:
            trainer.scaler._scale = torch.tensor(2048.0, device=device)  # Modified scale
            print(f"✓ Modified GradScaler scale to 2048.0")
        
        trainer.save_checkpoint(checkpoint_path)
        print(f"✓ Saved AMP checkpoint")
        
        # Verify scaler state is in checkpoint
        ckpt = torch.load(checkpoint_path)
        assert "scaler" in ckpt, "GradScaler state not saved"
        assert "use_amp" in ckpt, "use_amp metadata not saved"
        print(f"✓ Checkpoint contains scaler state")
        
        # Create new trainer and load
        model2 = DummyModel().to(device)
        diffusion2 = DummyDiffusion(model2)
        optimizer2 = torch.optim.Adam(diffusion2.parameters(), lr=0.001)
        
        trainer2 = DiffusionTrainer(
            diffusion=diffusion2,
            optimizer=optimizer2,
            device=device,
            save_dir=tmpdir,
            use_amp=True
        )
        
        initial_scale = trainer2.scaler.get_scale()
        print(f"✓ New trainer scaler scale: {initial_scale}")
        
        trainer2.load_checkpoint(checkpoint_path)
        restored_scale = trainer2.scaler.get_scale()
        print(f"✓ Restored scaler scale: {restored_scale}")
        
        assert restored_scale == 2048.0, f"Scaler scale not restored: {restored_scale}"
        
        print("✅ Test 2 PASSED\n")
        return True


def test_checkpoint_with_compile():
    """Test that torch.compile doesn't break checkpoint save/load."""
    print("=" * 80)
    print("Test 3: Checkpoint with torch.compile")
    print("=" * 80)
    
    if not hasattr(torch, 'compile'):
        print("⚠️  SKIPPED (torch.compile not available)\n")
        return True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_compile_checkpoint.pt")
        
        device = torch.device("cpu")
        model = DummyModel()
        diffusion = DummyDiffusion(model)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=0.001)
        
        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=tmpdir,
            use_amp=False
        )
        trainer.epoch = 5
        
        # Compile the model
        diffusion.model = torch.compile(diffusion.model, mode="reduce-overhead")
        print(f"✓ Compiled model with torch.compile()")
        
        # Verify it's wrapped
        assert hasattr(diffusion.model, "_orig_mod"), "Model should be wrapped by torch.compile"
        print(f"✓ Model is wrapped (_orig_mod exists)")
        
        initial_weight = diffusion.model._orig_mod.linear.weight.data.clone()
        
        # Save checkpoint with compiled model
        trainer.save_checkpoint(checkpoint_path)
        print(f"✓ Saved checkpoint with compiled model")
        
        # Verify checkpoint has clean keys (no _orig_mod prefix)
        ckpt = torch.load(checkpoint_path)
        model_keys = list(ckpt["model"].keys())
        has_orig_mod = any("_orig_mod" in k for k in model_keys)
        assert not has_orig_mod, f"Checkpoint should not have _orig_mod keys: {model_keys[:3]}"
        print(f"✓ Checkpoint has clean keys (no _orig_mod prefix)")
        
        # Modify weights
        diffusion.model._orig_mod.linear.weight.data.fill_(777.0)
        
        # Load into same compiled model
        trainer.load_checkpoint(checkpoint_path)
        restored_weight = diffusion.model._orig_mod.linear.weight.data
        assert torch.allclose(restored_weight, initial_weight), "Weights not restored in compiled model"
        print(f"✓ Loaded into compiled model successfully")
        
        # Test loading into non-compiled model
        model2 = DummyModel()
        diffusion2 = DummyDiffusion(model2)
        optimizer2 = torch.optim.Adam(diffusion2.parameters(), lr=0.001)
        
        trainer2 = DiffusionTrainer(
            diffusion=diffusion2,
            optimizer=optimizer2,
            device=device,
            save_dir=tmpdir,
            use_amp=False
        )
        
        diffusion2.model.linear.weight.data.fill_(888.0)
        trainer2.load_checkpoint(checkpoint_path)
        restored_weight2 = diffusion2.model.linear.weight.data
        assert torch.allclose(restored_weight2, initial_weight), "Weights not restored in non-compiled model"
        print(f"✓ Loaded into non-compiled model successfully")
        
        print("✅ Test 3 PASSED\n")
        return True


if __name__ == "__main__":
    try:
        results = []
        results.append(test_checkpoint_save_load_basic())
        results.append(test_checkpoint_with_amp())
        results.append(test_checkpoint_with_compile())
        
        print("=" * 80)
        if all(results):
            print("✅ ALL TESTS PASSED")
            print("=" * 80)
            sys.exit(0)
        else:
            print("❌ SOME TESTS FAILED")
            print("=" * 80)
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
