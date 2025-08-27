"""
Critical Fix for Pruning: Ensure Masks Actually Work
Add this to your script to fix the 0% sparsity issue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


# ============= FIX 1: PERMANENT MASKING WITH CUSTOM OPTIMIZER =============

class MaskedAdam(torch.optim.Adam):
    """Custom Adam optimizer that respects pruning masks"""
    
    def __init__(self, params, masks=None, lr=1e-3, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.masks = masks or {}
        
    def step(self, closure=None):
        """Perform optimization step while maintaining sparsity"""
        loss = super().step(closure)
        
        # Re-apply masks after weight update
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p in self.masks:
                        # Ensure mask is on same device as parameter
                        mask = self.masks[p]
                        if mask.device != p.device:
                            mask = mask.to(p.device)
                            self.masks[p] = mask
                        p.data.mul_(mask)
        
        return loss


class VerifiedPruningModule:
    """Memory-efficient pruning module for large models with proper device handling"""
    
    def __init__(self, model, target_sparsity, device='cuda'):
        self.model = model
        self.target_sparsity = target_sparsity
        self.device = device
        self.param_to_mask = {}
        self.hooks = []
        self.current_sparsity = 0.0
        
    def create_masks_magnitude_based(self, sample_rate=0.1):
        """
        Create masks using memory-efficient sampling approach
        
        Args:
            sample_rate: Fraction of weights to sample for threshold calculation
        """
        logger.info(f"Creating masks for {self.target_sparsity:.0%} sparsity")
        
        # Compute threshold using sampling
        threshold = self._compute_threshold_sampling(sample_rate)
        
        # Apply threshold to create masks
        total_params = 0
        zero_params = 0
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    # Create mask on the same device as the parameter
                    mask = (param.data.abs() > threshold).float()
                    self.param_to_mask[param] = mask
                    
                    # Track sparsity
                    n_zeros = (mask == 0).sum().item()
                    zero_params += n_zeros
                    total_params += param.numel()
                    
                    # Apply mask immediately
                    param.data.mul_(mask)
        
        self.current_sparsity = zero_params / total_params if total_params > 0 else 0
        logger.info(f"Created masks - Target: {self.target_sparsity:.0%}, Actual: {self.current_sparsity:.2%}")
        
        # Install hooks to maintain sparsity
        self._install_hooks()
        
        return self.param_to_mask
    
    def _compute_threshold_sampling(self, sample_rate=0.1):
        """
        Compute pruning threshold using sampling (memory-efficient)
        
        Args:
            sample_rate: Fraction of weights to sample (0.01 = 1% for BERT)
        """
        logger.info(f"Computing threshold with {sample_rate:.1%} sampling...")
        
        all_weights = []
        total_params = 0
        
        # Collect samples from all layers
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    # Keep computations on the same device as the parameter
                    weight_abs = param.data.abs()
                    n_params = weight_abs.numel()
                    total_params += n_params
                    
                    # Sample a subset of weights
                    n_samples = max(1, int(n_params * sample_rate))
                    
                    if n_samples < n_params:
                        # Create indices on the same device as weight_abs
                        indices = torch.randperm(n_params, device=weight_abs.device)[:n_samples]
                        sampled = weight_abs.flatten()[indices]
                    else:
                        sampled = weight_abs.flatten()
                    
                    # Move to CPU for collecting across all layers
                    all_weights.append(sampled.cpu())
        
        # Combine all samples on CPU
        all_weights = torch.cat(all_weights)
        
        # Sort and find threshold
        sorted_weights, _ = torch.sort(all_weights)
        threshold_idx = int(len(sorted_weights) * self.target_sparsity)
        threshold = sorted_weights[threshold_idx].item()
        
        logger.info(f"Sampled {len(all_weights):,}/{total_params:,} weights ({len(all_weights)/total_params:.1%})")
        logger.info(f"Threshold for {self.target_sparsity:.0%} sparsity: {threshold:.6f}")
        
        return threshold
    
    def _compute_threshold_layerwise(self):
        """
        Alternative: Compute threshold layer by layer (for extreme memory constraints)
        """
        logger.info("Computing threshold layer-wise...")
        
        thresholds = []
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    weight_abs = param.data.abs()
                    
                    # Sort weights in this layer
                    sorted_weights = weight_abs.flatten().sort()[0]
                    
                    # Find threshold for this layer
                    threshold_idx = int(len(sorted_weights) * self.target_sparsity)
                    layer_threshold = sorted_weights[threshold_idx].item()
                    thresholds.append(layer_threshold)
                    
                    # Free memory
                    del sorted_weights
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Use median threshold across layers
        threshold = np.median(thresholds)
        logger.info(f"Layer-wise threshold: {threshold:.6f}")
        
        return threshold
    
    def _install_hooks(self):
        """Install forward hooks to apply masks during forward pass"""
        
        def mask_hook(module, input, output, mask):
            """Hook to apply mask to weights"""
            if hasattr(module, 'weight') and module.weight is not None:
                with torch.no_grad():
                    # Ensure mask is on same device
                    if mask.device != module.weight.device:
                        mask = mask.to(module.weight.device)
                    module.weight.data.mul_(mask)
            return output
        
        # Remove existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Install new hooks
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                param = module.weight
                if param in self.param_to_mask:
                    mask = self.param_to_mask[param]
                    hook = module.register_forward_hook(
                        lambda m, i, o, mask=mask: mask_hook(m, i, o, mask)
                    )
                    self.hooks.append(hook)
    
    def verify_sparsity(self):
        """Verify actual sparsity of the model"""
        
        total_params = 0
        zero_params = 0
        
        with torch.no_grad():
            for param, mask in self.param_to_mask.items():
                # Count zeros in actual weights
                zeros_in_weight = (param.data.abs() < 1e-8).sum().item()
                
                # Count expected zeros from mask
                zeros_in_mask = (mask == 0).sum().item()
                
                # Check consistency
                if abs(zeros_in_weight - zeros_in_mask) > 10:
                    # Re-apply mask if inconsistent
                    if mask.device != param.device:
                        mask = mask.to(param.device)
                    param.data.mul_(mask)
                    zeros_in_weight = (param.data.abs() < 1e-8).sum().item()
                
                total_params += param.numel()
                zero_params += zeros_in_weight
        
        actual_sparsity = zero_params / total_params if total_params > 0 else 0
        self.current_sparsity = actual_sparsity
        return actual_sparsity
    
    def enforce_masks(self):
        """Force re-application of masks"""
        with torch.no_grad():
            for param, mask in self.param_to_mask.items():
                # Ensure mask is on same device as parameter
                if mask.device != param.device:
                    mask = mask.to(param.device)
                    self.param_to_mask[param] = mask
                param.data.mul_(mask)
    
    def get_masks(self):
        """Get masks dictionary compatible with MaskedAdam"""
        return self.param_to_mask


# ============= FIX 3: TRAINING LOOP WITH GUARANTEED PRUNING =============

def train_with_verified_pruning(
    model, 
    train_loader, 
    eval_loader, 
    target_sparsity, 
    device='cuda',
    num_epochs=2
):
    """Training loop that guarantees pruning works"""
    
    # Step 1: Create pruning module
    pruning_module = VerifiedPruningModule(model, target_sparsity, device)
    
    # Step 2: Apply initial pruning
    masks = pruning_module.create_masks_magnitude_based()
    
    # Step 3: Verify initial sparsity
    initial_sparsity = pruning_module.verify_sparsity()
    logger.info(f"Initial sparsity after pruning: {initial_sparsity:.2%}")
    
    if initial_sparsity < target_sparsity * 0.9:
        logger.error(f"❌ Pruning failed! Target: {target_sparsity:.0%}, Actual: {initial_sparsity:.2%}")
        return None
    
    # Step 4: Create optimizer with mask support
    optimizer = MaskedAdam(model.parameters(), masks=masks, lr=5e-5)
    
    # Step 5: Training loop
    model.train()
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 100:  # Limited for testing
                break
            
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            logits = outputs.logits.squeeze()
            loss = F.mse_loss(logits, batch['labels'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Optimizer step (includes mask enforcement)
            optimizer.step()
            
            # Double-check: enforce masks again
            pruning_module.enforce_masks()
            
            # Verify sparsity periodically
            if batch_idx % 50 == 0:
                current_sparsity = pruning_module.verify_sparsity()
                if current_sparsity < target_sparsity * 0.9:
                    logger.warning(f"⚠️ Sparsity degraded to {current_sparsity:.2%}")
                    pruning_module.enforce_masks()
    
    # Final verification
    final_sparsity = pruning_module.verify_sparsity()
    logger.info(f"✅ Final sparsity: {final_sparsity:.2%}")
    
    return model, final_sparsity


# ============= FIX 4: BASELINE TRAINING WITH VERIFICATION =============

def train_proper_baseline(model, train_loader, device='cuda', min_correlation=0.10):
    """Train baseline until it reaches minimum performance"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    logger.info("Training baseline to minimum performance...")
    
    for epoch in range(5):  # Max 5 epochs
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 200:  # More training for baseline
                break
            
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            logits = outputs.logits.squeeze()
            loss = F.mse_loss(logits, batch['labels'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Quick evaluation
        model.eval()
        correlations = []
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                if i >= 20:  # Quick eval
                    break
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = model(batch['input_ids'], batch['attention_mask'])
                logits = outputs.logits.squeeze().cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                correlations.append(np.corrcoef(logits, labels)[0, 1])
        
        avg_correlation = np.mean(correlations)
        logger.info(f"Epoch {epoch+1} - Baseline correlation: {avg_correlation:.4f}")
        
        if avg_correlation >= min_correlation:
            logger.info(f"✅ Baseline reached minimum performance: {avg_correlation:.4f}")
            break
    
    return model


# ============= VALIDATION SCRIPT =============

def validate_pruning_results(model, target_sparsity, tolerance=0.05):
    """Validate that pruning actually worked"""
    
    total_params = 0
    zero_params = 0
    layer_info = {}
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                n_params = param.numel()
                n_zeros = (param.data.abs() < 1e-8).sum().item()
                
                total_params += n_params
                zero_params += n_zeros
                
                layer_sparsity = n_zeros / n_params if n_params > 0 else 0
                layer_info[name] = {
                    'sparsity': layer_sparsity,
                    'shape': param.shape,
                    'zeros': n_zeros,
                    'total': n_params
                }
    
    actual_sparsity = zero_params / total_params if total_params > 0 else 0
    
    # Validation checks
    validation_results = {
        'target_sparsity': target_sparsity,
        'actual_sparsity': actual_sparsity,
        'within_tolerance': abs(actual_sparsity - target_sparsity) <= tolerance,
        'total_parameters': total_params,
        'zero_parameters': zero_params,
        'status': 'PASS' if abs(actual_sparsity - target_sparsity) <= tolerance else 'FAIL'
    }
    
    # Print validation report
    print("\n" + "="*60)
    print("PRUNING VALIDATION REPORT")
    print("="*60)
    print(f"Target Sparsity: {target_sparsity:.1%}")
    print(f"Actual Sparsity: {actual_sparsity:.1%}")
    print(f"Difference: {abs(actual_sparsity - target_sparsity):.1%}")
    print(f"Status: {validation_results['status']}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Zero Parameters: {zero_params:,}")
    
    # Show top 5 most sparse layers
    print("\nTop 5 Most Sparse Layers:")
    sorted_layers = sorted(layer_info.items(), key=lambda x: x[1]['sparsity'], reverse=True)[:5]
    for name, info in sorted_layers:
        if info['sparsity'] > 0:
            print(f"  {name}: {info['sparsity']:.1%} sparse ({info['zeros']}/{info['total']})")
    
    if validation_results['status'] == 'FAIL':
        print("\n❌ PRUNING VALIDATION FAILED!")
        print("Possible causes:")
        print("1. Masks not being applied during optimizer.step()")
        print("2. Gradient hooks not registered properly")
        print("3. Model wrapped incorrectly (DDP issues)")
        print("4. Using wrong model reference for pruning")
    else:
        print("\n✅ PRUNING VALIDATION PASSED!")
    
    return validation_results


# ============= MAIN TEST FUNCTION =============

def test_pruning_pipeline():
    """Test the complete pruning pipeline"""
    
    print("Testing Pruning Pipeline with Verification...")
    
    # 1. Create dummy model for testing
    from transformers import AutoModel, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    base_model = AutoModel.from_pretrained('bert-base-uncased')
    
    # Simple wrapper
    class TestModel(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.bert = base
            self.classifier = nn.Linear(base.config.hidden_size, 1)
        
        def forward(self, input_ids, attention_mask=None, **kwargs):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.classifier(outputs.pooler_output)
            return type('Output', (), {'logits': logits})()
    
    model = TestModel(base_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # 2. Test pruning at different sparsity levels
    for target_sparsity in [0.3, 0.5, 0.7]:
        print(f"\n{'='*60}")
        print(f"Testing {target_sparsity:.0%} Sparsity")
        print('='*60)
        
        # Apply pruning
        pruning_module = VerifiedPruningModule(model, target_sparsity, device)
        pruning_module.create_masks_magnitude_based()
        
        # Verify
        actual_sparsity = pruning_module.verify_sparsity()
        print(f"Target: {target_sparsity:.0%}, Actual: {actual_sparsity:.2%}")
        
        # Validate
        results = validate_pruning_results(model, target_sparsity)
        
        if results['status'] == 'FAIL':
            print("❌ Test failed! Debugging required.")
            break
    
    print("\nTest complete!")


if __name__ == "__main__":
    # Run this to test if pruning actually works
    test_pruning_pipeline()
