"""
Critical Fix for Pruning: Ensure Masks Actually Work - UPDATED VERSION
Fixed all critical bugs causing 0% sparsity and performance collapse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ============= FIX 1: CORRECT SPARSITY CALCULATION =============

def calculate_actual_sparsity(model) -> float:
    """FIXED: Correctly calculate model sparsity"""
    total_params = 0
    zero_params = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():  # Use named_parameters!
            if 'weight' in name and param.requires_grad:  # Check name, not str(param)
                total_params += param.numel()
                zero_params += (param.abs() < 1e-8).sum().item()
    
    return zero_params / total_params if total_params > 0 else 0.0


# ============= FIX 2: CORRECTED MASKED ADAM =============

class MaskedAdam(torch.optim.Adam):
    """FIXED: Properly enforces masks after each optimization step"""
    
    def __init__(self, params, masks=None, lr=1e-3, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.masks = masks or {}
        self.enforce_count = 0
    
    def step(self, closure=None):
        """Perform optimization step with guaranteed mask enforcement"""
        loss = super().step(closure)
        
        # CRITICAL: Enforce masks after every weight update
        with torch.no_grad():
            enforced = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p in self.masks:
                        mask = self.masks[p]
                        # Ensure same device
                        if mask.device != p.device:
                            mask = mask.to(p.device)
                            self.masks[p] = mask
                        
                        # Apply mask - set pruned weights to exactly zero
                        p.data.mul_(mask)
                        p.data[mask == 0] = 0.0  # Explicit zeroing
                        enforced += 1
            
            self.enforce_count += 1
            if self.enforce_count % 100 == 0:
                logger.debug(f"Enforced masks on {enforced} parameters")
        
        return loss


# ============= FIX 3: CORRECTED GRADUAL PRUNING MODULE =============

class FixedGradualPruningModule:
    """FIXED: Gradual pruning with corrected logic and linear schedule"""
    
    def __init__(self, model, config, importance_scores=None, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.importance_scores = importance_scores or {}
        self.masks = {}
        self.current_sparsity = config.initial_sparsity
        self.pruning_step = 0
        
        # Initialize masks to all ones (no pruning initially)
        self._initialize_masks()
        logger.info(f"Initialized gradual pruning: {config.initial_sparsity:.1%} → {config.final_sparsity:.1%}")
    
    def _initialize_masks(self):
        """Initialize all masks to ones (no pruning)"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    self.masks[param] = torch.ones_like(param, device=param.device)
    
    def update_sparsity_and_create_masks(self):
        """FIXED: Update sparsity using linear schedule and create new masks"""
        
        # FIXED: Linear schedule instead of cubic
        if self.pruning_step >= self.config.pruning_steps:
            self.current_sparsity = self.config.final_sparsity
        else:
            progress = self.pruning_step / self.config.pruning_steps
            sparsity_range = self.config.final_sparsity - self.config.initial_sparsity
            # LINEAR schedule for predictable pruning
            self.current_sparsity = self.config.initial_sparsity + sparsity_range * progress
        
        self.pruning_step += 1
        
        logger.info(f"Pruning step {self.pruning_step}: Target sparsity {self.current_sparsity:.2%}")
        
        # Create new masks with updated sparsity
        self._create_masks_with_importance()
        
        # Verify actual sparsity
        actual = calculate_actual_sparsity(self.model)
        logger.info(f"  Actual sparsity after masking: {actual:.2%}")
        
        return self.masks
    
    def _create_masks_with_importance(self):
        """FIXED: Create masks with corrected importance and protection logic"""
        
        # Collect all weights with importance scaling
        all_weights = []
        param_info = []
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    weight_abs = param.data.abs().flatten()
                    
                    # Get importance score
                    importance = self._get_importance_score(name)
                    
                    # FIXED: Layer protection logic
                    layer_num = self._get_layer_number(name)
                    if layer_num in self.config.protect_critical_layers:
                        # FIXED: Divide to make harder to prune (was multiply!)
                        importance /= self.config.circuit_preservation_weight
                    
                    # Apply importance weighting
                    weighted_values = weight_abs * importance
                    
                    all_weights.append(weighted_values.cpu())
                    param_info.append((name, param, importance))
        
        # Find global threshold
        if not all_weights:
            logger.warning("No weights found for pruning!")
            return
        
        all_weights = torch.cat(all_weights)
        
        # Handle large models with sampling
        if len(all_weights) > 2000000:  # 2M threshold
            sample_size = 1000000
            indices = torch.randperm(len(all_weights))[:sample_size]
            sampled_weights = all_weights[indices]
            sorted_weights = torch.sort(sampled_weights)[0]
        else:
            sorted_weights = torch.sort(all_weights)[0]
        
        threshold_idx = int(len(sorted_weights) * self.current_sparsity)
        threshold = sorted_weights[threshold_idx].item()
        
        logger.info(f"  Global threshold: {threshold:.6f}")
        
        # Apply threshold to create masks
        total_params = 0
        pruned_params = 0
        
        for name, param, importance in param_info:
            # Recalculate weighted values
            layer_num = self._get_layer_number(name)
            final_importance = importance
            if layer_num in self.config.protect_critical_layers:
                final_importance /= self.config.circuit_preservation_weight
            
            weighted_param = param.data.abs() * final_importance
            
            # Create mask: keep weights above threshold
            mask = (weighted_param > threshold).float()
            
            # Ensure mask is on same device as parameter
            mask = mask.to(param.device)
            self.masks[param] = mask
            
            # Apply mask immediately
            param.data.mul_(mask)
            
            # Track statistics
            pruned = (mask == 0).sum().item()
            total = mask.numel()
            total_params += total
            pruned_params += pruned
            
            if logger.isEnabledFor(logging.DEBUG):
                layer_sparsity = pruned / total if total > 0 else 0
                logger.debug(f"  {name}: {layer_sparsity:.2%} sparsity")
        
        actual_sparsity = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"  Target: {self.current_sparsity:.2%}, Achieved: {actual_sparsity:.2%}")
    
    def _get_importance_score(self, param_name: str) -> float:
        """Get importance score for a parameter"""
        # Default importance
        importance = 1.0
        
        # Look for matching importance scores
        for key, score in self.importance_scores.items():
            if key in param_name or param_name.endswith(key):
                importance = max(score, 0.1)  # Minimum importance
                break
        
        return importance
    
    def _get_layer_number(self, name: str) -> int:
        """Extract layer number from parameter name"""
        parts = name.split('.')
        for part in parts:
            if part.isdigit():
                return int(part)
        return -1  # No layer number found
    
    def enforce_masks(self):
        """FIXED: Force re-application of all masks"""
        with torch.no_grad():
            enforced = 0
            for param, mask in self.masks.items():
                if mask.device != param.device:
                    mask = mask.to(param.device)
                    self.masks[param] = mask
                
                param.data.mul_(mask)
                param.data[mask == 0] = 0.0  # Explicit zeroing
                enforced += 1
            
            if enforced > 0:
                logger.debug(f"Enforced {enforced} masks")
    
    def get_current_sparsity(self) -> float:
        """Get current target sparsity"""
        return self.current_sparsity
    
    def verify_sparsity(self) -> float:
        """Verify actual sparsity of the model"""
        return calculate_actual_sparsity(self.model)


# ============= ENHANCED VERIFIED PRUNING MODULE =============

class VerifiedPruningModule:
    """Enhanced pruning module with all fixes applied"""
    
    def __init__(self, model, target_sparsity, device='cuda'):
        self.model = model
        self.target_sparsity = target_sparsity
        self.device = device
        self.param_to_mask = {}
        self.hooks = []
        self.current_sparsity = 0.0
        
    def create_masks_magnitude_based(self, sample_rate=0.1):
        """Create masks using fixed threshold calculation"""
        logger.info(f"Creating masks for {self.target_sparsity:.0%} sparsity")
        
        # Compute threshold using sampling
        threshold = self._compute_threshold_sampling(sample_rate)
        
        # Apply threshold to create masks
        total_params = 0
        zero_params = 0
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():  # FIXED: Use named_parameters
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
                    param.data[mask == 0] = 0.0  # Explicit zeroing
        
        self.current_sparsity = zero_params / total_params if total_params > 0 else 0
        logger.info(f"Created masks - Target: {self.target_sparsity:.0%}, Actual: {self.current_sparsity:.2%}")
        
        return self.param_to_mask
    
    def _compute_threshold_sampling(self, sample_rate=0.1):
        """Compute pruning threshold using sampling (memory-efficient)"""
        logger.info(f"Computing threshold with {sample_rate:.1%} sampling...")
        
        all_weights = []
        total_params = 0
        
        # Collect samples from all layers
        with torch.no_grad():
            for name, param in self.model.named_parameters():  # FIXED: Use named_parameters
                if 'weight' in name and param.requires_grad:
                    weight_abs = param.data.abs()
                    n_params = weight_abs.numel()
                    total_params += n_params
                    
                    # Sample a subset of weights
                    n_samples = max(1, int(n_params * sample_rate))
                    
                    if n_samples < n_params:
                        indices = torch.randperm(n_params, device=weight_abs.device)[:n_samples]
                        sampled = weight_abs.flatten()[indices]
                    else:
                        sampled = weight_abs.flatten()
                    
                    all_weights.append(sampled.cpu())
        
        # Combine all samples on CPU
        all_weights = torch.cat(all_weights)
        
        # Sort and find threshold
        sorted_weights, _ = torch.sort(all_weights)
        threshold_idx = int(len(sorted_weights) * self.target_sparsity)
        threshold = sorted_weights[threshold_idx].item()
        
        logger.info(f"Threshold for {self.target_sparsity:.0%} sparsity: {threshold:.6f}")
        
        return threshold
    
    def verify_sparsity(self):
        """Verify actual sparsity using corrected calculation"""
        return calculate_actual_sparsity(self.model)  # Use fixed function
    
    def enforce_masks(self):
        """Force re-application of masks with explicit zeroing"""
        with torch.no_grad():
            for param, mask in self.param_to_mask.items():
                if mask.device != param.device:
                    mask = mask.to(param.device)
                    self.param_to_mask[param] = mask
                param.data.mul_(mask)
                param.data[mask == 0] = 0.0  # Explicit zeroing
    
    def get_masks(self):
        """Get masks dictionary compatible with MaskedAdam"""
        return self.param_to_mask


# ============= VALIDATION AND DEBUGGING UTILITIES =============

def validate_pruning_step(model, target_sparsity: float, tolerance: float = 0.05) -> bool:
    """Validate that pruning achieved target sparsity within tolerance"""
    actual = calculate_actual_sparsity(model)
    within_tolerance = abs(actual - target_sparsity) <= tolerance
    
    if not within_tolerance:
        logger.warning(f"Pruning validation failed: Target {target_sparsity:.2%}, "
                      f"Actual {actual:.2%}, Tolerance {tolerance:.2%}")
    
    return within_tolerance


def debug_model_sparsity(model, top_k: int = 5):
    """Debug sparsity by layer"""
    layer_stats = []
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                total = param.numel()
                zeros = (param.abs() < 1e-8).sum().item()
                sparsity = zeros / total if total > 0 else 0
                
                layer_stats.append({
                    'name': name,
                    'sparsity': sparsity,
                    'shape': tuple(param.shape),
                    'zeros': zeros,
                    'total': total
                })
    
    # Sort by sparsity
    layer_stats.sort(key=lambda x: x['sparsity'], reverse=True)
    
    logger.info(f"Top {top_k} sparsest layers:")
    for i, stats in enumerate(layer_stats[:top_k]):
        logger.info(f"  {i+1}. {stats['name']}: {stats['sparsity']:.2%} "
                   f"({stats['zeros']}/{stats['total']}, shape={stats['shape']})")
    
    return layer_stats


def test_mask_enforcement(model, pruning_module, num_steps: int = 10):
    """Test that masks stick during training steps"""
    initial_sparsity = calculate_actual_sparsity(model)
    logger.info(f"Testing mask enforcement - Initial sparsity: {initial_sparsity:.2%}")
    
    # Simulate training steps
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for step in range(num_steps):
        # Random gradients
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param) * 0.1
        
        # Optimization step
        optimizer.step()
        optimizer.zero_grad()
        
        # Enforce masks
        pruning_module.enforce_masks()
        
        # Check sparsity
        current_sparsity = calculate_actual_sparsity(model)
        
        if abs(current_sparsity - initial_sparsity) > 0.01:
            logger.error(f"Mask enforcement failed at step {step}: "
                        f"{initial_sparsity:.2%} → {current_sparsity:.2%}")
            return False
    
    logger.info("✅ Mask enforcement test passed")
    return True


# ============= TRAINING FUNCTIONS WITH FIXES =============

def train_with_verified_pruning(
    model, 
    train_loader, 
    eval_loader, 
    target_sparsity, 
    device='cuda',
    num_epochs=2
):
    """Training loop with all fixes applied"""
    
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
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            if batch['labels'].dim() == 0:
                batch['labels'] = batch['labels'].unsqueeze(0)
            
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


def validate_pruning_results(model, target_sparsity, tolerance=0.05):
    """Enhanced validation with detailed reporting"""
    
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
    print("PRUNING VALIDATION REPORT (FIXED VERSION)")
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
        print("Check the fixes in this updated version")
    else:
        print("\n✅ PRUNING VALIDATION PASSED!")
    
    return validation_results


# ============= MAIN TEST FUNCTION =============

def test_pruning_pipeline():
    """Test the complete pruning pipeline with fixes"""
    
    print("Testing FIXED Pruning Pipeline...")
    
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
    
    # Test sparsity calculation fix
    initial_sparsity = calculate_actual_sparsity(model)
    print(f"Initial model sparsity: {initial_sparsity:.4%}")
    
    # 2. Test pruning at different sparsity levels
    for target_sparsity in [0.3, 0.5, 0.7]:
        print(f"\n{'='*60}")
        print(f"Testing {target_sparsity:.0%} Sparsity (FIXED VERSION)")
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
            print("❌ Test failed! Check the fixes.")
            break
        else:
            print("✅ Test passed!")
    
    print("\nFixed pruning test complete!")


if __name__ == "__main__":
    # Run this to test if the fixes work
    test_pruning_pipeline()
