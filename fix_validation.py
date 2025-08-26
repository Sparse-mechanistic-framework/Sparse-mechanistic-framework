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
        # Standard Adam step
        loss = super().step(closure)
        
        # Re-apply masks after weight update
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p in self.masks:
                        p.data.mul_(self.masks[p])
        
        return loss


# ============= FIX 2: PRUNING MODULE WITH VERIFIED MASKING =============

class VerifiedPruningModule:
    """Pruning module that guarantees sparsity is maintained"""
    
    def __init__(self, model, target_sparsity, device='cuda'):
        self.model = model
        self.target_sparsity = target_sparsity
        self.device = device
        self.param_to_mask = {}
        self.hooks = []
        
    def create_masks_magnitude_based(self):
        """Create masks using magnitude-based pruning"""
        logger.info(f"Creating masks for {self.target_sparsity:.0%} sparsity")
        
        # Collect all weights
        all_weights = []
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    weight_abs = param.data.abs().flatten()
                    all_weights.append(weight_abs.cpu())  # Move to CPU to avoid memory issues
        
        # Concatenate all weights
        all_weights = torch.cat(all_weights)
        
        # FIX: Replace quantile with sorting approach for large tensors
        if len(all_weights) > 10_000_000:  # If tensor is large (>10M elements)
            # Use sorting instead of quantile
            sorted_weights, _ = torch.sort(all_weights)
            threshold_idx = int(len(sorted_weights) * self.target_sparsity)
            threshold = sorted_weights[threshold_idx].item() if threshold_idx < len(sorted_weights) else 0.0
        else:
            # Use quantile for smaller tensors
            threshold = torch.quantile(all_weights, self.target_sparsity).item()
        
        # Apply threshold to create masks
        total_params = 0
        zero_params = 0
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    mask = (param.data.abs() > threshold).float()
                    self.masks[param] = mask
                    
                    # Apply mask
                    param.data.mul_(mask)
                    
                    # Track sparsity
                    n_zeros = (mask == 0).sum().item()
                    zero_params += n_zeros
                    total_params += param.numel()
        
        actual_sparsity = zero_params / total_params if total_params > 0 else 0
        logger.info(f"Created masks - Target: {self.target_sparsity:.0%}, Actual: {actual_sparsity:.2%}")
        
        return self.masks
    
    def verify_sparsity(self):
        """Verify that sparsity is actually maintained"""
        total_params = 0
        zero_params = 0
        
        for param, mask in self.param_to_mask.items():
            # Check if masked weights are actually zero
            masked_zeros = (param.data[mask == 0].abs() < 1e-8).sum().item()
            expected_zeros = (mask == 0).sum().item()
            
            if masked_zeros != expected_zeros:
                logger.warning(f"Sparsity violation: {masked_zeros}/{expected_zeros} zeros")
            
            total_params += param.numel()
            zero_params += (param.data.abs() < 1e-8).sum().item()
        
        actual_sparsity = zero_params / total_params if total_params > 0 else 0
        return actual_sparsity
    
    def enforce_masks(self):
        """Force re-application of masks (call after each optimizer step)"""
        with torch.no_grad():
            for param, mask in self.param_to_mask.items():
                param.data.mul_(mask)


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
