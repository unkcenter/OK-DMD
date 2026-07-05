"""
Online Koopman Dynamic Mode Decomposition (OK-DMD) for KV-Cache Eviction.

This module implements a query-agnostic, mathematically grounded KV-cache 
eviction policy by tracking the global semantic attractor of key sequences 
on a Riemannian manifold using Koopman Operator Theory.
"""

from typing import Optional
import torch


class OKDMDEviction:
    """
    Online Koopman Dynamic Mode Decomposition (OK-DMD) for KV-cache eviction.
    
    This class tracks the semantic trajectory of attention key sequences in real time,
    estimating the linear transition operator A_t using Recursive Least Squares (RLS)
    with exponential forgetting. Persistent semantic modes are extracted via 
    QR-based subspace iteration to score and evict transient cache states.
    """
    
    def __init__(
        self,
        num_heads_kv: int,
        head_dim: int,
        k_modes: int = 16,
        lambda_forget: float = 0.995,
        stab_reg: float = 1e-5,
        sink_size: int = 4,
        window_size: int = 16,
        update_interval: int = 16
    ) -> None:
        """
        Initializes the OK-DMD eviction manager.

        Args:
            num_heads_kv: Number of Key-Value heads in the attention mechanism.
            head_dim: Dimensionality of each attention head (d).
            k_modes: Dimensionality of the tracked persistent subspace (k).
            lambda_forget: Exponential forgetting factor (lambda) for RLS.
            stab_reg: Regularization parameter (sigma) for numerical stability.
            sink_size: Number of initial tokens to protect (Attention Sinks).
            window_size: Number of recent tokens to protect (Sliding Window).
            update_interval: Step interval (M) to perform the QR subspace update.
        """
        self.H_kv = num_heads_kv
        self.d = head_dim
        self.k = k_modes
        self.lambda_ = lambda_forget
        self.sigma = stab_reg
        self.sink_size = sink_size
        self.window_size = window_size
        self.update_interval = update_interval
        
        # State tensors initialized dynamically in initialize()
        self.P: Optional[torch.Tensor] = None  # Inverse covariance [B, H_kv, d, d]
        self.A: Optional[torch.Tensor] = None  # Koopman operator [B, H_kv, d, d]
        self.Q: Optional[torch.Tensor] = None  # Subspace projection basis [B, H_kv, d, k]
        self.step_counter: int = 0
        self.batch_size: int = 0

    def initialize(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        Allocates and initializes state tensors on the specified hardware and precision.

        Args:
            batch_size: The batch size of the active inference block.
            device: Target torch device (e.g., torch.device('cuda')).
            dtype: Target data precision (e.g., torch.float32, torch.float16, torch.bfloat16).
        """
        self.batch_size = batch_size
        self.step_counter = 0
        
        # Identity matrix scaled for Tikhonov regularization
        identity = torch.eye(self.d, device=device, dtype=dtype).view(1, 1, self.d, self.d)
        
        # P_1 initialized to 1e4 * I (representing inverse of near-zero initial covariance)
        self.P = identity.repeat(batch_size, self.H_kv, 1, 1) / 1e-4
        self.A = identity.repeat(batch_size, self.H_kv, 1, 1)
        
        # Orthogonal random initial projection basis Q_1
        random_matrix = torch.randn(batch_size, self.H_kv, self.d, self.k, device=device, dtype=dtype)
        self.Q, _ = torch.linalg.qr(random_matrix)

    def prefill_step(self, K_prompt: torch.Tensor, qr_iterations: int = 10) -> None:
        """
        Processes prompt keys in parallel (Batch DMD) during the prefill phase.
        Estimates the initial transition operator A and converges the subspace Q.

        Args:
            K_prompt: Key tensor from the prompt of shape [B, H_kv, Seq_len, d].
            qr_iterations: Number of QR power iterations to guarantee early convergence.
        """
        B, H, N, d = K_prompt.shape
        self.batch_size = B
        device = K_prompt.device
        dtype = K_prompt.dtype
        
        if N < 2:
            return  # Sequence is too short to estimate transitions
            
        # Shifted key matrices representing inputs (X) and targets (Y)
        X = K_prompt[:, :, :-1].transpose(-2, -1)  # [B, H, d, N-1]
        Y = K_prompt[:, :, 1:].transpose(-2, -1)   # [B, H, d, N-1]
        
        # Compute exact prompt covariance
        XXT = torch.matmul(X, X.transpose(-2, -1))  # [B, H, d, d]
        identity = torch.eye(self.d, device=device, dtype=dtype).view(1, 1, self.d, self.d)
        reg = identity * self.sigma
        
        # Batch inversion of regularized covariance
        self.P = torch.linalg.inv(XXT + reg)
        
        # Initialize Koopman matrix: A = (Y * X^T) * P
        YXT = torch.matmul(Y, X.transpose(-2, -1))
        self.A = torch.matmul(YXT, self.P)
        
        # Force convergence of initial subspace Q using prefill transitions
        for _ in range(qr_iterations):
            self.Q, _ = torch.linalg.qr(torch.matmul(self.A, self.Q))

    def decode_step(self, k_prev: torch.Tensor, k_curr: torch.Tensor) -> None:
        """
        Performs a sequential O(d^2) Sherman-Morrison update of the Koopman operator.
        Includes active stabilization safeguards for FP16 and BF16.

        Args:
            k_prev: Key vector from the previous token step, shape [B, H_kv, d].
            k_curr: Key vector from the current newly generated token step, shape [B, H_kv, d].
        """
        assert self.P is not None and self.A is not None, "OK-DMD states are not initialized."
        
        # Ensure column vector shape: [B, H_kv, d, 1]
        k_prev_col = k_prev.unsqueeze(-1)
        k_curr_col = k_curr.unsqueeze(-1)
        
        # 1. Compute dynamic projection u_t = P_{t-1} * k_{t-1}
        u = torch.matmul(self.P, k_prev_col)
        
        # 2. Compute adaptive gain denominator
        denom = self.lambda_ + torch.matmul(k_prev_col.transpose(-2, -1), u) + 1e-6
        g = u / denom  # Kalman Gain vector
        
        # 3. Compute temporal update of inverse covariance
        P_next = (self.P - torch.matmul(g, u.transpose(-2, -1))) / self.lambda_
        
        # 4. Active Stabilization (Forces symmetry and applies Tikhonov diagonal loading)
        identity = torch.eye(self.d, device=k_prev.device, dtype=k_prev.dtype).view(1, 1, self.d, self.d)
        self.P = 0.5 * (P_next + P_next.transpose(-2, -1)) + (identity * self.sigma)
        
        # 5. Compute transition error and update Koopman transition matrix A_t
        error = k_curr_col - torch.matmul(self.A, k_prev_col)
        self.A = self.A + torch.matmul(error, g.transpose(-2, -1))
        
        self.step_counter += 1

    def compute_eviction_scores(self, K_cache: torch.Tensor, force_update: bool = False) -> torch.Tensor:
        """
        Projects cached keys onto the persistent Koopman subspace to calculate survival scores.

        Args:
            K_cache: Key cache tensor of shape [B, H_kv, Cache_Size, d].
            force_update: Force an immediate QR power iteration, bypassing the step interval.

        Returns:
            Scores tensor of shape [B, H_kv, Cache_Size]. Higher score represents a token 
            that is more aligned with the persistent global semantic attractor.
        """
        assert self.A is not None and self.Q is not None, "OK-DMD states are not initialized."
        
        # Periodically realign the orthonormal projection basis Q
        if force_update or (self.step_counter % self.update_interval == 0):
            Y_qr = torch.matmul(self.A, self.Q)
            self.Q, _ = torch.linalg.qr(Y_qr)  # Real Schur vector approximation
            
        # Project keys onto the persistent subspace: [B, H, N, d] x [B, H, d, k] -> [B, H, N, k]
        projected = torch.matmul(K_cache, self.Q)
        
        # Sum of energy along the k modes
        scores = torch.sum(projected ** 2, dim=-1)
        return scores

    def get_eviction_mask(self, scores: torch.Tensor, target_budget: int) -> torch.Tensor:
        """
        Generates a boolean mask indicating which keys to keep in the cache.
        Attention sinks (start) and the sliding window (end) are automatically protected.

        Args:
            scores: Survival scores tensor of shape [B, H_kv, Cache_Size].
            target_budget: Maximum allowed size of the KV-cache.

        Returns:
            Boolean mask of shape [B, H_kv, Cache_Size]. True indicates retention.
        """
        B, H, N = scores.shape
        
        # If current cache size is already within budget, keep everything
        if N <= target_budget:
            return torch.ones_like(scores, dtype=torch.bool)
            
        mask = torch.zeros_like(scores, dtype=torch.bool)
        
        # 1. Protect attention sinks (always retain start)
        mask[:, :, :self.sink_size] = True
        
        # 2. Protect sliding window (always retain recent context)
        mask[:, :, -self.window_size:] = True
        
        # 3. Determine remaining budget for middle tokens
        remaining_slots = target_budget - self.sink_size - self.window_size
        if remaining_slots <= 0:
            return mask  # Budget is too strict, only protected regions survive
            
        # Isolate the middle pool (candidates for eviction)
        middle_scores = scores[:, :, self.sink_size : -self.window_size]
        
        # Select the top-K indices with the highest semantic scores
        _, topk_indices = torch.topk(middle_scores, k=remaining_slots, dim=-1)
        
        # Map indices back to original global coordinates
        global_indices = topk_indices + self.sink_size
        
        # Scatter True into the mask at selected global indices
        mask.scatter_(dim=-1, index=global_indices, value=True)
        
        return mask
