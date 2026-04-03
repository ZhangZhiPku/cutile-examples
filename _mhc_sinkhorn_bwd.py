# cutile-lsp: on
from pathlib import Path


# cutile-lsp: start
import cuda.tile as ct


EPS = 1e-10


tilesize = 32


@ct.function(host=False, tile=True)
def matvec_S(R, x):
    """
    Compute S @ x = (I - R^T R) x = x - R^T (R x)
    without materializing R^T R.

    R: (tilesize, n_stream, n_stream)
    x: (tilesize, n_stream, 1)
    returns: (tilesize, n_stream, 1)
    """
    Rx = ct.matmul(R, x)  # q = R x, (tilesize, n_stream, 1)
    RTRx = ct.matmul(R.transpose(-2, -1), Rx)  # R^T q, (tilesize, n_stream, 1)
    return x - RTRx


@ct.function(host=False, tile=True)
def dot(a, b):  # a/b: (..., dim, 1)
    return ct.matmul(a.transpose(-2, -1), b)


@ct.kernel
def sinkhorn_knopp_bwd_implicit_cg_opt1(
    out,
    dout,
    res,
    n_stream: ct.Constant[int],
):
    """
    <typecheck>
    Tensor((1024, 4, 4), dtype="float32")
    Tensor((1024, 4, 4), dtype="float32")
    Tensor((1024, 4, 4), dtype="float32")
    4
    </typecheck>

    Side note:
    1. Number of CG iterations is typically num_streams.
        This is derived from the theoretical properties of CG method.
    2. Matrix R is theoretically singular (not full-rank) and numerically near-singular,
        so the solution of x_sol can be very different from the real solution x_real.
        However, the tensor sum of the first half and the second half of x_sol is same with the result of x_real, which **is what we need**.
        This means the solution set has some mathematical property that applies to every element in it.
        We shall make use of that property.
    """

    i_seq = ct.bid(0)

    R = ct.load(
        out,
        index=(i_seq, 0, 0),
        shape=(tilesize, n_stream, n_stream),
    )
    dR = ct.load(
        dout,
        index=(i_seq, 0, 0),
        shape=(tilesize, n_stream, n_stream),
    )

    RdR = R * dR
    # r_vec = (G⊙R) 1  (row sums),  shape (tilesize, n_stream, 1)
    r_vec = ct.sum(RdR, axis=-1).reshape((tilesize, n_stream, 1))
    # c_vec = (G⊙R)^T 1  (col sums), shape (tilesize, n_stream, 1)
    c_vec = ct.sum(RdR, axis=-2).reshape((tilesize, n_stream, 1))

    # b = c_vec - R^T r_vec
    b = c_vec - ct.matmul(R.transpose(-2, -1), r_vec)

    # Solve: (I - R^T R) x = b  via CG ========================
    x = ct.zeros((tilesize, n_stream, 1), dtype=ct.float32)
    r = b - matvec_S(R, x)
    p = r
    r_normsq = dot(r, r)

    # n iterations suffice for an n×n system
    for _ in range(n_stream):
        Sp = matvec_S(R, p)
        pSp = dot(p, Sp)
        alpha = r_normsq / (pSp + EPS)
        x += alpha * p
        r -= alpha * Sp
        r_new_normsq = dot(r, r)
        beta = r_new_normsq / (r_normsq + EPS)
        p = r + beta * p
        r_normsq = r_new_normsq
    # End solve ================================================

    # u = r_vec - R x,  v = x
    u = r_vec - ct.matmul(R, x)  # (tilesize, n_stream, 1)
    v = x  # (tilesize, n_stream, 1)

    # res = (dR - u_i - v_j) * R
    res_tile = (dR - u.reshape((tilesize, n_stream, 1)) - v.reshape((tilesize, 1, n_stream))) * R

    ct.store(
        res,
        index=(i_seq, 0, 0),
        tile=res_tile,
    )


# cutile-lsp: end
import torch


def sinkhorn_forward(M, iters=20):
    P = torch.exp(M)
    R = P

    for _ in range(iters):
        R = R / R.sum(-2, keepdim=True)
        R = R / R.sum(-1, keepdim=True)

    return R, P


seqlen = 65536
n_stream = 4

######################################################################
# Variable
######################################################################
dist = torch.distributions.uniform.Uniform(0.0, 4.0)
device = torch.device("cuda")
M = dist.sample((seqlen, n_stream, n_stream)).to(device)
M.requires_grad_()


######################################################################
# Shared forward + one shared loss weight
######################################################################
R, P = sinkhorn_forward(M, iters=20)
loss_weight = torch.randn_like(R)

######################################################################
# Method A: Autograd
######################################################################
loss_a = (R * loss_weight).sum()
loss_a.backward()
grad_M_autograd = M.grad.detach().clone()

######################################################################
# Method B: Implicit differentiation opt1 (n×n implicit matvec_S)
######################################################################
grad_R = loss_weight
grad_M_implicit = torch.empty_like(R)
ct.launch(
    torch.cuda.current_stream(0),
    (seqlen // tilesize,),
    sinkhorn_knopp_bwd_implicit_cg_opt1,
    [R, grad_R, grad_M_implicit, n_stream],
)

######################################################################
# Compare
######################################################################
g1 = grad_M_autograd
g2 = grad_M_implicit

abs_diff = (g1 - g2).abs()
rel_diff = abs_diff / (g1.abs() + 1e-12)

print("Comparison of gradients dL/dM")
print("--------------------------------")

MAE = abs_diff.mean(dim=(-1, -2)).tolist()
max_abs_diff = abs_diff.reshape(seqlen, -1).max(-1).values.tolist()
mean_rel_diff = rel_diff.mean(dim=(-1, -2)).tolist()
max_rel_diff = rel_diff.reshape(seqlen, -1).max(-1).values.tolist()

print(f"Max MAE = {max(MAE)}")
print(f"Max max_abs_diff = {max(max_abs_diff)}")
print(f"Max mean_rel_diff = {max(mean_rel_diff)}")
print(f"Max max_rel_diff = {max(max_rel_diff)}")

print("\nGrad (autograd) sample:\n", g1[0, :3, :3])
print("\nGrad (implicit) sample:\n", g2[0, :3, :3])
