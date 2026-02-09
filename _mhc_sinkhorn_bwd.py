# cutile-lsp: on
from pathlib import Path


# cutile-lsp: start
import cuda.tile as ct


EPS = 1e-10


tilesize = 32


@ct.function(host=False, tile=True)
def matvec_A(R, x, n_stream: ct.Constant[int]):
    """
    R: (tilesize, n_stream, n_stream)
    x: (tilesize, n_stream*2, 1)
    """
    x1 = ct.extract(x, index=(0, 0, 0), shape=(tilesize, n_stream, 1))
    x2 = ct.extract(x, index=(0, 1, 0), shape=(tilesize, n_stream, 1))
    ax1 = x1 + ct.matmul(R, x2)
    ax2 = ct.matmul(R.transpose(-2, -1), x1) + x2
    return ct.cat((ax1, ax2), axis=-2)  # (tilesize, n_stream*2, 1)


@ct.function(host=False, tile=True)
def dot(a, b):  # a/b: (..., dim, 1)
    return ct.matmul(a.transpose(-2, -1), b)


@ct.kernel
def sinkhorn_knopp_bwd_implicit_cg(
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
    1. Number of CG iterations is typically num_streams*2.
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
    # row sum
    b1 = ct.sum(RdR, axis=-1).reshape((tilesize, n_stream, 1))
    # col sum
    b2 = ct.sum(RdR, axis=-2).reshape((tilesize, n_stream, 1))

    b = ct.cat((b1, b2), axis=-2)

    # Solve: Ax=b =========================================
    R = R.reshape((tilesize, n_stream, n_stream))
    # Conjugate Gradients: init
    x = ct.zeros((tilesize, n_stream * 2, 1), dtype=ct.float32)
    r = b - matvec_A(R, x, n_stream=n_stream)
    p = r
    r_normsq = dot(r, r)

    # Conjugate Gradients: iter
    num_iter_cg = n_stream * 2
    for _ in range(num_iter_cg):
        Ap = matvec_A(R, p, n_stream=n_stream)
        pAp = dot(p, Ap)
        # VERY important to avoid divide by zero
        alpha = r_normsq / (pAp + EPS)
        x += alpha * p
        r -= alpha * Ap
        r_new_normsq = dot(r, r)
        # not very important to avoid divide by zero, but it's good to have it
        beta = r_new_normsq / (r_normsq + EPS)
        p = r + beta * p
        r_normsq = r_new_normsq
    # End solve: Ax=b =========================================

    x1 = ct.extract(x, index=(0, 0, 0), shape=(tilesize, n_stream, 1))
    x2 = ct.extract(x, index=(0, 1, 0), shape=(tilesize, n_stream, 1))

    x1_expanded = x1.reshape((tilesize, n_stream, 1))
    x2_expanded = x2.reshape((tilesize, 1, n_stream))

    res_tile = dR - x1_expanded - x2_expanded
    res_tile = res_tile * R

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
# Method B: Implicit differentiation
######################################################################
grad_R = loss_weight
grad_M_implicit = torch.empty_like(R)
ct.launch(
    torch.cuda.current_stream(0),
    (seqlen // tilesize,),
    sinkhorn_knopp_bwd_implicit_cg,
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


def format_list(ls):
    return [f"{x:.2e}" for x in ls]


MAE = abs_diff.mean(dim=(-1, -2)).tolist()
max_abs_diff = abs_diff.reshape(seqlen, -1).max(-1).values.tolist()
mean_rel_diff = rel_diff.mean(dim=(-1, -2)).tolist()
max_rel_diff = rel_diff.reshape(seqlen, -1).max(-1).values.tolist()

# print(f"MAE: {format_list(MAE)}")
# print(f"max_abs_diff: {format_list(max_abs_diff)}")
# print(f"mean_rel_diff: {format_list(mean_rel_diff)}")
# print(f"max_rel_diff: {format_list(max_rel_diff)}")

print(f"Max MAE = {max(MAE)}")
print(f"Max max_abs_diff = {max(max_abs_diff)}")
print(f"Max mean_rel_diff = {max(mean_rel_diff)}")
print(f"Max max_rel_diff = {max(max_rel_diff)}")

print("\nGrad (autograd) sample:\n", g1[0, :3, :3])
print("\nGrad (implicit) sample:\n", g2[0, :3, :3])
