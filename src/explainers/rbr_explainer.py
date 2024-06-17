import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.utils import check_random_state
import torch
import pandas as pd
import numpy as np
from torch import nn

from .base_explainer import BaseExplainer

class ModelWrap:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        return self.model.predict_crisp(x)

    def predict_proba(self, x):
        out = np.zeros((x.shape[0], 2))
        out[:, 1] = self.model.predict_proba(x)
        out[:, 0] = 1 - out[:, 1]
        return out

class RBRExplainer(BaseExplainer):

    def __init__(self, 
        train_dataset: pd.DataFrame, 
        explained_model: nn.Module,
    ) -> None:
        
        model_wrap = ModelWrap(explained_model)
        self.model = model_wrap
        self.train_data = train_dataset.to_numpy()
    
    def prep(self) -> None:
        '''
        Prepare the explainer.
        '''
        # RBR CONFIG
        self.config = {
            "rbr_params": {
                "delta_plus": 0.2,
                "sigma": 1.0,
                "epsilon_op": 0.0,
                "epsilon_pe": 0.0,
            },
            "ec": {
                "num_samples": 100,
                "max_distance": 1.0,
                "rbr_params": {
                    "delta_plus": 0.2,
                    "sigma": 1.0,
                    "epsilon_op": 0.0,
                    "epsilon_pe": 0.0,
                },
            },
            "perturb_radius": {
                "synthesis": 0.2,
                "german": 0.2,
                "sba": 0.2,
                "gmc": 0.2,
            },
            "params_to_vary": {
                "delta_max": {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 0.2,
                    "step": 0.02,
                },
                "epsilon_op": {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.5,
                },
                "epsilon_pe": {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.5,
                },
                "delta_plus": {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.2,
                },
                "none": {"min": 0.0, "max": 0.0, "step": 0.1},
            },
            "kfold": 5,
            "num_future": 100,
            "perturb_std": 1.0,
            "num_samples": 200,
            "max_ins": 100,
            "max_distance": 1.0,
            "train_data": self.train_data,
            "device": "cuda",
        }

    def generate(self, query_instance: pd.DataFrame) -> np.ndarray:
        '''
        Generate counterfactuals.
        '''
        if isinstance(query_instance, pd.DataFrame):
            query_instance = query_instance.to_numpy().flatten()
            
        arg = RBR(self.model, self.train_data, num_cfacts=1, max_iter=500, random_state=123, device=self.config["device"])
        x_ar = arg.fit_instance(
            query_instance,
            self.config["ec"]["num_samples"],
            self.config["perturb_radius"]["synthesis"] * self.config["ec"]["max_distance"],
            self.config["ec"]["rbr_params"]["delta_plus"],
            self.config["ec"]["rbr_params"]["sigma"],
            self.config["ec"]["rbr_params"]["epsilon_op"],
            self.config["ec"]["rbr_params"]["epsilon_pe"],
            self.config["ec"],
        )
        return x_ar


@torch.no_grad()
def l2_projection(x, radius):
    """
    Euclidean projection onto an L2-ball
    """
    norm = torch.linalg.norm(x, ord=2, axis=-1)
    return (radius * 1 / torch.max(radius, norm)).unsqueeze(1) * x


class OptimisticLikelihood(torch.nn.Module):
    def __init__(self, x_dim, epsilon_op, sigma, device="cpu"):
        super(OptimisticLikelihood, self).__init__()
        self.device = device
        self.x_dim = x_dim.to(self.device)
        self.epsilon_op = epsilon_op.to(self.device)
        self.sigma = sigma.to(self.device)

    @torch.no_grad()
    def projection(self, v):
        v = v.clone()
        v = torch.max(v, torch.tensor(0, device=self.device))

        result = l2_projection(v, self.epsilon_op)

        return result.to(self.device)

    def _forward(self, v, x, x_feas):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = v[..., 1] + self.sigma
        p = self.x_dim

        L = torch.log(d) + (c - v[..., 0]) ** 2 / (2 * d**2) + (p - 1) * torch.log(self.sigma)

        return L

    def forward(self, v, x, x_feas):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = v[..., 1] + self.sigma
        p = self.x_dim

        L = torch.log(d) + (c - v[..., 0]) ** 2 / (2 * d**2) + (p - 1) * torch.log(self.sigma)

        v_grad = torch.zeros_like(v, device=self.device)
        v_grad[..., 0] = -(c - v[..., 0]) / d**2
        v_grad[..., 1] = 1 / d - (c - v[..., 0]) ** 2 / d**3

        return L, v_grad

    def optimize(self, x, x_feas, theta=0.7, beta=1, max_iter=int(1e3), verbose=False):
        v = torch.zeros([x.shape[0], 2], device=self.device)

        loss_diff = 1.0
        min_loss = float("inf")
        num_stable_iter = 0
        max_stable_iter = 10

        for t in range(max_iter):
            F, grad = self.forward(v, x, x_feas)
            v = self.projection(v - 1 / torch.sqrt(torch.tensor(max_iter, device=self.device)) * grad)

            loss_sum = F.sum().data.item()
            loss_diff = min_loss - loss_sum

            if loss_diff <= 1e-10:
                num_stable_iter += 1
                if num_stable_iter >= max_stable_iter:
                    break
            else:
                num_stable_iter = 0

            min_loss = min(min_loss, loss_sum)

            if verbose:
                print("sopf: iter: ", t)
                print("sopf: \tloss: %f, min loss: %f" % (loss_sum, min_loss))
                print("sopf: v: ", v)
                print("sopf: grad: ", grad)

        return v


class PessimisticLikelihood(torch.nn.Module):
    def __init__(self, x_dim, epsilon_pe, sigma, device="cpu"):
        super(PessimisticLikelihood, self).__init__()
        self.device = device
        self.epsilon_pe = epsilon_pe.to(self.device)
        self.sigma = sigma.to(self.device)
        self.x_dim = x_dim.to(self.device)

    @torch.no_grad()
    def projection(self, u):
        u = u.clone()
        u = torch.max(u, torch.tensor(0, device=self.device))

        result = l2_projection(u, self.epsilon_pe / torch.sqrt(self.x_dim))

        return result.to(self.device)

    def _forward(self, u, x, x_feas, zeta=1e-6):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = u[..., 1] + self.sigma
        p = self.x_dim
        sqrt_p = torch.sqrt(p)
        f = torch.sqrt((zeta + self.epsilon_pe**2 - p * u[..., 0] ** 2 - u[..., 1] ** 2) / (p - 1))

        L = -torch.log(d) - (c + sqrt_p * u[..., 0]) ** 2 / (2 * d**2) - (p - 1) * torch.log(f + self.sigma)

        return L

    def forward(self, u, x, x_feas, zeta=1e-6):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = u[..., 1] + self.sigma
        p = self.x_dim
        sqrt_p = torch.sqrt(p)
        f = torch.sqrt((zeta + self.epsilon_pe**2 - p * u[..., 0] ** 2 - u[..., 1] ** 2) / (p - 1))

        L = -torch.log(d) - (c + sqrt_p * u[..., 0]) ** 2 / (2 * d**2) - (p - 1) * torch.log(f + self.sigma)

        u_grad = torch.zeros_like(u, device=self.device)
        u_grad[..., 0] = -sqrt_p * (c + sqrt_p * u[..., 0]) / d**2 - (p * u[..., 0]) / (f * (f + self.sigma))
        u_grad[..., 1] = -1 / d + (c + sqrt_p * u[..., 0]) ** 2 / d**3 + u[..., 1] / (f * (f + self.sigma))

        return L, u_grad

    def optimize(self, x, x_feas, theta=0.7, beta=1, max_iter=int(1e3), verbose=False):
        u = torch.zeros([x.shape[0], 2], device=self.device)

        loss_diff = 1.0
        min_loss = float("inf")
        num_stable_iter = 0
        max_stable_iter = 10

        for t in range(max_iter):
            F, grad = self.forward(u, x, x_feas)
            u = self.projection(u - 1 / torch.sqrt(torch.tensor(max_iter, device=self.device)) * grad)

            loss_sum = F.sum().data.item()
            loss_diff = min_loss - loss_sum

            if loss_diff <= 1e-10:
                num_stable_iter += 1
                if num_stable_iter >= max_stable_iter:
                    break
            else:
                num_stable_iter = 0

            min_loss = min(min_loss, loss_sum)

            if verbose:
                print("sopf: iter: ", t)
                print("sopf: \tloss: %f, min loss: %f" % (loss_sum, min_loss))
                print("sopf: u: ", u)
                print("sopf: grad: ", grad)

        return u


class RBRLoss(torch.nn.Module):
    def __init__(self, X_feas, X_feas_pos, X_feas_neg, epsilon_op, epsilon_pe, sigma, device="cpu", verbose=False):
        super(RBRLoss, self).__init__()
        self.device = device
        self.verbose = verbose

        self.X_feas = X_feas.to(self.device)
        self.X_feas_pos = X_feas_pos.to(self.device)
        self.X_feas_neg = X_feas_neg.to(self.device)

        self.epsilon_op = torch.tensor(epsilon_op).to(self.device)
        self.epsilon_pe = torch.tensor(epsilon_pe).to(self.device)
        self.sigma = torch.tensor(sigma).to(self.device)
        self.x_dim = torch.tensor(X_feas.shape[-1]).to(self.device)

        self.op_likelihood = OptimisticLikelihood(self.x_dim, self.epsilon_op, self.sigma, self.device)
        self.pe_likelihood = PessimisticLikelihood(self.x_dim, self.epsilon_pe, self.sigma, self.device)

    def forward(self, x, verbose=False):
        if verbose:
            print(f"N_neg: {self.X_feas_neg.shape}")
            print(f"N_pos: {self.X_feas_pos.shape}")

        u = self.pe_likelihood.optimize(
            x.detach().clone().expand([self.X_feas_pos.shape[0], -1]), self.X_feas_pos, verbose=self.verbose
        )

        F = self.pe_likelihood._forward(u, x.expand([self.X_feas_pos.shape[0], -1]), self.X_feas_pos)
        denom = torch.logsumexp(F, -1)

        if verbose:
            print(f"Pessimistic self.denominator: {denom}")

        v = self.op_likelihood.optimize(
            x.detach().clone().expand([self.X_feas_neg.shape[0], -1]), self.X_feas_neg, verbose=self.verbose
        )

        F = self.op_likelihood._forward(v, x.expand([self.X_feas_neg.shape[0], -1]), self.X_feas_neg)
        numer = torch.logsumexp(-F, -1)

        if verbose:
            print(f"Optimistic numerator: {numer}")

        result = numer - denom

        return result, denom, numer



plt.gca().set_aspect("equal", adjustable="box")
obj_fig, obj_ax = plt.subplots()
point_fig, point_ax = plt.subplots()
pe_fig, pe_ax = plt.subplots()
op_fig, op_ax = plt.subplots()
grad0_fig, grad0_ax = plt.subplots()
grad1_fig, grad1_ax = plt.subplots()


def power10floor(x):
    return 10 ** math.floor(math.log10(x))


def reconstruct_encoding_constraints(x, cat_pos):
    x_enc = x.clone()
    for pos in cat_pos:
        x_enc.data[pos] = torch.clamp(torch.round(x_enc[pos]), 0, 1)
    return x_enc


class RBR(object):
    """Class for generate counterfactual samples for framework: Wachter"""

    DECISION_THRESHOLD = 0.5

    def __init__(self, model, train_data, y_target=1, num_cfacts=10, max_iter=1000, random_state=None, device="cuda"):
        self.random_state = check_random_state(random_state)
        self.model = model
        self.max_iter = max_iter
        self.y_target = y_target
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = "cpu"
        self.feasible = True
        self.train_data = torch.tensor(train_data).float()
        self.train_label = self.make_prediction(self.train_data).to(self.device)
        self.train_data = self.train_data.to(self.device)
        self.num_cfacts = num_cfacts

    def make_prediction(self, x):
        return torch.tensor(self.model.predict(x.cpu().detach().numpy()))

    def dist(self, a, b):
        return torch.linalg.norm(a - b, ord=1, axis=-1)

    def find_x_boundary(self, x, k=None):
        k = k or self.num_cfacts
        x_label = self.make_prediction(x)

        d = self.dist(self.train_data, x)
        order = torch.argsort(d)
        x_cfact_list = self.train_data[order[self.train_label[order] == (1 - x_label)]][:k]
        best_x_b = None
        best_dist = torch.tensor(float("inf"))

        for x_cfact in x_cfact_list:
            lambd_list = torch.linspace(0, 1, 100)
            for lambd in lambd_list:
                x_b = (1 - lambd) * x + lambd * x_cfact
                label = self.make_prediction(x_b)
                if label == 1 - x_label:
                    dist = self.dist(x, x_b)
                    if dist < best_dist:
                        best_x_b = x_b
                        best_dist = dist
                    break
        return best_x_b, best_dist

    def uniform_ball(self, x, r, n, random_state=None):
        # muller method
        rng = check_random_state(random_state)
        d = len(x)
        V_x = rng.randn(n, d)
        V_x = V_x / np.linalg.norm(V_x, axis=1).reshape(-1, 1)
        V_x = V_x * (rng.random(n) ** (1.0 / d)).reshape(-1, 1)
        V_x = V_x * r + x.numpy()
        return torch.from_numpy(V_x).to(self.device)

    def feasible_set(self, x, radius=0.3, num_samples=20, random_state=None):
        return self.uniform_ball(x.cpu(), radius, num_samples, random_state)

    def simplex_projection(self, x, delta):
        """
        Euclidean projection on a positive simplex
        """
        (p,) = x.shape
        if torch.linalg.norm(x, ord=1) == delta and torch.all(x >= 0):
            return x
        u, _ = torch.sort(x, descending=True)
        cssv = torch.cumsum(u, 0)
        rho = torch.nonzero(u * torch.arange(1, p + 1).to(self.device) > (cssv - delta))[-1, 0]
        theta = (cssv[rho] - delta) / (rho + 1.0)
        w = torch.clip(x - theta, min=0)
        return w

    def projection(self, x, delta):
        """
        Euclidean projection on an L1-ball
        """
        x_abs = torch.abs(x)
        if x_abs.sum() <= delta:
            return x

        proj = self.simplex_projection(x_abs, delta=delta)
        proj *= torch.sign(x)

        return proj

    def optimize(self, x, delta, loss_fn, theta=0.7, beta=1, verbose=False):
        x_t = x.detach().clone()
        x_t.requires_grad_(True)

        min_loss = float("inf")
        num_stable_iter = 0
        max_stable_iter = 10

        F_list = []
        pe_list = []
        op_list = []
        grad0_list = []
        grad1_list = []

        for t in range(self.max_iter):

            if x_t.grad is not None:
                x_t.grad.data.zero_()
            F, pe, op = loss_fn(x_t)

            F.backward()
            if verbose:
                print(f"Iteration {t+1}/{self.max_iter}")
                print(x_t)
                print(x_t.grad)
                print(F)
                print(op)
                print(pe)
                print(self.dist(x_t.detach().cpu(), self.x0.detach().cpu()))
                if x_t.shape[0] == 2:
                    x_t_handle = x_t.detach()
                else:
                    x_t_handle = x_t.detach().unsqueeze(0) @ self.pca
                    x_t_handle = x_t_handle.squeeze()
                point_ax.scatter(x_t_handle[0].cpu().numpy(), x_t_handle[1].cpu().numpy(), color="black")
                point_ax.arrow(
                    x_t_handle[0].cpu().numpy(),
                    x_t_handle[1].cpu().numpy(),
                    1 / np.sqrt(1e3) * x_t.grad[0].detach().cpu().numpy(),
                    1 / np.sqrt(1e3) * x_t.grad[1].detach().cpu().numpy(),
                    head_width=0.05,
                    head_length=0.1,
                )
                point_fig.savefig("points.png")
                F_list.append(F.detach().cpu().numpy())
                pe_list.append(pe.detach().cpu().numpy())
                op_list.append(op.detach().cpu().numpy())
                grad0_list.append(x_t.grad.data[0].item())
                grad1_list.append(x_t.grad.data[1].item())

            if torch.ge(self.dist(x_t.detach(), self.x0), delta):
                break

            with torch.no_grad():
                x_new = x_t - 1 / torch.sqrt(torch.tensor(1e3, device=self.device)) * x_t.grad
                if verbose:
                    if x_new.shape[0] == 2:
                        x_new_handle = x_new.detach()
                    else:
                        x_new_handle = x_new.detach().unsqueeze(0) @ self.pca
                        x_new_handle = x_new_handle.squeeze()
                    point_ax.scatter(x_new_handle[0].cpu().numpy(), x_new_handle[1].cpu().numpy(), color="cyan")
                    point_fig.savefig("points.png")

                x_new = self.projection(x_new - self.x0, delta) + self.x0  # shift to origin before project

            for i, elem in enumerate(x_new.data):
                x_t.data[i] = elem

            if verbose:
                obj_ax.plot(F_list)
                obj_fig.savefig("loss.png")
                pe_ax.plot(pe_list)
                pe_fig.savefig("pessimistic.png")
                op_ax.plot(op_list)
                op_fig.savefig("optimistic.png")
                grad0_ax.plot(grad0_list)
                grad0_fig.savefig("grad0.png")
                grad1_ax.plot(grad1_list)
                grad1_fig.savefig("grad1.png")

            loss_sum = F.sum().data.item()
            loss_diff = min_loss - loss_sum

            if loss_diff <= 1e-10:
                num_stable_iter += 1
                if num_stable_iter >= max_stable_iter:
                    break
            else:
                num_stable_iter = 0

            min_loss = min(min_loss, loss_sum)

        return F, x_t.detach()

    def fit_instance(
        self, x0, num_samples, perturb_radius, delta_plus, sigma, epsilon_op, epsilon_pe, ec, verbose=False
    ):
        x0 = torch.from_numpy(x0.copy()).float().to(self.device)
        self.x0 = x0
        self.delta_plus = delta_plus
        x = x0.clone()
        x_b, delta_base = self.find_x_boundary(x)

        x_b = x_b.detach().clone()
        delta_base = delta_base.detach().clone()
        delta = delta_base + delta_plus

        X_feas = self.feasible_set(
            x_b, radius=perturb_radius, num_samples=num_samples, random_state=self.random_state
        ).float()

        if verbose:
            _, _, self.pca = torch.pca_lowrank(X_feas)
            self.pca = self.pca[:, :2]

        y_feas = self.make_prediction(X_feas).to(self.device)

        X_feas_pos = X_feas[y_feas == self.y_target].reshape([sum(y_feas == self.y_target), -1])
        X_feas_neg = X_feas[y_feas == (1 - self.y_target)].reshape([sum(y_feas == (1 - self.y_target)), -1])

        if verbose:
            if x0.shape[0] == 2:
                x0_handle = x0.detach()
            else:
                x0_handle = x0.detach().unsqueeze(0) @ self.pca
                x0_handle = x0_handle.squeeze()

            if X_feas_pos.shape[1] == 2:
                X_feas_pos_handle = X_feas_pos
            else:
                X_feas_pos_handle = X_feas_pos.detach() @ self.pca

            if X_feas_neg.shape[1] == 2:
                X_feas_neg_handle = X_feas_neg
            else:
                X_feas_neg_handle = X_feas_neg.detach() @ self.pca

            point_ax.scatter(x0_handle[0].cpu().numpy(), x0_handle[1].cpu().numpy(), color="blue")
            point_ax.scatter(
                [x[0].cpu().numpy() for x in X_feas_neg_handle],
                [x[1].cpu().numpy() for x in X_feas_neg_handle],
                color="red",
            )
            point_ax.scatter(
                [x[0].cpu().numpy() for x in X_feas_pos_handle],
                [x[1].cpu().numpy() for x in X_feas_pos_handle],
                color="green",
            )
            point_fig.savefig("points.png")

        loss_fn = RBRLoss(X_feas, X_feas_pos, X_feas_neg, epsilon_op, epsilon_pe, sigma, device=self.device)

        loss, x = self.optimize(x_b, delta, loss_fn, verbose=verbose)

        self.feasible = self.make_prediction(x) == self.y_target
        return x.cpu().detach().numpy().squeeze()


def generate_recourse(x0, model, random_state, params=dict()):
    train_data = params["train_data"]

    ec = params["config"]
    arg = RBR(model, train_data, num_cfacts=1000, max_iter=500, random_state=random_state, device=params["device"])

    x_ar = arg.fit_instance(
        x0,
        ec.num_samples,
        params["perturb_radius"] * ec.max_distance,
        ec.rbr_params["delta_plus"],
        ec.rbr_params["sigma"],
        ec.rbr_params["epsilon_op"],
        ec.rbr_params["epsilon_pe"],
        ec,
    )
    report = dict(feasible=arg.feasible)

    return x_ar, report
