"""One FAIR-style NGF term on already-warped intensities, not gradient transport.

The exact stencil and image-only edge rule are declared in
docs/coordinated_instance_registration/OPTIMIZER_REDESIGN.md. No epsilon term
is added to the dot-product numerator. Finite epsilon need not give a zero
identity loss or a stationary identity deformation.
"""
from __future__ import annotations

import torch


def unit_canvas_gradient(image: torch.Tensor) -> torch.Tensor:
    """B1HW -> B2HW (dx,dy), intensity per normalized canvas length.

    Central differences in the interior; explicit first-order one-sided
    differences at BOTH endpoints. Pixel centers have hx=1/W, hy=1/H.
    Constants differentiate to zero and affine intensity ramps exactly.
    """
    if image.ndim!=4 or image.shape[1]!=1 or min(image.shape[-2:])<2:
        raise ValueError("B1HW intensity with H,W>=2 required")
    height,width=image.shape[-2:]
    dx=torch.cat(((image[...,1:2]-image[...,:1])*width,
                  (image[...,2:]-image[...,:-2])*(width/2),
                  (image[...,-1:]-image[...,-2:-1])*width),dim=-1)
    dy=torch.cat(((image[:,:,1:2,:]-image[:,:,:1,:])*height,
                  (image[:,:,2:,:]-image[:,:,:-2,:])*(height/2),
                  (image[:,:,-1:,:]-image[:,:,-2:-1,:])*height),dim=-2)
    return torch.cat((dx,dy),dim=1)


def squared_dot_ngf_errors(fixed_gradient,moving_gradient,epsilon_fixed,epsilon_moving):
    """Exact no-augmentation squared-dot formula; no detach or value clamp."""
    dot=(fixed_gradient*moving_gradient).sum(1,keepdim=True)
    norm_fixed=fixed_gradient.square().sum(1,keepdim=True)+epsilon_fixed.square()
    norm_moving=moving_gradient.square().sum(1,keepdim=True)+epsilon_moving.square()
    return 1-dot.square()/(norm_fixed*norm_moving)


class FrozenPostwarpNGF:
    """Cache only fixed-image gradients and frozen per-scale edge thresholds.

    ``initial_warped`` is original moving intensity sampled once through the
    initial affine. It is used ONLY to calibrate epsilon_moving; every trial
    passes its freshly warped original moving intensity to ``errors``.
    """

    def __init__(self,fixed,initial_warped,mask):
        if (fixed.ndim!=4 or fixed.shape[:2]!=(1,1)
                or fixed.shape[-1]!=fixed.shape[-2] or fixed.shape[-1]<2
                or initial_warped.shape!=fixed.shape or mask.shape!=fixed.shape
                or fixed.dtype not in (torch.float32,torch.float64)
                or any(t.dtype!=fixed.dtype or t.device!=fixed.device or t.requires_grad
                       for t in (initial_warped,mask)) or fixed.requires_grad):
            raise ValueError("frozen equal-dtype 11SS intensities/mask required")
        if (not all(bool(torch.isfinite(t).all()) for t in (fixed,initial_warped,mask))
                or not bool(((mask>=0)&(mask<=1)).all()) or not float(mask.sum())>0):
            raise ValueError("finite frozen inputs and nonempty fixed mask in [0,1] required")
        self.fixed_gradient=unit_canvas_gradient(fixed)
        initial_gradient=unit_canvas_gradient(initial_warped)
        side=fixed.shape[-1];denominator=mask.sum()
        floor=fixed.new_tensor(side/255.)
        def calibrate(gradient):
            mean=(gradient.square().sum(1,keepdim=True).sqrt()*mask).sum()/denominator
            return torch.maximum(floor,.1*mean),mean
        self.epsilon_fixed,mean_fixed=calibrate(self.fixed_gradient)
        self.epsilon_moving,mean_moving=calibrate(initial_gradient)
        self.metadata=dict(raster_side=side,
            epsilon_fixed=float(self.epsilon_fixed),epsilon_moving=float(self.epsilon_moving),
            mean_fixed_gradient=float(mean_fixed),mean_initial_warped_gradient=float(mean_moving),
            epsilon_floor=float(floor),gradient_units="intensity per normalized canvas length",
            stencil="central interior; first-order one-sided at both endpoints; hx=hy=1/side",
            edge_rule="max(side/255, .1 fixed-mask mean gradient norm), independently per frozen input",
            moving_calibration="original moving intensity sampled ONCE through initial affine; frozen thereafter",
            data_formula="1 - dot(grad_fixed,grad_warped)^2 / ((norm_fixed^2+eps_fixed^2)*(norm_warped^2+eps_moving^2))",
            numerator_epsilon_added=False,
            fixed_mask_denominator=float(denominator),
            derivative="differentiate spatial gradient of freshly warped intensity; no moving-gradient transport or detach",
            flat_region_caution="flat moving intensity has zero NGF image gradient, even against a fixed edge",
            identity_caution="finite epsilon need not give zero identical-image loss or stationary map gradient")

    def errors(self,warped):
        if warped.shape[0]!=1 or warped.shape[1]!=1 or warped.shape[-2:]!=self.fixed_gradient.shape[-2:]:
            raise ValueError("fresh warped intensity must match frozen raster")
        return squared_dot_ngf_errors(self.fixed_gradient,unit_canvas_gradient(warped),
                                      self.epsilon_fixed,self.epsilon_moving)
