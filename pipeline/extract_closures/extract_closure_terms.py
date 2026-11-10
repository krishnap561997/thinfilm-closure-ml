#!/usr/bin/env python3
"""Extract DNS thin-film variables and Lavalle discrepancy closures.

Each Basilisk snapshot contains fields at t and t-dt. Output field datasets have
shape (nx, nt); *_m denotes the t-dt quantity stored in the same snapshot.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import h5py
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as f: c=json.load(f)
    req=("snapshot_dir","output_file","start_time","end_time","delta_time")
    miss=[k for k in req if k not in c]
    if miss: raise KeyError("Missing configuration entries: "+", ".join(miss))
    return c

def build_snapshot_values(start,end,step):
    if step<=0 or end<start: raise ValueError("Invalid snapshot range.")
    n=int(np.floor((end-start)/step+1e-10))+1
    return start+step*np.arange(n,dtype=float)

def format_snapshot_filename(value,prefix="snapshot_",suffix=".h5",width=6,decimals=0):
    label=f"{value:0{width}.0f}" if decimals==0 else f"{value:0{width}.{decimals}f}"
    return f"{prefix}{label}{suffix}"

def compute_cell_centers(points,topology): return np.asarray(points)[np.asarray(topology,dtype=np.int64)].mean(axis=1)

def build_cartesian_cell_grid(cc,decimals=12):
    x,y=cc[:,0],cc[:,1]; xr=np.round(x,decimals); yr=np.round(y,decimals)
    ux,uy=np.unique(xr),np.unique(yr); nx,ny=len(ux),len(uy)
    if nx*ny!=len(cc): raise ValueError("Cell centers do not form a complete Cartesian grid.")
    idx=np.lexsort((xr,yr)); return x[idx].reshape(ny,nx),y[idx].reshape(ny,nx),idx

def scalar_to_matrix(a,idx,shape):
    a=np.asarray(a).squeeze()
    if a.ndim!=1 or a.size!=idx.size: raise ValueError(f"Bad scalar shape {a.shape}")
    return a[idx].reshape(shape)

def vector_x_to_matrix(a,idx,shape):
    a=np.asarray(a); ux=a if a.ndim==1 else a[:,0]
    if ux.size!=idx.size: raise ValueError(f"Bad vector shape {a.shape}")
    return ux[idx].reshape(shape)

def depth_integral(x,y,field,lower_boundary=0.0):
    ny,nx=field.shape; out=np.zeros(nx)
    for i in range(nx):
        order=np.argsort(y[:,i]); ys=y[order,i]; fs=field[order,i]
        faces=np.empty(ny+1); faces[0]=lower_boundary
        faces[1:ny]=0.5*(ys[:-1]+ys[1:]); faces[ny]=ys[-1]+0.5*(ys[-1]-ys[-2])
        out[i]=np.sum(fs*np.diff(faces))
    return x[0,:].copy(),out

def integrate_to_height(y,field,h,lower_boundary=0.0):
    """Direct trapezoidal integration of a cell-centered field from wall to y=h(x)."""
    ny,nx=field.shape; out=np.full(nx,np.nan)
    for i in range(nx):
        order=np.argsort(y[:,i]); ys=y[order,i]; fs=field[order,i]
        top=float(h[i]); mask=(ys>lower_boundary)&(ys<top)
        yi=ys[mask]; fi=fs[mask]
        # endpoint values are linearly interpolated/extrapolated by nearest stored value at wall
        f0=np.interp(lower_boundary,ys,fs); ft=np.interp(top,ys,fs)
        yi=np.concatenate(([lower_boundary],yi,[top])); fi=np.concatenate(([f0],fi,[ft]))
        out[i]=np.trapezoid(fi,yi)
    return out

def compute_wall_gradient(x,y,u):
    dy=y[1,0]-y[0,0]
    if dy<=0 or u.shape[0]<4: raise ValueError("Need >=4 uniformly ordered y cells.")
    u1,u2,u3,u4=u[0],u[1],u[2],u[3]
    dudy=(3675*u1-1225*u2+441*u3-75*u4)/(840*dy)
    return x[0,:].copy(),dudy

def first_derivative_x(x,field):
    x=np.asarray(x,float); f=np.asarray(field,float); dx=np.diff(x); d=dx[0]
    if len(x)<5 or not np.allclose(dx,d,rtol=1e-7,atol=1e-12): raise ValueError("Uniform x with >=5 points required.")
    o=np.empty_like(f); o[2:-2]=(f[:-4]-8*f[1:-3]+8*f[3:-1]-f[4:])/(12*d)
    o[0]=(-25*f[0]+48*f[1]-36*f[2]+16*f[3]-3*f[4])/(12*d)
    o[1]=(-3*f[0]-10*f[1]+18*f[2]-6*f[3]+f[4])/(12*d)
    o[-2]=(-f[-5]+6*f[-4]-18*f[-3]+10*f[-2]+3*f[-1])/(12*d)
    o[-1]=(3*f[-5]-16*f[-4]+36*f[-3]-48*f[-2]+25*f[-1])/(12*d)
    return o

def compute_px(x,p):
    x1=x if np.ndim(x)==1 else x[0,:]
    return np.vstack([first_derivative_x(x1,row) for row in p])

def best_savgol_filter(x,values,m_values,p_values,metric="rmse",power=4):
    x=np.asarray(x); values=np.asarray(values)
    results=[]
    for m in m_values:
        for p in p_values:
            if m<3 or m%2==0 or m>len(values) or p<2 or p>=m: continue
            filt=savgol_filter(values,m,p); diff=np.abs(filt-values)
            if metric=="rmse": err=np.sqrt(np.mean(diff**2))
            elif metric=="mae": err=np.mean(diff)
            elif metric=="maxae": err=np.max(diff)
            else: err=np.mean(diff**power)**(1/power)
            results.append((err,m,p))
    if not results: raise ValueError("No valid Savitzky-Golay candidates.")
    results.sort(); err,m,p=results[0]; return m,p,err,results

def sg_derivatives(x,v,max_deriv,min_poly=2,m_values=range(11,42,2),p_values=range(2,4)):
    # h_xxx requires p>=3; enforce that while retaining the same best-filter criterion.
    pv=[p for p in p_values if p>=max(min_poly,max_deriv)]
    m,p,_,_=best_savgol_filter(x,v,m_values,pv,metric="rmse",power=4)
    dx=np.diff(x)[0]
    return tuple(savgol_filter(v,m,p,deriv=d,delta=dx) for d in range(1,max_deriv+1))

# ---------------------------------------------------------------------------
# ΔP_x v2 (decided 2026-11-10, docs/results/dpx-closure.md "Extraction"):
#   I_p,liq = ∂x ∫_0^h p_l dz - p_l(h) h_x     (Leibniz form on the liquid pressure)
#   p_l      = cell pressure in fully liquid cells (y <= h - PX_GAP*Δ); p_l(h) is the
#              linear least-squares extrapolation of the top PX_NFIT such cells
#   P_Lav,s  = h κ_x/We - cosβ/Fr² h h_x + q_xx/Re, κ = h_xx/(1+h_x²)^{3/2}, all from a
#              Gaussian filter of width σ (h_N) with analytic derivatives
#   ΔP_x,v2  = -G_σ(I_p,liq) - P_Lav,s
# Rows cut by the interface (pressure jump) are never differentiated in x.
# ---------------------------------------------------------------------------
PX_NFIT=4; PX_GAP=1.5; PX_SIGMA_DEFAULT=0.5

def _ls_weights(n):
    """LS line through n equally spaced points k=0..n-1, evaluated at s: sum_k (w0_k + s w1_k) v_k."""
    k=np.arange(n,dtype=float); kb=k.mean(); w1=(k-kb)/((k-kb)**2).sum(); w0=1.0/n-kb*w1
    return w0,w1

def liquid_top_index(ycut,dy,ny):
    """Index of the highest cell centre (j+1/2)dy <= ycut, clipped to [PX_NFIT-1, ny-1]."""
    return np.clip(np.floor(ycut/dy-0.5).astype(int),PX_NFIT-1,ny-1)

def extrapolate_top(field,jtop,ytarget,dy):
    """Per column: LS line through rows jtop-PX_NFIT+1..jtop of field (ny,nx), evaluated at ytarget."""
    cols=np.arange(field.shape[1]); w0,w1=_ls_weights(PX_NFIT)
    vals=np.stack([field[jtop-PX_NFIT+1+k,cols] for k in range(PX_NFIT)])
    s=(ytarget-(jtop-PX_NFIT+1.5)*dy)/dy
    return (w0[:,None]*vals).sum(0)+s*(w1[:,None]*vals).sum(0)

def trapz_to(field,jtop,ytop,fwall,top_val,dy):
    """∫_0^ytop of a cell-centred column field: wall value fwall at y=0, trapezoid over
    centres 0..jtop, then a segment from y_jtop to ytop with end value top_val."""
    cols=np.arange(field.shape[1])
    C=np.zeros_like(field); C[1:]=np.cumsum(0.5*(field[1:]+field[:-1])*dy,axis=0)
    return 0.5*(fwall+field[0])*(0.5*dy)+C[jtop,cols]+0.5*(field[jtop,cols]+top_val)*(ytop-(jtop+0.5)*dy)

def ip_liquid_leibniz(x,ym,p,h):
    """Return (I_p,liq, p_l(h)) for cell-centred pressure p (ny,nx) on a uniform grid with wall at y=0."""
    ny=p.shape[0]; dy=ym[1,0]-ym[0,0]
    if not np.allclose(ym[:,0],(np.arange(ny)+0.5)*dy): raise ValueError("ip_liquid_leibniz needs uniform rows with the wall at y=0")
    jb=liquid_top_index(h-PX_GAP*dy,dy,ny)
    pl_h=extrapolate_top(p,jb,h,dy)
    Ipl=trapz_to(p,jb,h,1.5*p[0]-0.5*p[1],pl_h,dy)
    return first_derivative_x(x,Ipl)-pl_h*first_derivative_x(x,h),pl_h

def gauss_deriv(v,sig,dx,order=0):
    return gaussian_filter1d(v,sig/dx,order=order,mode="nearest")/dx**order

def lavalle_px_smooth(h,q,Re,Ca,beta,Lambda,sig,dx):
    """Filtered exact-curvature Lavalle P_x; returns (P_Lav,s, κ_s, κ_x,s)."""
    hs=gauss_deriv(h,sig,dx); hx=gauss_deriv(h,sig,dx,1); hxx=gauss_deriv(h,sig,dx,2)
    hxxx=gauss_deriv(h,sig,dx,3); qxx=gauss_deriv(q,sig,dx,2); g=1.0+hx**2
    kap=hxx/g**1.5; kap_x=hxxx/g**1.5-3.0*hx*hxx**2/g**2.5
    Fr2=Re*np.sin(beta)/Lambda
    return hs*kap_x/(Re*Ca)-np.cos(beta)/Fr2*hs*hx+qxx/Re,kap,kap_x

def px_v2_fields(x,ym,p,h,q,Re,Ca,beta,Lambda,sig):
    dx=np.diff(x)[0]
    Ipl,pl_h=ip_liquid_leibniz(x,ym,p,h)
    Pls,kap,kap_x=lavalle_px_smooth(h,q,Re,Ca,beta,Lambda,sig,dx)
    return {"int_px_dy_liq":Ipl,"p_liq_interface":pl_h,"kappa_s":kap,"kappa_x_s":kap_x,
            "delta_Px_ML_v2":-gauss_deriv(Ipl,sig,dx)-Pls}

V2_FIELDS=("int_px_dy_liq","p_liq_interface","kappa_s","kappa_x_s","delta_Px_ML_v2")
V2_DESCRIPTION=("delta_Px_ML_v2 = -G_s(I_p,liq) - [h kappa_x/We - cos(beta)/Fr2 h h_x + q_xx/Re]_s; "
                "I_p,liq = d/dx int_0^h p_l dz - p_l(h) h_x (liquid cells only, top 4 cells extrapolated, gap 1.5 dy); "
                "G_s Gaussian filter of width px_filter_width (h_N) with analytic derivatives; "
                "kappa_s, kappa_x_s from the filtered h. See docs/results/dpx-closure.md.")


def compute_ui(
    u,
    y,
    h,
    x,
    *,
    k=3,
    prominence=0.05,
    distance=10,
    Z_h_threshold=3.0,
    Z_u_region_threshold=2.0,
    return_diagnostics=False,
):
    """
    Compute interfacial velocity u_i(x) and repair isolated numerical
    discontinuities that are not associated with physical interface
    features in h(x).

    Procedure
    ---------
    1. Compute raw interfacial velocity by interpolating u(x,y) to y=h(x).

    2. Compute local residuals

           r_u(i) = u_i(i)
                    - 0.5 * [u_i(i-k) + u_i(i+k)]

           r_h(i) = h(i)
                    - 0.5 * [h(i-k) + h(i+k)]

    3. Detect significant negative events in r_u using find_peaks(-r_u).

    4. Use the corresponding normalized h residual to distinguish
       interface-associated events from suspected u_i discontinuities.

       If

           |Z_h| >= Z_h_threshold,

       the event is considered associated with real interface behavior
       and is retained.

       Otherwise, it is flagged as a candidate u_i anomaly.

    5. Expand each candidate anomaly into a contaminated region using
       the normalized u_i residual:

           |Z_u| >= Z_u_region_threshold.

    6. Replace the contaminated region by interpolation between
       neighboring valid u_i values.

    Parameters
    ----------
    u : ndarray, shape (ny, nx)
        Streamwise velocity field.

    y : ndarray
        Vertical coordinate. Can be either:
            (ny,)
        or
            (ny, nx)

    h : ndarray, shape (nx,)
        Interface height.

    x : ndarray, shape (nx,)
        Streamwise coordinate corresponding to h and u_i.

    k : int, optional
        Number of grid points on either side used to define the
        local residual. Default is 3.

    prominence : float, optional
        Minimum prominence used by find_peaks to identify negative
        u_i residual events. Default is 0.05.

    distance : int, optional
        Minimum grid-point separation between detected events.
        Default is 10.

    Z_h_threshold : float, optional
        Threshold used to determine whether a detected u_i event is
        accompanied by significant local interface behavior.
        Default is 3.0.

    Z_u_region_threshold : float, optional
        Threshold used to expand a detected anomaly into its
        surrounding contaminated region. Default is 2.0.

    return_diagnostics : bool, optional
        If True, also return a dictionary containing the raw u_i,
        residuals, normalized residuals, event indices, anomaly
        centers, ignored events, and complete bad-region mask.

    Returns
    -------
    u_i : ndarray, shape (nx,)
        Corrected interfacial velocity.

    diagnostics : dict, optional
        Returned only when return_diagnostics=True.
    """

    import numpy as np
    from scipy.signal import find_peaks

    # ========================================================
    # Convert inputs
    # ========================================================

    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    h = np.asarray(h, dtype=float)
    x = np.asarray(x, dtype=float)

    if u.ndim != 2:
        raise ValueError("u must have shape (ny, nx).")

    ny, nx = u.shape

    if h.shape != (nx,):
        raise ValueError(
            f"h must have shape ({nx},), got {h.shape}."
        )

    if x.shape != (nx,):
        raise ValueError(
            f"x must have shape ({nx},), got {x.shape}."
        )

    if y.ndim == 1:
        if y.shape != (ny,):
            raise ValueError(
                f"1D y must have shape ({ny},), got {y.shape}."
            )

    elif y.ndim == 2:
        if y.shape != u.shape:
            raise ValueError(
                "2D y must have the same shape as u."
            )

    else:
        raise ValueError(
            "y must be either 1D or 2D."
        )

    if k < 1:
        raise ValueError("k must be >= 1.")

    if nx <= 2 * k:
        raise ValueError(
            "Not enough x points for the requested k."
        )

    # ========================================================
    # 1. Raw interfacial velocity
    # ========================================================

    u_i_raw = np.full(nx, np.nan)

    for i in range(nx):

        if y.ndim == 1:
            y_col = y
        else:
            y_col = y[:, i]

        u_col = u[:, i]

        valid = (
            np.isfinite(y_col)
            & np.isfinite(u_col)
        )

        if np.count_nonzero(valid) < 2:
            continue

        y_valid = y_col[valid]
        u_valid = u_col[valid]

        # Sort in y in case the grid is not strictly ordered
        order = np.argsort(y_valid)

        y_valid = y_valid[order]
        u_valid = u_valid[order]

        u_i_raw[i] = np.interp(
            h[i],
            y_valid,
            u_valid
        )

    # ========================================================
    # 2. Local residuals
    # ========================================================

    residual_u = np.full(nx, np.nan)
    residual_h = np.full(nx, np.nan)

    residual_u[k:-k] = (
        u_i_raw[k:-k]
        - 0.5 * (
            u_i_raw[:-2*k]
            + u_i_raw[2*k:]
        )
    )

    residual_h[k:-k] = (
        h[k:-k]
        - 0.5 * (
            h[:-2*k]
            + h[2*k:]
        )
    )

    # ========================================================
    # 3. Robust residual scales
    # ========================================================

    def robust_scale(values):

        valid = np.isfinite(values)

        if not np.any(valid):
            return np.nan

        values_valid = values[valid]

        median = np.median(values_valid)

        mad = np.median(
            np.abs(values_valid - median)
        )

        return 1.4826 * mad

    sigma_u = robust_scale(residual_u)
    sigma_h = robust_scale(residual_h)

    if not np.isfinite(sigma_u) or sigma_u <= 0:
        raise ValueError(
            "Unable to determine a non-zero robust scale "
            "for the u_i residual."
        )

    if not np.isfinite(sigma_h) or sigma_h <= 0:
        raise ValueError(
            "Unable to determine a non-zero robust scale "
            "for the h residual."
        )

    Z_u = residual_u / sigma_u
    Z_h = residual_h / sigma_h

    # ========================================================
    # 4. Detect ALL significant negative u_i events
    # ========================================================

    residual_for_peaks = residual_u.copy()

    # find_peaks does not handle NaNs reliably.
    residual_for_peaks[
        ~np.isfinite(residual_for_peaks)
    ] = 0.0

    event_indices, properties = find_peaks(
        -residual_for_peaks,
        prominence=prominence,
        distance=distance
    )

    # ========================================================
    # 5. Separate physical/interface-associated events from
    #    candidate u_i anomalies
    # ========================================================

    anomaly_indices = np.array(
        [
            i for i in event_indices
            if (
                np.isfinite(Z_h[i])
                and abs(Z_h[i]) < Z_h_threshold
            )
        ],
        dtype=int
    )

    ignored_indices = np.array(
        [
            i for i in event_indices
            if (
                np.isfinite(Z_h[i])
                and abs(Z_h[i]) >= Z_h_threshold
            )
        ],
        dtype=int
    )

    # ========================================================
    # 6. Expand each anomaly into its contaminated neighborhood
    # ========================================================

    bad_mask = np.zeros(nx, dtype=bool)

    for i in anomaly_indices:

        # Anomaly center is always bad
        bad_mask[i] = True

        # ----------------------------------------------------
        # Expand left
        # ----------------------------------------------------

        j = i - 1

        while (
            j >= 0
            and np.isfinite(Z_u[j])
            and abs(Z_u[j]) >= Z_u_region_threshold
        ):
            bad_mask[j] = True
            j -= 1

        # ----------------------------------------------------
        # Expand right
        # ----------------------------------------------------

        j = i + 1

        while (
            j < nx
            and np.isfinite(Z_u[j])
            and abs(Z_u[j]) >= Z_u_region_threshold
        ):
            bad_mask[j] = True
            j += 1

    bad_indices = np.where(bad_mask)[0]

    # ========================================================
    # 7. Replace contaminated regions
    # ========================================================

    u_i_corrected = u_i_raw.copy()

    if np.any(bad_mask):

        good_mask = (
            ~bad_mask
            & np.isfinite(u_i_raw)
        )

        if np.count_nonzero(good_mask) < 2:
            raise ValueError(
                "Not enough valid u_i values remain to "
                "interpolate across anomalous regions."
            )

        u_i_corrected[bad_mask] = np.interp(
            x[bad_mask],
            x[good_mask],
            u_i_raw[good_mask]
        )

    # ========================================================
    # 8. Diagnostics
    # ========================================================

    diagnostics = {
        "u_i_raw": u_i_raw,
        "u_i_corrected": u_i_corrected,

        "residual_u": residual_u,
        "residual_h": residual_h,

        "Z_u": Z_u,
        "Z_h": Z_h,

        "sigma_u": sigma_u,
        "sigma_h": sigma_h,

        "event_indices": event_indices,
        "anomaly_indices": anomaly_indices,
        "ignored_indices": ignored_indices,

        "bad_mask": bad_mask,
        "bad_indices": bad_indices,

        "peak_properties": properties,
    }

    if return_diagnostics:
        return u_i_corrected, diagnostics

    return u_i_corrected

def scalar_value(ds): return float(np.asarray(ds[...]).squeeze())

def get_param(h5,name,default=None):
    path=f"Parameters/{name}"
    if path in h5: return scalar_value(h5[path])
    if default is not None: return default
    raise KeyError(f"Missing {path}")

def copy_static_parameters(source,dest):
    if "Parameters" not in source: return
    skip={"time","time_minus","t","tm","dt"}
    def rec(src,dst):
        for n,it in src.items():
            if n in skip: continue
            if isinstance(it,h5py.Dataset):
                dst.create_dataset(n,data=it[...])
                for k,v in it.attrs.items(): dst[n].attrs[k]=v
            else: rec(it,dst.create_group(n))
    rec(source["Parameters"],dest)

def validate_snapshot(h5,path):
    req=("Geometry/Points","Topology","Cells/f","Cells/f_minus","Cells/u.x","Cells/u_minus.x","Cells/p","Cells/p_minus","Parameters/time","Parameters/time_minus")
    miss=[r for r in req if r not in h5]
    if miss: raise KeyError(f"{path} missing: {miss}")

def closure_set(x,h,q,hx,hxx,hxxx,qxx,M,ui,uix,dudy,Ip,Re,Ca,beta,Lambda):
    We = Re * Ca
    Fr2 = Re * np.sin(beta) / Lambda

    # Integrated momentum fluxes
    M_lav = q**2 / h + Lambda**2 * h**5 / 45.0

    # Choose SG settings from the DNS momentum flux.
    # Apply the same differentiation operator to both fluxes.
    m_M, p_M, _, _ = best_savgol_filter(
        x, M,
        range(11, 42, 2),
        range(2, 4),
        metric="rmse",
        power=4,
    )
    dx = np.diff(x)[0]

    Mx_dns = savgol_filter(
        M, m_M, p_M, deriv=1, delta=dx
    )
    Mx_lav = savgol_filter(
        M_lav, m_M, p_M, deriv=1, delta=dx
    )

    Tw_dns = dudy / Re
    Tw_lav = (
        3.0 * q / (Re * h**2)
        - Lambda**2 * h**4 * hx / 15.0
    )
    P_lav = h*hxxx/We - (np.cos(beta)/Fr2)*h*hx + qxx/Re
    D_dns = (qxx - 2.0*hx*uix - hxx*ui) / Re
    D_lav = qxx / Re

    return {
        "delta_M_ML": M - M_lav,
        "delta_Mx_ML": Mx_dns - Mx_lav,
        "delta_Tw_ML": Tw_dns - Tw_lav,
        "delta_Px_ML": (-Ip) - P_lav,
        "delta_Dx_ML": D_dns - D_lav,
    }

def process_state(xm,ym,idx,snap,suffix,Re,Ca,beta,Lambda,lower,px_sigma=PX_SIGMA_DEFAULT):
    sf="_minus" if suffix else ""
    f=scalar_to_matrix(snap[f"Cells/f{sf}"][...],idx,xm.shape)
    u=vector_x_to_matrix(snap[f"Cells/u{sf}.x"][...],idx,xm.shape)
    p=scalar_to_matrix(snap[f"Cells/p{sf}"][...],idx,xm.shape)
    x=xm[0,:]
    _,h=depth_integral(xm,ym,f,lower)
    _,q=depth_integral(xm,ym,u*f,lower)
    _,M=depth_integral(xm,ym,u*u*f,lower)
    hx,hxx,hxxx=sg_derivatives(x,h,3,min_poly=3)
    qx,qxx=sg_derivatives(x,q,2,min_poly=2)
    ui=compute_ui(u,ym,h,x,k=5,prominence=0.05,distance=10,Z_h_threshold=3.0,Z_u_region_threshold=2.0,return_diagnostics=False)
    # Select SG settings from ui itself and evaluate its first derivative directly.
    mui,pui,_,_=best_savgol_filter(x,ui,range(11,42,2),range(2,4),metric="rmse",power=4)
    dx=np.diff(x)[0]; uix=savgol_filter(ui,mui,pui,deriv=1,delta=dx)
    _,dudy=compute_wall_gradient(xm,ym,u)
    px=compute_px(x,p)
    Ip=integrate_to_height(ym,px,h,lower)
    out={"h":h,"q":q,"h_x":hx,"q_x":qx,"h_xx":hxx,"q_xx":qxx,"h_xxx":hxxx,
         "int_u2_dy":M,"u_interface":ui,"u_interface_x":uix,"int_px_dy":Ip,"dudy_wall":dudy}
    out.update(closure_set(x,h,q,hx,hxx,hxxx,qxx,M,ui,uix,dudy,Ip,Re,Ca,beta,Lambda))
    out.update(px_v2_fields(x,ym,p,h,q,Re,Ca,beta,Lambda,px_sigma))
    return out

def state_v2(xm,ym,idx,snap,suffix,Re,Ca,beta,Lambda,lower,px_sigma):
    """Only what ΔP_x v2 needs (append mode): h, q and the v2 fields."""
    sf="_minus" if suffix else ""
    f=scalar_to_matrix(snap[f"Cells/f{sf}"][...],idx,xm.shape)
    u=vector_x_to_matrix(snap[f"Cells/u{sf}.x"][...],idx,xm.shape)
    p=scalar_to_matrix(snap[f"Cells/p{sf}"][...],idx,xm.shape)
    x=xm[0,:]
    _,h=depth_integral(xm,ym,f,lower)
    _,q=depth_integral(xm,ym,u*f,lower)
    out={"h":h}
    out.update(px_v2_fields(x,ym,p,h,q,Re,Ca,beta,Lambda,px_sigma))
    return out

def append_v2(config):
    """Add the ΔP_x v2 datasets to an existing closure file, for exactly its stored times.
    Existing datasets are never modified; refuses if any v2 dataset already exists.
    Optional config "t_max": times > t_max are filled with NaN and their snapshots are not
    read (for runs whose later snapshots were overwritten after the original extraction)."""
    sd=Path(config["snapshot_dir"]).expanduser(); op=Path(config["output_file"]).expanduser()
    sig=float(config.get("px_filter_width",PX_SIGMA_DEFAULT)); lower=float(config.get("lower_boundary",0.0))
    with h5py.File(op,"r+") as out:
        present=[n for n in V2_FIELDS if n in out or n+"_m" in out]
        if present: raise FileExistsError(f"{op} already has {present}; not overwriting")
        t_all=out["t"][:].ravel(); nx=int(out["x"].shape[0]); nt=t_all.size
        t_max=float(config.get("t_max",np.inf))
        paths=[sd/format_snapshot_filename(v,str(config.get("snapshot_prefix","snapshot_")),str(config.get("snapshot_suffix",".h5")),int(config.get("filename_width",6)),int(config.get("filename_decimals",0))) for v in t_all]
        use=[it for it,v in enumerate(t_all) if v<=t_max+1e-9]
        miss=[paths[it] for it in use if not paths[it].is_file()]
        if miss: raise FileNotFoundError("Missing snapshots:\n"+"\n".join(map(str,miss[:20])))
        if len(use)<nt: print(f"t_max={t_max:g}: {nt-len(use)} of {nt} times left as NaN")
        with h5py.File(paths[use[0]],"r") as first:
            cc=compute_cell_centers(first["Geometry/Points"][...],first["Topology"][...]); xm,ym,idx=build_cartesian_cell_grid(cc)
            Re=get_param(first,"Re"); Ca=get_param(first,"Ca"); beta=get_param(first,"angle"); Lambda=get_param(first,"Lambda",3.0)
        if not np.allclose(xm[0,:],out["x"][:].ravel()): raise ValueError("snapshot x grid differs from the closure file")
        res={n:np.full((nx,nt),np.nan) for n in V2_FIELDS}; resm={n:np.full((nx,nt),np.nan) for n in V2_FIELDS}
        hdev=0.0
        for it in use:
            path=paths[it]
            print(f"[{it+1}/{nt}] {path.name}",flush=True)
            with h5py.File(path,"r") as s:
                t=scalar_value(s["Parameters/time"])
                if abs(t-out["t"][it])>1e-8: raise ValueError(f"{path}: time {t} != stored {out['t'][it]}")
                cur=state_v2(xm,ym,idx,s,"",Re,Ca,beta,Lambda,lower,sig)
                prv=state_v2(xm,ym,idx,s,"m",Re,Ca,beta,Lambda,lower,sig)
            hdev=max(hdev,np.max(np.abs(cur["h"]-out["h"][:,it])),np.max(np.abs(prv["h"]-out["h_m"][:,it])))
            for n in V2_FIELDS: res[n][:,it]=cur[n]; resm[n][:,it]=prv[n]
        if hdev>1e-10: raise ValueError(f"recomputed h differs from stored h by {hdev:g}; not writing")
        comp=config.get("compression","gzip"); comp=None if comp in (None,"none","None","") else comp
        kw=dict(dtype=np.float64,chunks=(min(nx,1024),1))
        if comp: kw["compression"]=comp; kw["compression_opts"]=int(config.get("compression_level",4)) if comp=="gzip" else None
        for n in V2_FIELDS:
            out.create_dataset(n,data=res[n],**kw); out.create_dataset(n+"_m",data=resm[n],**kw)
        out.attrs["delta_Px_ML_v2"]=V2_DESCRIPTION; out.attrs["px_filter_width"]=sig
        if np.isfinite(t_max): out.attrs["delta_Px_ML_v2_t_max"]=t_max
    print(f"Appended {len(V2_FIELDS)} v2 fields (+_m) to {op}; max |h - stored h| = {hdev:.3g}")
    return op

def create_output_datasets(out,nx,nt,compression,level):
    base = (
            "h", "q", "h_x", "q_x", "h_xx", "q_xx", "h_xxx",
            "int_u2_dy", "u_interface", "u_interface_x",
            "int_px_dy", "dudy_wall",
            "delta_M_ML", "delta_Mx_ML",
            "delta_Tw_ML", "delta_Px_ML", "delta_Dx_ML",
            ) + V2_FIELDS

    kw=dict(shape=(nx,nt),dtype=np.float64,chunks=(min(nx,1024),1))
    if compression:
        kw["compression"]=compression
        if compression=="gzip": kw["compression_opts"]=level
    ds={}
    for n in base:
        ds[n]=out.create_dataset(n,**kw); ds[n+"_m"]=out.create_dataset(n+"_m",**kw)
    for n in ("t","tm","dt"): ds[n]=out.create_dataset(n,shape=(nt,),dtype=np.float64)
    return ds

def extract_snapshots(config):
    sd=Path(config["snapshot_dir"]).expanduser(); op=Path(config["output_file"]).expanduser()
    vals=build_snapshot_values(float(config["start_time"]),float(config["end_time"]),float(config["delta_time"]))
    paths=[sd/format_snapshot_filename(v,str(config.get("snapshot_prefix","snapshot_")),str(config.get("snapshot_suffix",".h5")),int(config.get("filename_width",6)),int(config.get("filename_decimals",0))) for v in vals]
    miss=[p for p in paths if not p.is_file()]
    if miss: raise FileNotFoundError("Missing snapshots:\n"+"\n".join(map(str,miss[:20])))
    with h5py.File(paths[0],"r") as first:
        validate_snapshot(first,paths[0]); cc=compute_cell_centers(first["Geometry/Points"][...],first["Topology"][...]); xm,ym,idx=build_cartesian_cell_grid(cc)
        Re=get_param(first,"Re"); Ca=get_param(first,"Ca"); beta=get_param(first,"angle"); Lambda=get_param(first,"Lambda",3.0)
    x=xm[0,:]; nx=len(x); nt=len(paths); lower=float(config.get("lower_boundary",0.0)); comp=config.get("compression","gzip"); comp=None if comp in (None,"none","None","") else comp; lev=int(config.get("compression_level",4))
    sig=float(config.get("px_filter_width",PX_SIGMA_DEFAULT))
    if op.exists(): raise FileExistsError(f"{op} exists; use mode append_v2 or remove it")
    op.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(op,"w") as out:
        out.attrs["description"]="DNS thin-film closure variables and Lavalle discrepancy closures"
        out.attrs["matrix_layout"]="x_by_time"; out.attrs["nx"]=nx; out.attrs["nt"]=nt
        out.attrs["x_derivative_method"]="Savitzky-Golay; RMSE-selected per state; filter parameters not stored"
        out.attrs["delta_Px_ML_v2"]=V2_DESCRIPTION; out.attrs["px_filter_width"]=sig
        out.create_dataset("x",data=x); ds=create_output_datasets(out,nx,nt,comp,lev)
        with h5py.File(paths[0],"r") as first:
            pg=out.create_group("parameters"); copy_static_parameters(first,pg)
            # Store derived nondimensional parameters used by closures when absent in source.
            if "Lambda" not in pg: pg.create_dataset("Lambda",data=Lambda)
            if "We" not in pg: pg.create_dataset("We",data=Re*Ca)
            if "Fr2" not in pg: pg.create_dataset("Fr2",data=Re*np.sin(beta)/Lambda)
        for it,path in enumerate(paths):
            print(f"[{it+1}/{nt}] {path.name}")
            with h5py.File(path,"r") as s:
                validate_snapshot(s,path); t=scalar_value(s["Parameters/time"]); tm=scalar_value(s["Parameters/time_minus"]); dt=t-tm
                if dt<=0: raise ValueError(f"Non-positive dt in {path}: time={t}, time_minus={tm}")
                cur=process_state(xm,ym,idx,s,"",Re,Ca,beta,Lambda,lower,sig)
                prv=process_state(xm,ym,idx,s,"m",Re,Ca,beta,Lambda,lower,sig)
                for n,a in cur.items(): ds[n][:,it]=a
                for n,a in prv.items(): ds[n+"_m"][:,it]=a
                ds["t"][it]=t; ds["tm"][it]=tm; ds["dt"][it]=dt
        out.flush()
    print(f"Wrote {op}\nField matrix shape: ({nx}, {nt})")
    return op

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("config",type=Path); a=ap.parse_args(); cfg=load_config(a.config)
    # mode "append_v2": add the ΔP_x v2 fields to an existing closure file; default: full extraction
    (append_v2 if cfg.get("mode")=="append_v2" else extract_snapshots)(cfg)
if __name__=="__main__": main()
