from common import *
import pickle, numpy as np
fam=pickle.load(open('fam.pkl','rb'))
ss=sp.Symbol('ss')
for name,d in fam.items():
    f2=sp.lambdify(ss,d['A2'].subs(s,ss)); 
    f2s=d['A2'].subs(s,ss); f3s=d['A3'].subs(s,ss)
    # at h=q=1: A3 = f3(s(A2)); dA3/dh etc via scaling: A3 = q^3/h^2 f3(s), A2 = q^2/h f2(s)
    hh,qq,AA=sp.symbols('hh qq AA',positive=True)
    G=AA-qq**2/hh*f2s; A3=qq**3/hh**2*f3s
    row=[sp.diff(A3,v)-sp.diff(A3,ss)*sp.diff(G,v)/sp.diff(G,ss) for v in (hh,qq,AA)]
    rowf=sp.lambdify((ss,),[r.subs({hh:1,qq:1,AA:f2s}) for r in row])
    smax=float(sp.solve(sp.diff(f2s,ss),ss)[0])
    bad=[]
    for sv in np.linspace(-15,smax-1e-3,3001):
        J=np.array([[0,1,0],[0,0,1],rowf(sv)],dtype=float)
        if np.max(np.abs(np.linalg.eigvals(J).imag))>1e-9: bad.append(sv)
    print(name,'branch s <',round(smax,3),' non-hyperbolic s-range:',(round(min(bad),3),round(max(bad),3)) if bad else None)
