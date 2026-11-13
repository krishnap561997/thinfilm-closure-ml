import sympy as sp
N=9
z,eta=sp.symbols('z eta',real=True)
Re,cb,Ca,We=sp.symbols('Re cotb Ca We',positive=True)
H=sp.symbols('h0:%d'%N,real=True); Q=sp.symbols('q0:%d'%N,real=True); S=sp.symbols('s0:%d'%N,real=True)
h,q,s=H[0],Q[0],S[0]
def dx(f):
    return sp.expand(sum(sp.diff(f,X[k])*X[k+1] for X in (H,Q,S) for k in range(N-1)))
def dxn(f,n):
    for _ in range(n): f=dx(f)
    return f
def dt(f,ht,qt,st=0):
    r=0
    for k in range(N-3):
        r+=sp.diff(f,H[k])*dxn(ht,k)+sp.diff(f,Q[k])*dxn(qt,k)
        if st!=0: r+=sp.diff(f,S[k])*dxn(st,k)
    return sp.expand(r)
def solve_uzz(F):
    t,ss=sp.symbols('t ss',real=True)
    inner=sp.integrate(F.subs(z,t),(t,ss,h))
    return sp.expand(-sp.integrate(inner,(ss,0,z)))
def w_of(u):
    return sp.expand(-sp.integrate(dx(u),(z,0,z)))
