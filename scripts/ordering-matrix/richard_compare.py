# Compare the 3-moment closure with the Richard-type (Teshukov / shear-shallow-water) structure
from common import *
import numpy as np
phi=3*(eta-eta**2/2)
Sa=sp.expand(eta*(eta-2)*(4-10*eta+5*eta**2)); Sb=sp.expand(eta*(eta-2)*(-4-2*eta+29*eta**2-28*eta**3+7*eta**4))
fams={}
for nm,S0 in [('quartic',Sa),('sextic',Sb)]:
    psi=sp.expand(S0/sp.diff(S0,eta).subs(eta,0)); f=phi+s*psi
    f2=sp.expand(sp.integrate(f**2,(eta,0,1))); f3=sp.expand(sp.integrate(f**3,(eta,0,1))); fD=sp.expand(sp.integrate(sp.diff(f,eta)**2,(eta,0,1)))
    fams[nm]=(f2,f3,fD)
    print(nm,' skewness sigma(s)=f3-3f2+2 =',sp.expand(f3-3*f2+2))
fams['SSW (skewness dropped, quartic f2,fD)']=(fams['quartic'][0],sp.expand(3*fams['quartic'][0]-2),fams['quartic'][2])

Re_,cb_,Ca_=Re,cb,Ca
for nm,(f2,f3,fD) in fams.items():
    # 1) slaved small-Re limit: q = h^3 + q1, s = s1 (both first order); leading-order transport
    q1,s1=sp.symbols('q1 s1')
    hq=h**3
    A2_0=hq**2/h*f2.subs(s,0); A3_0=hq**3/h**2*f3.subs(s,0)
    ht=-dx(hq)
    LHS1=sp.expand(dt(hq,ht,0)+dx(A2_0)); LHS2=sp.expand(dt(A2_0,ht,0)+dx(A3_0))
    Pix=3*cb/Re*H[1]-H[3]/(Re*Ca)
    q_=hq+q1; s_=s1
    tw=q_/h**2*(3+s_); D=q_**2/h**3*fD.subs(s,s_)
    R1=(3*h-tw)/Re-h*Pix; R2=2/Re*(3*q_-D)-2*hq*Pix
    # linearise sources in (q1,s1)
    R1l=sp.expand(sp.series(R1.subs({q1:sp.Symbol('e')*q1,s1:sp.Symbol('e')*s1}),sp.Symbol('e'),0,2).removeO().subs(sp.Symbol('e'),1))
    R2l=sp.expand(sp.series(R2.subs({q1:sp.Symbol('e')*q1,s1:sp.Symbol('e')*s1}),sp.Symbol('e'),0,2).removeO().subs(sp.Symbol('e'),1))
    sol=sp.solve([sp.Eq(LHS1,R1l),sp.Eq(LHS2,R2l)],[q1,s1],dict=True)[0]
    print('\n==',nm); print(' slaved q1 =',sp.collect(sp.expand(sol[q1]),[H[1],H[3]],sp.factor))
    print(' slaved tau_w - 3q/h^2 =',sp.collect(sp.expand(h*sol[s1]),[H[1],H[3]],sp.factor))
    # 2) homogeneous viscous relaxation rates (units 1/(Re h^2)), h=1
    qq,ss=sp.symbols('qq ss')
    F1=3-qq*(3+ss); F2=2*(3*qq-qq**2*fD.subs(s,ss))
    A2=qq**2*f2.subs(s,ss)
    M=sp.Matrix([[1,0],[sp.diff(A2,qq),sp.diff(A2,ss)]]).subs({qq:1,ss:0})
    Jsrc=sp.Matrix([[sp.diff(F1,qq),sp.diff(F1,ss)],[sp.diff(F2,qq),sp.diff(F2,ss)]]).subs({qq:1,ss:0})
    ev=sorted([float(-e) for e in (M.inv()*Jsrc).eigenvals(multiple=True)])
    print(' decay rates:',[round(e,4) for e in ev],'  exact (2n+1)^2 pi^2/4:',[round(np.pi**2/4,4),round(9*np.pi**2/4,4)])
    # 3) implied Richard relaxation constant B (phi-equation source ~ -(B/(Re h))(phi-phi_eq))
    c1=sp.diff(f2,s).subs(s,0); print(' implied B =',sp.nsimplify(-2/c1))
# SSW hyperbolicity check
hh,qq,AA=sp.symbols('hh qq AA',positive=True)
A3=qq**3/hh**2+3*qq/hh*(AA-qq**2/hh)
J=sp.Matrix([[0,1,0],[0,0,1],[sp.diff(A3,v) for v in (hh,qq,AA)]])
print('\nSSW eigenvalues:',[sp.simplify(e) for e in J.eigenvals()])
